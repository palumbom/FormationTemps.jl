using Revise
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Printf, Statistics
using DelimitedFiles

# output directory
datadir = joinpath(FT.moddir, "benchmarks", "data")
!isdir(datadir) && mkpath(datadir)

# ── setup ──────────────────────────────────────────────────────────────────────
use_gpu = FT.GPU_DEFAULT
println("GPU available: ", use_gpu)

# solar linelist — ~10 Fe I lines near 6300 A
linelist_full = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist_full = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist_full]
specs = [string(l.species) for l in linelist_full]
linelist_fe = linelist_full[specs .== "Fe I"]
wls_all = [l.wl * 1e8 for l in linelist_fe]
idx_start = findfirst(x -> x >= 6298.0, wls_all)
idx_end = findfirst(x -> x >= 6304.0, wls_all)
linelist = linelist_fe[idx_start:idx_end]
println("Linelist: ", length(linelist), " lines from ",
        @sprintf("%.2f", linelist[1].wl * 1e8), " to ",
        @sprintf("%.2f", linelist[end].wl * 1e8), " A")

# stellar params
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3500.0
ξ = 850.0
Nϕ = 128
Δλ = 0.01

star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# wavelength grid
wls = [l.wl * 1e8 for l in linelist]
buffer = 2.0
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=Δλ)
Nλ = length(λs_korg)

# atmosphere
A_X = star.A_X
atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(Teff, logg, A_X))
zs = atm_cpu.zs
Ts = atm_cpu.Ts
Natm = length(zs)

# absorption coefficients (computed once, shared by all tiles)
αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_cpu, A_X;
                  α_ref_out=α_ref, ne_warn_thresh=Inf)

# stellar grid
μs_grid, dA_grid, z_rot_grid = FT.calc_stellar_grid_cpu(star.ρstar, star.istar, star.vsini, Nϕ)
idx = findall(x -> x > 0.0, μs_grid)
μs_cpu = μs_grid[idx]
dA_cpu = dA_grid[idx]
z_rot_cpu = z_rot_grid[idx]
if iszero(vsini)
    z_rot_cpu .= 0.0
end
n_tiles = length(μs_cpu)
unique_μs = length(unique(round.(μs_cpu, sigdigits=10)))

println()
println("Problem size:")
println("  Natm     = ", Natm)
println("  Nλ       = ", Nλ)
println("  Nϕ       = ", Nϕ)
println("  N tiles  = ", n_tiles)
println("  Unique μ = ", unique_μs)
println("  Threads  = ", Threads.nthreads())
println()

# ── CPU per-tile benchmark ────────────────────────────────────────────────────
function benchmark_cpu_pertile(αs, αs_cont, atm_cpu, λs_korg, star, μs_cpu, z_rot_cpu;
                               n_repeat=5)
    T = Float64
    Natm = length(atm_cpu.zs)
    Nλ = length(λs_korg)
    Ts = atm_cpu.Ts

    # tau dispatcher
    if isempty(atm_cpu.τs)
        _calc_tau_cpu! = (μ_i, αs_in, τs_out) -> FT.calc_tau_bezier_cpu!(μ_i, atm_cpu.zs, αs_in, τs_out)
    else
        _calc_tau_cpu! = (μ_i, αs_in, τs_out) -> FT.calc_tau_anchored_cpu!(μ_i, atm_cpu.τs, α_ref, αs_in, τs_out)
    end

    σ_v = fill(star.ξ, Natm)
    μ_v_rot = zeros(T, Natm)
    τs_int = zeros(T, Natm, Nλ)
    cfunc_int = zeros(T, Natm - 1, Nλ)

    t_micro = zeros(n_repeat)
    t_tau = zeros(n_repeat)
    t_cfunc = zeros(n_repeat)
    t_macro = zeros(n_repeat)

    for r in 1:n_repeat
        μ_tile = μs_cpu[1]
        μ_v_rot .= z_rot_cpu[1] .* FT.c_ms

        t_micro[r] = @elapsed begin
            αs_broad_i = FT.convolve_wavelength_axis(λs_korg, αs, μ_v_rot, σ_v)
        end
        t_tau[r] = @elapsed _calc_tau_cpu!(μ_tile, αs_broad_i, τs_int)
        t_cfunc[r] = @elapsed FT.calc_intensity_cfunc_cpu!(cfunc_int, Ts, λs_korg, τs_int)
        cfunc_dt_int = cfunc_int .* diff(τs_int, dims=1)
        t_macro[r] = @elapsed FT.convolve_rt_macro(λs_korg, cfunc_dt_int, star.ζ, μ_tile)
    end

    return (micro=median(t_micro), tau=median(t_tau),
            cfunc=median(t_cfunc), macro_conv=median(t_macro))
end

# ── GPU batched kernel benchmark ──────────────────────────────────────────────
function benchmark_gpu_batched(αs, star, λs_korg, μs_cpu, z_rot_cpu;
                               n_repeat=10, B=8)
    T = Float64
    A_X = star.A_X
    atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(star.Teff, star.logg, A_X))
    Natm = length(atm_gpu.zs)
    Nλ = length(λs_korg)
    Natm1 = Natm - 1
    Npad = 512

    α_ref_gpu = zeros(Natm)
    FT.compute_alpha!(αs, zeros(Natm, Nλ), Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                      α_ref_out=α_ref_gpu, ne_warn_thresh=Inf)

    # GPU memory
    λs_gpu = CuArray(collect(λs_korg))
    σ_v = CUDA.zeros(T, Natm) .+ star.ξ
    log_τ_ref = CuArray{T}(log.(atm_gpu.τs))
    ifactor_base = CuArray{T}(atm_gpu.τs ./ α_ref_gpu)
    Ts_gpu = CuArray{T}(atm_gpu.Ts)

    bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)
    cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm1, Npad)

    # batch parameters
    μ_tiles = CuArray{T}(μs_cpu[1:B])
    dA_tiles = CuArray{T}(ones(B) * 0.001)
    μ_v_batch_cpu = zeros(T, B * Natm)
    for bi in 1:B
        v = z_rot_cpu[min(bi, length(z_rot_cpu))] * FT.c_ms
        for k in 1:Natm
            μ_v_batch_cpu[(bi-1)*Natm+k] = v
        end
    end
    μ_v_batch = CuArray{T}(μ_v_batch_cpu)

    # working arrays
    τs_batch = CUDA.zeros(T, B * Natm, Nλ)
    cfdt_batch = CUDA.zeros(T, B * Natm1, Nλ)
    flux_acc = CUDA.zeros(T, Nλ)
    cfunc_acc = CUDA.zeros(T, Natm1, Nλ)

    # prime signal cache
    bcmem.signal_cached = false
    FT.convolve_wavelength_axis_batched!(bcmem, collect(λs_korg), αs,
        CUDA.zeros(T, Natm), σ_v, 1)
    bcmem.signal_cached = true

    # precompute macro kernel
    macro_kft = FT.precompute_rt_macro_kernel_ft(cmem_mac, collect(λs_korg), star.ζ, μs_cpu[1])
    CUDA.synchronize()

    # timing
    t_micro = zeros(n_repeat)
    t_tau = zeros(n_repeat)
    t_cfunc = zeros(n_repeat)
    t_macro = zeros(n_repeat)

    for r in 1:n_repeat
        CUDA.synchronize()
        t_micro[r] = CUDA.@elapsed begin
            αs_conv = FT.convolve_wavelength_axis_batched!(bcmem, collect(λs_korg), αs,
                μ_v_batch, σ_v, B)
        end

        t_tau[r] = CUDA.@elapsed begin
            FT.calc_tau_anchored_batched!(μ_tiles, log_τ_ref, ifactor_base,
                αs_conv, τs_batch, Natm, B)
        end

        t_cfunc[r] = CUDA.@elapsed begin
            FT.calc_intensity_cfunc_dt_batched!(cfdt_batch, τs_batch,
                Ts_gpu, λs_gpu, Natm, B)
        end

        t_macro[r] = CUDA.@elapsed begin
            for bi in 1:B
                tile_cfdt = @view cfdt_batch[(bi-1)*Natm1+1 : bi*Natm1, :]
                FT.convolve_rt_macro_gpu_cached(cmem_mac, tile_cfdt, macro_kft)
            end
        end
    end

    return (micro=median(t_micro), tau=median(t_tau),
            cfunc=median(t_cfunc), macro_conv=median(t_macro))
end

# ── end-to-end benchmark ──────────────────────────────────────────────────────
function benchmark_end_to_end(star, linelist; Δλ, Nϕ, use_gpu, n_repeat=3,
                              gpu_precision=Float64)
    # warmup
    calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=16,
                        use_gpu=use_gpu, gpu_precision=gpu_precision,
                        ne_warn_thresh=Inf, showprogress=false)

    times = zeros(n_repeat)
    local result
    for r in 1:n_repeat
        if use_gpu; CUDA.synchronize(); end
        times[r] = @elapsed begin
            result = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                         use_gpu=use_gpu, gpu_precision=gpu_precision,
                                         ne_warn_thresh=Inf, showprogress=false)
        end
        if use_gpu; CUDA.synchronize(); end
    end
    return median(times), result
end

# ── run benchmarks ─────────────────────────────────────────────────────────────
println("="^70)
println("DISK INTEGRATION BENCHMARK")
println("="^70)
println()

# CPU per-tile
println("─"^40)
println("CPU BENCHMARK (per-tile)")
println("─"^40)
cpu_times = benchmark_cpu_pertile(copy(αs), copy(αs_cont), atm_cpu, λs_korg, star,
                                  μs_cpu, z_rot_cpu)
@printf("  Microturbulence: %8.3f ms\n", cpu_times.micro * 1000)
@printf("  Tau integration: %8.3f ms\n", cpu_times.tau * 1000)
@printf("  Cfunc:           %8.3f ms\n", cpu_times.cfunc * 1000)
@printf("  Macro conv:      %8.3f ms\n", cpu_times.macro_conv * 1000)
cpu_total_pertile = sum(cpu_times) * 1000
@printf("  Total per tile:  %8.3f ms\n", cpu_total_pertile)
println()

# GPU batched kernels
gpu_times = nothing
if use_gpu
    B_bench = 8
    println("─"^40)
    println("GPU BENCHMARK (batched, B=$B_bench)")
    println("─"^40)
    gpu_times = benchmark_gpu_batched(copy(αs), star, λs_korg, μs_cpu, z_rot_cpu;
                                      B=B_bench)
    @printf("  Micro (batched):  %8.3f ms\n", gpu_times.micro * 1000)
    @printf("  Tau (batched):    %8.3f ms\n", gpu_times.tau * 1000)
    @printf("  Cfunc (batched):  %8.3f ms\n", gpu_times.cfunc * 1000)
    @printf("  Macro (per-tile): %8.3f ms\n", gpu_times.macro_conv * 1000)
    gpu_total = sum(gpu_times) * 1000
    @printf("  Total (B=%d):     %8.3f ms\n", B_bench, gpu_total)
    cpu_equiv = cpu_total_pertile * B_bench
    @printf("  CPU equiv (%d tiles): %8.3f ms  (%.1f× speedup)\n",
            B_bench, cpu_equiv, cpu_equiv / gpu_total)
    println()
end

# end-to-end
println("─"^40)
println("END-TO-END: calc_formation_temp")
println("─"^40)

t_cpu_e2e, result_cpu = benchmark_end_to_end(star, linelist; Δλ=Δλ, Nϕ=Nϕ, use_gpu=false)
@printf("CPU  (Nϕ=%d): %.2f s\n", Nϕ, t_cpu_e2e)

t_gpu64_e2e = NaN
t_gpu32_e2e = NaN
if use_gpu
    t_gpu64_e2e, result_gpu64 = benchmark_end_to_end(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                                      use_gpu=true, gpu_precision=Float64)
    @printf("GPU Float64 (Nϕ=%d): %.2f s  (%.1f× vs CPU)\n",
            Nϕ, t_gpu64_e2e, t_cpu_e2e / t_gpu64_e2e)

    t_gpu32_e2e, result_gpu32 = benchmark_end_to_end(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                                      use_gpu=true, gpu_precision=Float32)
    @printf("GPU Float32 (Nϕ=%d): %.2f s  (%.1f× vs CPU, %.2f× vs GPU64)\n",
            Nϕ, t_gpu32_e2e, t_cpu_e2e / t_gpu32_e2e, t_gpu64_e2e / t_gpu32_e2e)

    println()
    max_flux_diff_64 = maximum(abs.(result_cpu.flux .- result_gpu64.flux))
    max_flux_diff_32 = maximum(abs.(result_cpu.flux .- Float64.(result_gpu32.flux)))
    @printf("Max |flux diff| CPU vs GPU64: %.2e\n", max_flux_diff_64)
    @printf("Max |flux diff| CPU vs GPU32: %.2e\n", max_flux_diff_32)
end
println()

# ── save data ─────────────────────────────────────────────────────────────────
# per-tile/batch step timings (ms)
steps = ["microturbulence", "tau", "cfunc", "macro"]
cpu_vals = [cpu_times.micro, cpu_times.tau, cpu_times.cfunc, cpu_times.macro_conv] .* 1000.0

if use_gpu && gpu_times !== nothing
    gpu_vals = [gpu_times.micro, gpu_times.tau, gpu_times.cfunc, gpu_times.macro_conv] .* 1000.0
else
    gpu_vals = fill(NaN, 4)
end

open(joinpath(datadir, "pertile_timings.csv"), "w") do io
    println(io, "step,cpu_ms,gpu_ms")
    for i in eachindex(steps)
        @printf(io, "%s,%.4f,%.4f\n", steps[i], cpu_vals[i], gpu_vals[i])
    end
end

# end-to-end timings
open(joinpath(datadir, "e2e_timings.csv"), "w") do io
    println(io, "backend,precision,time_s,Nphi,Natm,Nlambda,Ntiles,threads")
    @printf(io, "cpu,float64,%.4f,%d,%d,%d,%d,%d\n", t_cpu_e2e, Nϕ, Natm, Nλ, n_tiles, Threads.nthreads())
    if use_gpu
        @printf(io, "gpu,float64,%.4f,%d,%d,%d,%d,1\n", t_gpu64_e2e, Nϕ, Natm, Nλ, n_tiles)
        @printf(io, "gpu,float32,%.4f,%d,%d,%d,%d,1\n", t_gpu32_e2e, Nϕ, Natm, Nλ, n_tiles)
    end
end

println("Data written to: ", datadir)

println()
println("="^70)
println("DONE")
println("="^70)
