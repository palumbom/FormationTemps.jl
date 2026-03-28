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
                  α_ref_out=α_ref, vmic_ref_cms=ξ * 100.0, ne_warn_thresh=Inf)

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

# ── CPU benchmark ──────────────────────────────────────────────────────────────
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

    # collect timings over n_repeat passes
    t_micro = zeros(n_repeat)
    t_tau = zeros(n_repeat)
    t_cfunc = zeros(n_repeat)
    t_macro = zeros(n_repeat)

    for r in 1:n_repeat
        μ_tile = μs_cpu[1]
        μ_v_rot .= z_rot_cpu[1] .* FT.c_ms

        t_micro[r] = @elapsed FT.convolve_wavelength_axis(λs_korg, αs, μ_v_rot, σ_v)
        αs_broad_i = FT.convolve_wavelength_axis(λs_korg, αs, μ_v_rot, σ_v)
        t_tau[r] = @elapsed _calc_tau_cpu!(μ_tile, αs_broad_i, τs_int)
        t_cfunc[r] = @elapsed FT.calc_intensity_cfunc_cpu!(cfunc_int, Ts, λs_korg, τs_int)
        cfunc_dt_int = cfunc_int .* diff(τs_int, dims=1)
        t_macro[r] = @elapsed FT.convolve_rt_macro(λs_korg, cfunc_dt_int, star.ζ, μ_tile)
    end

    return (micro=median(t_micro), tau=median(t_tau),
            cfunc=median(t_cfunc), macro_conv=median(t_macro))
end

# ── GPU benchmark ──────────────────────────────────────────────────────────────
function benchmark_gpu_pertile(αs, αs_cont, star, λs_korg, μs_cpu, z_rot_cpu;
                               n_repeat=10)
    T = Float64
    A_X = star.A_X
    atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(star.Teff, star.logg, A_X))
    Natm = length(atm_gpu.zs)
    Nλ = length(λs_korg)
    Npad = 512

    α_ref_gpu = zeros(Natm)
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                      α_ref_out=α_ref_gpu, vmic_ref_cms=star.ξ * 100.0, ne_warn_thresh=Inf)

    gpu_mem = if isempty(atm_gpu.τs)
        FT.GPUMemory(λs_korg, atm_gpu)
    else
        FT.GPUMemory(λs_korg, atm_gpu, α_ref_gpu)
    end
    cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
    cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)

    σ_v = CUDA.zeros(T, Natm) .+ star.ξ
    μ_v_rot = CUDA.zeros(T, Natm)

    # warmup
    μ_v_rot .= z_rot_cpu[1] .* FT.c_ms
    cfunc_warmup = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[1], μ_v_rot, σ_v)
    FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, Array(cfunc_warmup.cfunc_dt), star.ζ, μs_cpu[1])
    CUDA.synchronize()

    # collect timings
    t_intensity = zeros(n_repeat)
    t_macro = zeros(n_repeat)

    for r in 1:n_repeat
        μ_tile = μs_cpu[1]
        μ_v_rot .= z_rot_cpu[1] .* FT.c_ms

        CUDA.synchronize()
        t_intensity[r] = CUDA.@elapsed begin
            cfunc_intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
        end

        tbc = Array(cfunc_intensity.cfunc_dt)
        CUDA.synchronize()
        t_macro[r] = CUDA.@elapsed begin
            FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, star.ζ, μ_tile)
        end
    end

    return (intensity=median(t_intensity), macro_conv=median(t_macro))
end

# ── end-to-end benchmark ──────────────────────────────────────────────────────
function benchmark_end_to_end(star, linelist; Δλ, Nϕ, use_gpu, n_repeat=3)
    # warmup
    calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=16,
                        use_gpu=use_gpu, ne_warn_thresh=Inf,
                        showprogress=false)

    times = zeros(n_repeat)
    local result
    for r in 1:n_repeat
        if use_gpu
            CUDA.synchronize()
        end
        times[r] = @elapsed begin
            result = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                         use_gpu=use_gpu, ne_warn_thresh=Inf,
                                         showprogress=false)
        end
        if use_gpu
            CUDA.synchronize()
        end
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

# GPU per-tile
gpu_times = nothing
if use_gpu
    println("─"^40)
    println("GPU BENCHMARK (per-tile)")
    println("─"^40)
    gpu_times = benchmark_gpu_pertile(copy(αs), copy(αs_cont), star, λs_korg,
                                      μs_cpu, z_rot_cpu)
    @printf("  Intensity (micro+tau+cfunc): %8.3f ms\n", gpu_times.intensity * 1000)
    @printf("  Macro conv:                  %8.3f ms\n", gpu_times.macro_conv * 1000)
    gpu_total_pertile = sum(gpu_times) * 1000
    @printf("  Total per tile:              %8.3f ms\n", gpu_total_pertile)
    println()
end

# end-to-end
println("─"^40)
println("END-TO-END: calc_formation_temp")
println("─"^40)

t_cpu_e2e, result_cpu = benchmark_end_to_end(star, linelist; Δλ=Δλ, Nϕ=Nϕ, use_gpu=false)
@printf("CPU  (Nϕ=%d): %.2f s\n", Nϕ, t_cpu_e2e)

t_gpu_e2e = NaN
if use_gpu
    t_gpu_e2e, result_gpu = benchmark_end_to_end(star, linelist; Δλ=Δλ, Nϕ=Nϕ, use_gpu=true)
    @printf("GPU  (Nϕ=%d): %.2f s\n", Nϕ, t_gpu_e2e)
    @printf("Speedup: %.1fx\n", t_cpu_e2e / t_gpu_e2e)

    max_flux_diff = maximum(abs.(result_cpu.flux .- result_gpu.flux))
    @printf("Max flux difference (CPU vs GPU): %.2e\n", max_flux_diff)
end
println()

# ── save data ─────────────────────────────────────────────────────────────────
# per-tile step timings (ms)
header_pertile = "step cpu_ms gpu_ms"
steps = ["microturbulence", "tau", "cfunc", "macro"]
cpu_vals = [cpu_times.micro, cpu_times.tau, cpu_times.cfunc, cpu_times.macro_conv] .* 1000.0

if use_gpu && gpu_times !== nothing
    # GPU combines micro+tau+cfunc into one kernel
    gpu_vals = [gpu_times.intensity, 0.0, 0.0, gpu_times.macro_conv] .* 1000.0
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
    println(io, "backend,time_s,Nphi,Natm,Nlambda,Ntiles,threads")
    @printf(io, "cpu,%.4f,%d,%d,%d,%d,%d\n", t_cpu_e2e, Nϕ, Natm, Nλ, n_tiles, Threads.nthreads())
    if use_gpu
        @printf(io, "gpu,%.4f,%d,%d,%d,%d,1\n", t_gpu_e2e, Nϕ, Natm, Nλ, n_tiles)
    end
end

println("Data written to: ", datadir)

println()
println("="^70)
println("DONE")
println("="^70)
