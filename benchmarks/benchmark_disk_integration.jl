using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using LinearAlgebra
using BenchmarkTools
using Printf, Statistics
using DelimitedFiles

# output directory
datadir = joinpath(FT.moddir, "benchmarks", "data")
!isdir(datadir) && mkpath(datadir)

# ── setup ──────────────────────────────────────────────────────────────────────
use_gpu = FT.GPU_DEFAULT
run_pertile = true
run_e2e = true
println("GPU available: ", use_gpu)

# Fe I 6301 & 6302 lines
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

# stellar params
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3500.0
ξ = 850.0
Nϕ = 128
Δλ = 0.005

star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# wavelength grid
wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
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

iqr(x) = quantile(x, 0.75) - quantile(x, 0.25)
iqr_ms(trial) = (quantile(trial.times, 0.75) - quantile(trial.times, 0.25)) / 2e6

# ── CPU per-tile benchmark ────────────────────────────────────────────────────
# mirrors the production tile loop in convenience.jl: uses CPUTileWorkspace with
# in-place convolutions, and times both total + continuum paths
function benchmark_cpu_pertile(αs, αs_cont, atm_cpu, λs_korg, star, μs_cpu, z_rot_cpu;
                               n_repeat=10)
    T = Float64
    Natm = length(atm_cpu.zs)
    Nλ = length(λs_korg)
    Ts = atm_cpu.Ts

    # tau dispatcher (same logic as production)
    if isempty(atm_cpu.τs)
        _calc_tau_cpu! = (μ_i, αs_in, τs_out) -> FT.calc_tau_bezier_cpu!(μ_i, atm_cpu.zs, αs_in, τs_out)
    else
        _calc_tau_cpu! = (μ_i, αs_in, τs_out) -> FT.calc_tau_anchored_cpu!(μ_i, atm_cpu.τs, α_ref, αs_in, τs_out)
    end

    σ_v_scalar = star.ξ
    ws = FT.CPUTileWorkspace(T, Natm, Nλ)

    t_micro = zeros(n_repeat)
    t_tau = zeros(n_repeat)
    t_cfunc = zeros(n_repeat)
    t_macro = zeros(n_repeat)

    for r in 1:n_repeat
        μ_tile = μs_cpu[1]
        μ_v_scalar = T(z_rot_cpu[1] * FT.c_ms)

        # total path (in-place, matching production scalar dispatch)
        t_micro[r] = @elapsed begin
            FT._convolve_micro_inplace!(ws.αs_broad, λs_korg, αs, μ_v_scalar, σ_v_scalar, ws)
        end
        t_tau[r] = @elapsed _calc_tau_cpu!(μ_tile, ws.αs_broad, ws.τs_int)
        t_cfunc[r] = @elapsed begin
            FT.calc_intensity_cfunc_cpu!(ws.cfunc_int, Ts, λs_korg, ws.τs_int)
            @views ws.cfunc_dt_int .= ws.cfunc_int .* (ws.τs_int[2:end, :] .- ws.τs_int[1:end-1, :])
        end
        t_macro[r] = @elapsed begin
            FT._convolve_macro_inplace!(ws.macro_out, λs_korg, ws.cfunc_dt_int, star.ζ, μ_tile, ws)
        end

        # continuum path (in-place, matching production scalar dispatch)
        t_micro[r] += @elapsed begin
            FT._convolve_micro_inplace!(ws.αs_cont_broad, λs_korg, αs_cont, μ_v_scalar, σ_v_scalar, ws)
        end
        t_tau[r] += @elapsed _calc_tau_cpu!(μ_tile, ws.αs_cont_broad, ws.τs_int_cont)
        t_cfunc[r] += @elapsed begin
            FT.calc_intensity_cfunc_cpu!(ws.cfunc_int_cont, Ts, λs_korg, ws.τs_int_cont)
            @views ws.cfunc_dt_int_cont .= ws.cfunc_int_cont .* (ws.τs_int_cont[2:end, :] .- ws.τs_int_cont[1:end-1, :])
        end
        t_macro[r] += @elapsed begin
            FT._convolve_macro_inplace!(ws.macro_out, λs_korg, ws.cfunc_dt_int_cont, star.ζ, μ_tile, ws)
        end
    end

    return (micro=median(t_micro), tau=median(t_tau),
            cfunc=median(t_cfunc), macro_conv=median(t_macro),
            micro_iqr=iqr(t_micro), tau_iqr=iqr(t_tau),
            cfunc_iqr=iqr(t_cfunc), macro_conv_iqr=iqr(t_macro))
end

# ── GPU batched kernel benchmark ──────────────────────────────────────────────
# mirrors the production GPU tile loop in convenience.jl: dual-stream total +
# continuum, pre-uploaded tile params with tile_offset, batched Fourier-domain
# macro accumulation, batched macro kernel precomputation
function benchmark_gpu_batched(αs, αs_cont, star, λs_korg, μs_cpu, z_rot_cpu;
                               n_repeat=10, B=8, gpu_precision::Type{<:AbstractFloat}=Float64)
    T = gpu_precision
    A_X = star.A_X
    korg_atm = Korg.interpolate_marcs(star.Teff, star.logg, A_X)

    atm_f64 = FT.AtmosphereGPU(korg_atm; T=Float64)
    Natm = length(atm_f64.zs)
    Nλ = length(λs_korg)
    Natm1 = Natm - 1
    Npad = 512
    Ntiles = length(μs_cpu)

    α_ref_f64 = zeros(Natm)
    αs_f64 = copy(αs)
    αs_cont_f64 = zeros(Natm, Nλ)
    FT.compute_alpha!(αs_f64, αs_cont_f64, Korg.Wavelengths(λs_korg), linelist, atm_f64, A_X;
                      α_ref_out=α_ref_f64, ne_warn_thresh=Inf)

    atm_gpu = T == Float64 ? atm_f64 : FT.AtmosphereGPU(korg_atm; T=T)
    αs_T = T.(αs_f64)
    αs_cont_T = T.(αs_cont_f64)
    α_ref_T = T.(α_ref_f64)

    λs_T = T.(collect(λs_korg))
    λs_gpu = CuArray(λs_T)
    σ_v = T(star.ξ)
    log_τ_ref = CuArray{T}(log.(atm_gpu.τs))
    ifactor_base = CuArray{T}(atm_gpu.τs ./ α_ref_T)
    Ts_gpu = CuArray{T}(atm_gpu.Ts)

    # separate memory for total and continuum streams
    bcmem      = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad; T=T)
    bcmem_cont = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad; T=T)
    cmem_mac   = FT.MacroConvolutionMemory(Nλ, Natm1, Npad; T=T)

    # pre-upload all tile parameters (matches production)
    all_μ_tiles_gpu = CuArray(T.(μs_cpu))
    all_dA_tiles_gpu = CuArray(T.(ones(Ntiles) .* 0.001))  # dummy dA for timing
    all_μ_v_gpu = CuArray(repeat(T.(z_rot_cpu .* FT.c_ms), inner=Natm))

    # working arrays
    τs_batch      = CUDA.zeros(T, B * Natm, Nλ)
    τs_batch_cont = CUDA.zeros(T, B * Natm, Nλ)
    cfdt_batch      = CUDA.zeros(T, B * Natm1, Nλ)
    cfdt_batch_cont = CUDA.zeros(T, B * Natm1, Nλ)

    # prime signal caches
    μ_v_prime = CUDA.zeros(T, Natm)
    bcmem.signal_cached = false
    FT.convolve_wavelength_axis_batched!(bcmem, λs_T, αs_T, μ_v_prime, σ_v, 1)
    bcmem.signal_cached = true
    bcmem_cont.signal_cached = false
    FT.convolve_wavelength_axis_batched!(bcmem_cont, λs_T, αs_cont_T, μ_v_prime, σ_v, 1)
    bcmem_cont.signal_cached = true

    # batched macro kernel precomputation (matches production)
    L_mac = cmem_mac.L
    pad_left_mac = cmem_mac.pad_left
    nfreq_mac = fld(L_mac, 2) + 1
    i0_mac = Nλ ÷ 2 + 1
    unique_μ_sorted = sort(unique(T.(μs_cpu)))
    N_unique = length(unique_μ_sorted)
    μ_to_idx = Dict(μ => Int32(i) for (i, μ) in enumerate(unique_μ_sorted))
    μ_vals_gpu = CuArray(unique_μ_sorted)
    kbuf_mac = CUDA.zeros(T, N_unique, L_mac)
    ts_kc = (32, 32)
    bs_kc = (cld(Nλ, ts_kc[1]), cld(N_unique, ts_kc[2]))
    @cuda threads=ts_kc blocks=bs_kc FT.compute_rt_macro_dft_layout_2d!(
        kbuf_mac, λs_gpu, μ_vals_gpu, Int32(i0_mac), T(star.ζ),
        Int32(Nλ), Int32(L_mac))
    kbuf_mac ./= sum(kbuf_mac, dims=2)
    plan_kc = CUDA.CUFFT.plan_rfft(kbuf_mac, 2)
    kernel_cache_flat = CUDA.zeros(Complex{T}, N_unique, nfreq_mac)
    mul!(kernel_cache_flat, plan_kc, kbuf_mac)
    μ_idx_gpu = CuArray(Int32[μ_to_idx[T(μs_cpu[i])] for i in 1:Ntiles])

    # batched macro buffers + plans
    mac_pad      = CUDA.zeros(T, B * Natm1, L_mac)
    mac_pad_cont = CUDA.zeros(T, B * Natm1, L_mac)
    mac_ft_buf      = CUDA.zeros(Complex{T}, B * Natm1, nfreq_mac)
    mac_ft_buf_cont = CUDA.zeros(Complex{T}, B * Natm1, nfreq_mac)
    plan_mac_fwd      = CUDA.CUFFT.plan_rfft(mac_pad, 2)
    plan_mac_fwd_cont = CUDA.CUFFT.plan_rfft(mac_pad_cont, 2)
    acc_ft      = CUDA.zeros(Complex{T}, Natm1, nfreq_mac)
    acc_ft_cont = CUDA.zeros(Complex{T}, Natm1, nfreq_mac)

    stream_total = CUDA.CuStream()
    stream_cont  = CUDA.CuStream()
    CUDA.synchronize()

    # timing per batch (dual-stream overlap)
    t_micro = zeros(n_repeat)
    t_tau = zeros(n_repeat)
    t_cfunc = zeros(n_repeat)
    t_macro = zeros(n_repeat)
    local αs_conv, αs_conv_c

    for r in 1:n_repeat
        CUDA.synchronize()
        fill!(acc_ft, zero(Complex{T}))
        fill!(acc_ft_cont, zero(Complex{T}))

        t_micro[r] = CUDA.@elapsed begin
            CUDA.stream!(stream_total) do
                αs_conv = FT.convolve_wavelength_axis_batched!(bcmem, λs_T, αs_T,
                    all_μ_v_gpu, σ_v, B; tile_offset=0)
            end
            CUDA.stream!(stream_cont) do
                αs_conv_c = FT.convolve_wavelength_axis_batched!(bcmem_cont, λs_T, αs_cont_T,
                    all_μ_v_gpu, σ_v, B; tile_offset=0)
            end
        end

        t_tau[r] = CUDA.@elapsed begin
            CUDA.stream!(stream_total) do
                FT.calc_tau_anchored_batched!(all_μ_tiles_gpu, log_τ_ref, ifactor_base,
                    αs_conv, τs_batch, Natm, B; tile_offset=0)
            end
            CUDA.stream!(stream_cont) do
                FT.calc_tau_anchored_batched!(all_μ_tiles_gpu, log_τ_ref, ifactor_base,
                    αs_conv_c, τs_batch_cont, Natm, B; tile_offset=0)
            end
        end

        t_cfunc[r] = CUDA.@elapsed begin
            CUDA.stream!(stream_total) do
                FT.calc_intensity_cfunc_dt_batched!(cfdt_batch, τs_batch,
                    Ts_gpu, λs_gpu, Natm, B)
            end
            CUDA.stream!(stream_cont) do
                FT.calc_intensity_cfunc_dt_batched!(cfdt_batch_cont, τs_batch_cont,
                    Ts_gpu, λs_gpu, Natm, B)
            end
        end

        t_macro[r] = CUDA.@elapsed begin
            CUDA.stream!(stream_total) do
                ts_pad = (32, 32)
                bs_pad = (cld(B * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
                @cuda threads=ts_pad blocks=bs_pad FT.pad_signal!(mac_pad, cfdt_batch,
                    Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
                mul!(mac_ft_buf, plan_mac_fwd, mac_pad)
                FT.batched_macro_multiply_accumulate!(acc_ft, mac_ft_buf, kernel_cache_flat,
                    μ_idx_gpu, all_dA_tiles_gpu, Natm1, B; tile_offset=0)
            end
            CUDA.stream!(stream_cont) do
                ts_pad = (32, 32)
                bs_pad = (cld(B * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
                @cuda threads=ts_pad blocks=bs_pad FT.pad_signal!(mac_pad_cont, cfdt_batch_cont,
                    Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
                mul!(mac_ft_buf_cont, plan_mac_fwd_cont, mac_pad_cont)
                FT.batched_macro_multiply_accumulate!(acc_ft_cont, mac_ft_buf_cont, kernel_cache_flat,
                    μ_idx_gpu, all_dA_tiles_gpu, Natm1, B; tile_offset=0)
            end
        end
    end

    # normalize to per-tile cost (each measurement covers B tiles)
    t_micro ./= B
    t_tau   ./= B
    t_cfunc ./= B
    t_macro ./= B

    return (micro=median(t_micro), tau=median(t_tau),
            cfunc=median(t_cfunc), macro_conv=median(t_macro),
            micro_iqr=iqr(t_micro), tau_iqr=iqr(t_tau),
            cfunc_iqr=iqr(t_cfunc), macro_conv_iqr=iqr(t_macro))
end

# ── end-to-end benchmark ──────────────────────────────────────────────────────
function benchmark_end_to_end(star, linelist; Δλ, Nϕ, use_gpu, gpu_precision=Float64)
    f = () -> calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                   use_gpu=use_gpu, gpu_precision=gpu_precision,
                                   ne_warn_thresh=Inf, showprogress=false)
    trial = use_gpu ? (@benchmark CUDA.@sync $f()) : (@benchmark $f())
    result = f()
    return median(trial).time / 1e9, result
end

# ── run benchmarks ─────────────────────────────────────────────────────────────
println("="^70)
println("DISK INTEGRATION BENCHMARK")
println("="^70)
println()

cpu_times = nothing
gpu64_times = nothing
gpu32_times = nothing

if run_pertile
# CPU per-tile
println("─"^40)
println("CPU BENCHMARK (per-tile)")
println("─"^40)
cpu_times = benchmark_cpu_pertile(copy(αs), copy(αs_cont), atm_cpu, λs_korg, star,
                                  μs_cpu, z_rot_cpu)
@printf("  Microturbulence: %8.3f ms  (IQR %.3f)\n", cpu_times.micro * 1000, cpu_times.micro_iqr * 1000)
@printf("  Tau integration: %8.3f ms  (IQR %.3f)\n", cpu_times.tau * 1000, cpu_times.tau_iqr * 1000)
@printf("  Cfunc:           %8.3f ms  (IQR %.3f)\n", cpu_times.cfunc * 1000, cpu_times.cfunc_iqr * 1000)
@printf("  Macro conv:      %8.3f ms  (IQR %.3f)\n", cpu_times.macro_conv * 1000, cpu_times.macro_conv_iqr * 1000)
cpu_total_pertile = (cpu_times.micro + cpu_times.tau + cpu_times.cfunc + cpu_times.macro_conv) * 1000
@printf("  Total per tile:  %8.3f ms\n", cpu_total_pertile)
println()

# GPU batched kernels
if use_gpu
    B_bench = 8
    for (prec, G) in [("Float64", Float64), ("Float32", Float32)]
        global gpu64_times, gpu32_times
        println("─"^40)
        println("GPU BENCHMARK ($prec, batched, B=$B_bench)")
        println("─"^40)
        gt = benchmark_gpu_batched(copy(αs), copy(αs_cont), star, λs_korg, μs_cpu, z_rot_cpu;
                                   B=B_bench, gpu_precision=G)
        @printf("  Micro (per tile): %8.3f ms  (IQR %.3f)\n", gt.micro * 1000, gt.micro_iqr * 1000)
        @printf("  Tau (per tile):   %8.3f ms  (IQR %.3f)\n", gt.tau * 1000, gt.tau_iqr * 1000)
        @printf("  Cfunc (per tile): %8.3f ms  (IQR %.3f)\n", gt.cfunc * 1000, gt.cfunc_iqr * 1000)
        @printf("  Macro (per tile): %8.3f ms  (IQR %.3f)\n", gt.macro_conv * 1000, gt.macro_conv_iqr * 1000)
        gpu_total = (gt.micro + gt.tau + gt.cfunc + gt.macro_conv) * 1000
        @printf("  Total per tile:   %8.3f ms  (from B=%d batch)\n", gpu_total, B_bench)
        @printf("  Speedup vs CPU:   %.1f×\n", cpu_total_pertile / gpu_total)
        println()
        if G == Float64
            gpu64_times = gt
        else
            gpu32_times = gt
        end
    end
end
end  # run_pertile

# save pertile data immediately so it survives an e2e crash
if run_pertile && cpu_times !== nothing
    steps = ["microturbulence", "tau", "cfunc", "macro"]
    cpu_med = [cpu_times.micro, cpu_times.tau, cpu_times.cfunc, cpu_times.macro_conv] .* 1000.0
    cpu_iqr = [cpu_times.micro_iqr, cpu_times.tau_iqr, cpu_times.cfunc_iqr, cpu_times.macro_conv_iqr] .* 1000.0

    g64_med = gpu64_times !== nothing ?
        [gpu64_times.micro, gpu64_times.tau, gpu64_times.cfunc, gpu64_times.macro_conv] .* 1000.0 :
        fill(NaN, 4)
    g64_iqr = gpu64_times !== nothing ?
        [gpu64_times.micro_iqr, gpu64_times.tau_iqr, gpu64_times.cfunc_iqr, gpu64_times.macro_conv_iqr] .* 1000.0 :
        fill(NaN, 4)
    g32_med = gpu32_times !== nothing ?
        [gpu32_times.micro, gpu32_times.tau, gpu32_times.cfunc, gpu32_times.macro_conv] .* 1000.0 :
        fill(NaN, 4)
    g32_iqr = gpu32_times !== nothing ?
        [gpu32_times.micro_iqr, gpu32_times.tau_iqr, gpu32_times.cfunc_iqr, gpu32_times.macro_conv_iqr] .* 1000.0 :
        fill(NaN, 4)

    open(joinpath(datadir, "pertile_timings.csv"), "w") do io
        println(io, "step,cpu_med_ms,cpu_iqr_ms,gpu64_med_ms,gpu64_iqr_ms,gpu32_med_ms,gpu32_iqr_ms")
        for i in eachindex(steps)
            @printf(io, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    steps[i], cpu_med[i], cpu_iqr[i], g64_med[i], g64_iqr[i], g32_med[i], g32_iqr[i])
        end
    end
    # metadata for the plotting script
    open(joinpath(datadir, "pertile_meta.csv"), "w") do io
        println(io, "Nlambda,Natm,Nphi,Ntiles,B_gpu,threads")
        @printf(io, "%d,%d,%d,%d,%d,%d\n", Nλ, Natm, Nϕ, n_tiles,
                use_gpu ? 8 : 0, Threads.nthreads())
    end
    println("Saved: pertile_timings.csv, pertile_meta.csv")
end

# reclaim GPU memory before e2e
if use_gpu
    GC.gc()
    CUDA.reclaim()
end

t_cpu_e2e = NaN
t_gpu64_e2e = NaN
t_gpu32_e2e = NaN

if run_e2e
# end-to-end
println("─"^40)
println("END-TO-END: calc_formation_temp")
println("─"^40)

t_cpu_e2e, result_cpu = benchmark_end_to_end(star, linelist; Δλ=Δλ, Nϕ=Nϕ, use_gpu=false)
@printf("CPU  (Nϕ=%d): %.2f s\n", Nϕ, t_cpu_e2e)

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

# save e2e data
if !isnan(t_cpu_e2e)
    open(joinpath(datadir, "e2e_timings.csv"), "w") do io
        println(io, "backend,precision,time_s,Nphi,Natm,Nlambda,Ntiles,threads")
        @printf(io, "cpu,float64,%.4f,%d,%d,%d,%d,%d\n", t_cpu_e2e, Nϕ, Natm, Nλ, n_tiles, Threads.nthreads())
        if use_gpu
            @printf(io, "gpu,float64,%.4f,%d,%d,%d,%d,1\n", t_gpu64_e2e, Nϕ, Natm, Nλ, n_tiles)
            @printf(io, "gpu,float32,%.4f,%d,%d,%d,%d,1\n", t_gpu32_e2e, Nϕ, Natm, Nλ, n_tiles)
        end
    end
    println("Saved: e2e_timings.csv")
end
end  # run_e2e

println("Data written to: ", datadir)

println()
println("="^70)
println("DONE")
println("="^70)
