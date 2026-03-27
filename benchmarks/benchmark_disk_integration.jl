using Revise
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Printf, Statistics

# ── setup ──────────────────────────────────────────────────────────────────────
use_gpu = FT.GPU_DEFAULT
println("GPU available: ", use_gpu)

# solar linelist — ~500 Fe I lines near 6300 A
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
println()

# ── CPU benchmark ──────────────────────────────────────────────────────────────
function benchmark_cpu_loop(αs, αs_cont, atm_cpu, λs_korg, star, μs_cpu, dA_cpu, z_rot_cpu;
                            n_tiles_max=100)
    T = Float64
    Natm = length(atm_cpu.zs)
    Nλ = length(λs_korg)
    Ts = atm_cpu.Ts
    α_ref = zeros(T, Natm)
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_cpu, star.A_X;
                      α_ref_out=α_ref, vmic_ref_cms=star.ξ * 100.0, ne_warn_thresh=Inf)

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

    n_run = min(n_tiles_max, length(μs_cpu))

    # time individual steps on first tile
    μ_tile = μs_cpu[1]
    μ_v_rot .= z_rot_cpu[1] .* FT.c_ms

    t_micro = @elapsed αs_broad_i = FT.convolve_wavelength_axis(λs_korg, αs, μ_v_rot, σ_v)
    t_tau   = @elapsed _calc_tau_cpu!(μ_tile, αs_broad_i, τs_int)
    t_cfunc = @elapsed FT.calc_intensity_cfunc_cpu!(cfunc_int, Ts, λs_korg, τs_int)
    cfunc_dt_int = cfunc_int .* diff(τs_int, dims=1)
    t_macro = @elapsed FT.convolve_rt_macro(λs_korg, cfunc_dt_int, star.ζ, μ_tile)

    println("CPU per-tile step timings (first tile):")
    @printf("  Microturbulence convolution: %8.3f ms\n", t_micro * 1000)
    @printf("  Tau integration:             %8.3f ms\n", t_tau * 1000)
    @printf("  Cfunc computation:           %8.3f ms\n", t_cfunc * 1000)
    @printf("  Macro convolution:           %8.3f ms\n", t_macro * 1000)
    @printf("  Total per tile (est):        %8.3f ms\n", (t_micro + t_tau + t_cfunc + t_macro) * 1000)
    @printf("  Tiles × 2 (total+cont):      %8.3f ms\n", 2 * (t_micro + t_tau + t_cfunc + t_macro) * 1000)
    println()

    # time full loop over n_run tiles
    t_loop = @elapsed begin
        flux_integration = zeros(T, Nλ)
        cfunc_flux_integration = zeros(T, Natm - 1, Nλ)
        for i in 1:n_run
            μ_tile = μs_cpu[i]
            μ_v_rot .= z_rot_cpu[i] .* FT.c_ms
            αs_broad_i = FT.convolve_wavelength_axis(λs_korg, αs, μ_v_rot, σ_v)
            _calc_tau_cpu!(μ_tile, αs_broad_i, τs_int)
            FT.calc_intensity_cfunc_cpu!(cfunc_int, Ts, λs_korg, τs_int)
            cfunc_dt_int = cfunc_int .* diff(τs_int, dims=1)
            cfunc_int_i_mac = FT.convolve_rt_macro(λs_korg, cfunc_dt_int, star.ζ, μ_tile)
            flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
            cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]
        end
    end
    @printf("CPU loop (%d tiles, total only): %.3f s  (%.3f ms/tile)\n",
            n_run, t_loop, t_loop / n_run * 1000)
    @printf("CPU extrapolated full loop (%d tiles × 2): %.1f s\n",
            length(μs_cpu), 2 * t_loop / n_run * length(μs_cpu))
    println()
    return nothing
end

# ── GPU benchmark ──────────────────────────────────────────────────────────────
function benchmark_gpu_loop(αs, αs_cont, star, λs_korg, μs_cpu, dA_cpu, z_rot_cpu;
                            n_tiles_max=200)
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
    cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

    σ_v = CUDA.zeros(T, Natm) .+ star.ξ
    μ_v_rot = CUDA.zeros(T, Natm)

    n_run = min(n_tiles_max, length(μs_cpu))

    # warmup
    μ_v_rot .= z_rot_cpu[1] .* FT.c_ms
    cfunc_warmup = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[1], μ_v_rot, σ_v)
    tbc_warmup = cfunc_warmup.cfunc_dt
    FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, Array(tbc_warmup), star.ζ, μs_cpu[1])
    CUDA.synchronize()

    # per-step timings (single tile)
    μ_tile = μs_cpu[1]
    μ_v_rot .= z_rot_cpu[1] .* FT.c_ms

    # microturbulence + tau + cfunc (all inside calc_intensity_quantities)
    CUDA.synchronize()
    t_intensity = CUDA.@elapsed begin
        cfunc_intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
    end

    # macro convolution
    tbc = Array(cfunc_intensity.cfunc_dt)
    CUDA.synchronize()
    t_macro = CUDA.@elapsed begin
        cfunc_int_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, star.ζ, μ_tile)
    end

    # accumulation
    flux_integration = CUDA.zeros(T, Nλ)
    cfunc_flux_integration = CUDA.zeros(T, Natm - 1, Nλ)
    CUDA.synchronize()
    t_accum = CUDA.@elapsed begin
        flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[1]
        cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[1]
    end

    println("GPU per-tile step timings (single tile, after warmup):")
    @printf("  calc_intensity_quantities:  %8.3f ms\n", t_intensity * 1000)
    @printf("  convolve_rt_macro_gpu:      %8.3f ms\n", t_macro * 1000)
    @printf("  Accumulation:               %8.3f ms\n", t_accum * 1000)
    @printf("  Total per tile (1 path):    %8.3f ms\n", (t_intensity + t_macro + t_accum) * 1000)
    @printf("  Total per tile (×2 paths):  %8.3f ms\n", 2 * (t_intensity + t_macro + t_accum) * 1000)
    println()

    # GPU allocation measurement (single tile)
    println("GPU allocations (single tile, total path):")
    alloc_intensity = CUDA.@allocated begin
        cfunc_intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
    end
    @printf("  calc_intensity_quantities: %d bytes (%.2f MB)\n", alloc_intensity, alloc_intensity / 1e6)

    tbc = Array(cfunc_intensity.cfunc_dt)
    alloc_macro = CUDA.@allocated begin
        FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, star.ζ, μ_tile)
    end
    @printf("  convolve_rt_macro_gpu:     %d bytes (%.2f MB)\n", alloc_macro, alloc_macro / 1e6)

    alloc_accum = CUDA.@allocated begin
        flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[1]
        cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[1]
    end
    @printf("  Accumulation:              %d bytes (%.2f MB)\n", alloc_accum, alloc_accum / 1e6)
    @printf("  Total per tile (×2):       %d bytes (%.2f MB)\n",
            2 * (alloc_intensity + alloc_macro + alloc_accum),
            2 * (alloc_intensity + alloc_macro + alloc_accum) / 1e6)
    @printf("  Extrapolated full loop:    %.1f MB\n",
            2 * (alloc_intensity + alloc_macro + alloc_accum) * length(μs_cpu) / 1e6)
    println()

    # full loop timing
    CUDA.synchronize()
    t_loop = CUDA.@elapsed begin
        fill!(flux_integration, zero(T))
        fill!(cfunc_flux_integration, zero(T))
        for i in 1:n_run
            μ_tile = μs_cpu[i]
            μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

            cfunc_intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
            tbc = cfunc_intensity.cfunc_dt
            cfunc_int_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, Array(tbc), star.ζ, μ_tile)

            cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
            tbc_cont = cfunc_intensity_cont.cfunc_dt
            cfunc_int_cont_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, Array(tbc_cont), star.ζ, μ_tile)

            flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
            cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]
        end
    end
    CUDA.synchronize()

    @printf("GPU loop (%d tiles, total+cont): %.3f s  (%.3f ms/tile)\n",
            n_run, t_loop, t_loop / n_run * 1000)
    @printf("GPU extrapolated full loop (%d tiles): %.1f s\n",
            length(μs_cpu), t_loop / n_run * length(μs_cpu))
    println()

    # note: signal_cached is false here because this benchmark calls low-level functions
    # directly without the priming logic in _calc_formation_temp_gpu.
    # The end-to-end benchmark below exercises the full optimized path.
    println()
    return nothing
end

# ── run benchmarks ─────────────────────────────────────────────────────────────
println("="^70)
println("DISK INTEGRATION BENCHMARK")
println("="^70)
println()

println("─"^40)
println("CPU BENCHMARK")
println("─"^40)
benchmark_cpu_loop(copy(αs), copy(αs_cont), atm_cpu, λs_korg, star, μs_cpu, dA_cpu, z_rot_cpu;
                   n_tiles_max=50)

if use_gpu
    println("─"^40)
    println("GPU BENCHMARK (low-level)")
    println("─"^40)
    benchmark_gpu_loop(copy(αs), copy(αs_cont), star, λs_korg, μs_cpu, dA_cpu, z_rot_cpu;
                       n_tiles_max=200)
end

# ── end-to-end benchmark ──────────────────────────────────────────────────────
println("─"^40)
println("END-TO-END: calc_formation_temp")
println("─"^40)

# warmup (first call includes compilation)
result_warmup = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=16,
                                    use_gpu=use_gpu, ne_warn_thresh=Inf)

# # CPU end-to-end
# t_cpu_e2e = @elapsed begin
#     result_cpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
#                                      use_gpu=false, ne_warn_thresh=Inf)
# end
# @printf("CPU  calc_formation_temp (Nϕ=%d): %.2f s\n", Nϕ, t_cpu_e2e)

# GPU end-to-end
if use_gpu
    CUDA.synchronize()
    t_gpu_e2e = @elapsed begin
        result_gpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                         use_gpu=true, ne_warn_thresh=Inf)
    end
    CUDA.synchronize()
    @printf("GPU  calc_formation_temp (Nϕ=%d): %.2f s\n", Nϕ, t_gpu_e2e)
    @printf("Speedup: %.1fx\n", t_cpu_e2e / t_gpu_e2e)

    # sanity check: flux agreement
    max_flux_diff = maximum(abs.(result_cpu.flux .- result_gpu.flux))
    @printf("Max flux difference (CPU vs GPU): %.2e\n", max_flux_diff)
end
println()

println("="^70)
println("DONE")
println("="^70)
