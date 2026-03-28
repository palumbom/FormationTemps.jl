using Revise
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Printf, Statistics
using PythonPlot; plt = PythonPlot.pyplot
plt.style.use(joinpath(FT.moddir, "fig.mplstyle"))
using DelimitedFiles

# output directories
plotdir = joinpath(FT.moddir, "docs", "src", "static")
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
    cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

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

# ── plots ─────────────────────────────────────────────────────────────────────
plt.ioff()

# combined per-tile breakdown + end-to-end absolute timing
if use_gpu && gpu_times !== nothing
    let
    fig, (ax_brk, ax_abs) = plt.subplots(1, 2, figsize=(10, 3.2),
                                          gridspec_kw=Dict("width_ratios" => [3, 1], "wspace" => 0.35))

    # ── left panel: normalized per-tile breakdown ─────────────────────────────
    step_labels = ["{\\rm Microturbulence}", "{\\rm Optical depth}",
                   "{\\rm Contribution fn.}", "{\\rm Macroturbulence}"]
    cpu_ms = [cpu_times.micro, cpu_times.tau, cpu_times.cfunc, cpu_times.macro_conv] .* 1000.0
    gpu_ms_fused = [gpu_times.intensity, gpu_times.macro_conv] .* 1000.0
    cpu_total = sum(cpu_ms)
    gpu_total = sum(gpu_ms_fused)
    cpu_pct = cpu_ms ./ cpu_total .* 100.0
    gpu_pct = gpu_ms_fused ./ gpu_total .* 100.0

    colors_cpu = ["#91D1F7", "#56B4E9", "#2A96D1", "#1A7AB5"]
    colors_gpu = ["#F5A66E", "#D55E00"]
    pe = PythonPlot.pyimport("matplotlib.patheffects")
    bar_stroke = [pe.withStroke(linewidth=0.0, foreground="black")]

    bar_h = 0.55
    y_cpu = 0
    y_gpu = 1
    pct_min_inside = 12  # minimum segment width (%) to place label inside

    # grid behind everything
    ax_brk.set_axisbelow(true)
    ax_brk.grid(true, axis="x", color="#DDDDDD", lw=0.5)
    ax_brk.grid(false, axis="y")

    # CPU row
    cpu_left = 0.0
    for (i, (label, pct, ms)) in enumerate(zip(step_labels, cpu_pct, cpu_ms))
        ax_brk.barh(y_cpu, pct, left=cpu_left, color=colors_cpu[i],
                    edgecolor="white", height=bar_h, zorder=3)
        if pct > pct_min_inside
            ax_brk.text(cpu_left + pct / 2, y_cpu, @sprintf("{\\rm %.1f ms}", ms),
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold", zorder=4, path_effects=bar_stroke)
        end
        cpu_left += pct
    end

    # GPU row: fused intensity + macro
    ax_brk.barh(y_gpu, gpu_pct[1], left=0.0, color=colors_gpu[1],
                edgecolor="white", height=bar_h, zorder=3)
    if gpu_pct[1] > pct_min_inside
        ax_brk.text(gpu_pct[1] / 2, y_gpu, @sprintf("{\\rm %.2f ms}", gpu_ms_fused[1]),
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", zorder=4, path_effects=bar_stroke)
    end

    ax_brk.barh(y_gpu, gpu_pct[2], left=gpu_pct[1], color=colors_gpu[2],
                edgecolor="white", height=bar_h, zorder=3)
    if gpu_pct[2] > pct_min_inside
        ax_brk.text(gpu_pct[1] + gpu_pct[2] / 2, y_gpu,
                    @sprintf("{\\rm %.2f ms}", gpu_ms_fused[2]),
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", zorder=4, path_effects=bar_stroke)
    end

    # total time underneath y-axis tick labels (in axes coords for x, data coords for y)
    ax_brk.text(-0.01, y_cpu - 0.22, @sprintf("{\\rm (%.1f ms)}", cpu_total),
                ha="right", va="top", fontsize=7, color="#555555",
                transform=ax_brk.get_yaxis_transform(), zorder=4)
    ax_brk.text(-0.01, y_gpu - 0.22, @sprintf("{\\rm (%.2f ms)}", gpu_total),
                ha="right", va="top", fontsize=7, color="#555555",
                transform=ax_brk.get_yaxis_transform(), zorder=4)

    # segment name annotations below each bar
    ann_color = "#000000"
    ann_fs = 7
    ann_props = Dict("arrowstyle" => "-", "color" => "#999999", "lw" => 0.5)

    # CPU annotations (below CPU bar) — stagger vertically if x-centers are close
    min_sep = 15.0  # minimum horizontal separation (in %) before staggering
    base_dy = 0.3
    stagger_dy = 0.35

    cpu_x_mids = Float64[]
    cpu_cum = 0.0
    for pct in cpu_pct
        push!(cpu_x_mids, cpu_cum + pct / 2)
        cpu_cum += pct
    end

    cpu_dy = fill(base_dy, length(cpu_pct))
    for i in 2:length(cpu_pct)
        if abs(cpu_x_mids[i] - cpu_x_mids[i-1]) < min_sep
            cpu_dy[i] = cpu_dy[i-1] + stagger_dy
        end
    end

    ann_bbox = Dict("boxstyle" => "square,pad=0.05", "facecolor" => "white",
                     "edgecolor" => "none", "alpha" => 1.0)

    # draw all leader lines first (low zorder)
    for (i, x_mid) in enumerate(cpu_x_mids)
        ax_brk.plot([x_mid, x_mid],
                    [y_cpu - bar_h / 2, y_cpu - bar_h / 2 - cpu_dy[i]],
                    color="#999999", lw=0.5, zorder=4)
    end
    # then draw text on top (high zorder, white bbox covers lines)
    for (i, (label, x_mid)) in enumerate(zip(step_labels, cpu_x_mids))
        ax_brk.text(x_mid, y_cpu - bar_h / 2 - cpu_dy[i],
                    label, ha="center", va="top", fontsize=ann_fs,
                    color=ann_color, bbox=ann_bbox, zorder=5)
    end

    # GPU annotations (above GPU bar)
    gpu_labels = ["{\\rm Intensity (fused)}", "{\\rm Macroturbulence}"]
    gpu_lefts = [0.0, gpu_pct[1]]
    gpu_x_mids = [left + pct / 2 for (left, pct) in zip(gpu_lefts, gpu_pct)]

    gpu_dy = fill(base_dy, length(gpu_pct))
    for i in 2:length(gpu_pct)
        if abs(gpu_x_mids[i] - gpu_x_mids[i-1]) < min_sep
            gpu_dy[i] = gpu_dy[i-1] + stagger_dy
        end
    end

    for (i, x_mid) in enumerate(gpu_x_mids)
        ax_brk.plot([x_mid, x_mid],
                    [y_gpu + bar_h / 2, y_gpu + bar_h / 2 + gpu_dy[i]],
                    color="#999999", lw=0.5, zorder=4)
    end
    for (i, (label, x_mid)) in enumerate(zip(gpu_labels, gpu_x_mids))
        ax_brk.text(x_mid, y_gpu + bar_h / 2 + gpu_dy[i],
                    label, ha="center", va="bottom", fontsize=ann_fs,
                    color=ann_color, bbox=ann_bbox, zorder=5)
    end

    # expand y-limits to fit staggered annotations
    max_below = maximum(cpu_dy) + 0.5
    max_above = maximum(gpu_dy) + 0.7
    ax_brk.set_yticks([y_cpu, y_gpu])
    ax_brk.set_yticklabels(["{\\rm CPU}", "{\\rm GPU}"])
    ax_brk.set_xlim(-0.2, 100.3)
    ax_brk.set_ylim(y_cpu - max_below, y_gpu + max_above)
    ax_brk.set_xlabel("{\\rm Fraction of per-tile time [\\%]}")

    # ── right panel: absolute end-to-end times (log scale) ──────────────────
    speedup_e2e = t_cpu_e2e / t_gpu_e2e

    ax_abs.set_axisbelow(true)
    ax_abs.grid(true, axis="x", color="#DDDDDD", lw=0.5)
    ax_abs.grid(false, axis="y")
    ax_abs.set_xscale("log")

    ax_abs.barh([y_gpu, y_cpu], [t_gpu_e2e, t_cpu_e2e],
                color=["#D55E00", "#56B4E9"], edgecolor="white", height=bar_h, zorder=3)

    # time labels to the right of each bar
    for (y, t) in zip([y_cpu, y_gpu], [t_cpu_e2e, t_gpu_e2e])
        ax_abs.text(t * 1.15, y, @sprintf("{\\rm %.1f s}", t),
                    ha="left", va="center", fontsize=9, fontweight="bold",
                    color="#333333", zorder=4)
    end

    ax_abs.set_yticks([y_cpu, y_gpu])
    ax_abs.set_yticklabels(["{\\rm CPU}", "{\\rm GPU}"])
    ax_abs.set_xlabel("{\\rm Wall-clock time [s]}")
    ax_abs.set_title(@sprintf("{\\rm End-to-end (}\$%.0f\\times\$ {\\rm speedup)}", speedup_e2e))

    # clean scalar tick labels instead of 10^n
    ticker = PythonPlot.pyimport("matplotlib.ticker")
    ax_abs.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_abs.xaxis.get_major_formatter().set_scientific(false)
    ax_abs.set_xlim(t_gpu_e2e * 0.7 - 0.2, t_cpu_e2e * 4.75)
    ax_abs.set_ylim(y_cpu - max_below, y_gpu + max_above)

    ax_brk.set_title(@sprintf("{\\rm Per-tile breakdown (}\$N_\\lambda\${\\rm =%d, }\$N_{\\rm atm}\${\\rm =%d)}", Nλ, Natm))
    fig.subplots_adjust(bottom=0.18, top=0.88)
    fig.savefig(joinpath(plotdir, "benchmark_pertile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    println("Saved: benchmark_pertile.png")
    end # let
end

println()
println("="^70)
println("DONE")
println("="^70)
