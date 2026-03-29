using Printf, DelimitedFiles
using PythonPlot; plt = PythonPlot.pyplot

const PROJECT_DIR = dirname(@__DIR__)
const DATADIR = joinpath(PROJECT_DIR, "benchmarks", "data")
const PLOTDIR = joinpath(PROJECT_DIR, "docs", "src", "static")
!isdir(PLOTDIR) && mkpath(PLOTDIR)

stylefile = joinpath(PROJECT_DIR, "fig.mplstyle")
if isfile(stylefile)
    plt.style.use(stylefile)
end
plt.ioff()

# ── consistent color palette ─────────────────────────────────────────────────
const COL_CPU_1T  = "#91D1F7"  # light blue — CPU single-thread
const COL_CPU_MT  = "#2A96D1"  # dark blue  — CPU multi-thread
const COL_GPU64   = "#D55E00"  # orange     — GPU Float64
const COL_GPU32   = "#009E73"  # green      — GPU Float32

# ── helper ────────────────────────────────────────────────────────────────────
function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end] if !isempty(strip(l))]
    return header, rows
end

# ══════════════════════════════════════════════════════════════════════════════
# 1. Convolution kernel timings
# ══════════════════════════════════════════════════════════════════════════════
try
    csv = joinpath(DATADIR, "convolution_timings.csv")
    header, rows = read_csv(csv)

    kernels = [r[1] for r in rows]
    cpu_med = [parse(Float64, r[2]) for r in rows]
    cpu_iq = [parse(Float64, r[3]) for r in rows]

    # detect old (6-col) vs new (7-col) CSV format
    has_gpu32 = length(rows[1]) >= 7
    g64_med = [parse(Float64, r[4]) for r in rows]
    g64_iq = [parse(Float64, r[5]) for r in rows]
    if has_gpu32
        g32_med = [parse(Float64, r[6]) for r in rows]
        g32_iq = [parse(Float64, r[7]) for r in rows]
    end

    fig, ax = plt.subplots(figsize=(9, 4))
    x = 0:length(kernels)-1
    nbar = has_gpu32 ? 3 : 2
    w = 0.8 / nbar

    ax.bar(x .- (nbar - 1) * w / 2, cpu_med, w,
           yerr=cpu_iq, capsize=3, ecolor="#333333",
           label="{\\rm CPU}", color=COL_CPU_MT, edgecolor="none")
    ax.bar(x .- (nbar - 1) * w / 2 .+ w, g64_med, w,
           yerr=g64_iq, capsize=3, ecolor="#333333",
           label="{\\rm GPU (Float64)}", color=COL_GPU64, edgecolor="none")
    if has_gpu32
        ax.bar(x .- (nbar - 1) * w / 2 .+ 2w, g32_med, w,
               yerr=g32_iq, capsize=3, ecolor="#333333",
               label="{\\rm GPU (Float32)}", color=COL_GPU32, edgecolor="none")
    end

    # speedup annotations: nested brackets for GPU64 and GPU32
    for i in eachindex(kernels)
        xi = i - 1
        y_top = maximum([cpu_med[i] + cpu_iq[i], g64_med[i] + g64_iq[i]])
        if has_gpu32
            y_top = max(y_top, g32_med[i] + g32_iq[i])
        end

        # positions of the three bar centers
        x_cpu = xi - (nbar - 1) * w / 2
        x_g64 = x_cpu + w
        x_g32 = has_gpu32 ? x_cpu + 2w : x_g64

        # inner bracket: CPU → GPU64
        y_inner = y_top * 1.15
        tick_h = y_inner * 0.06
        sp64 = cpu_med[i] / g64_med[i]
        ax.plot([x_cpu - w/2, x_cpu - w/2, x_g64 + w/2, x_g64 + w/2],
                [y_inner - tick_h, y_inner, y_inner, y_inner - tick_h],
                color=COL_GPU64, lw=0.8, clip_on=false)
        ax.text((x_cpu + x_g64) / 2, y_inner * 1.04,
                @sprintf("\$\\sim %.0f\\times\$", sp64),
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=COL_GPU64)

        if has_gpu32
            # outer bracket: CPU → GPU32
            y_outer = y_inner * 1.55
            sp32 = cpu_med[i] / g32_med[i]
            ax.plot([x_cpu - w/2, x_cpu - w/2, x_g32 + w/2, x_g32 + w/2],
                    [y_outer - tick_h, y_outer, y_outer, y_outer - tick_h],
                    color=COL_GPU32, lw=0.8, clip_on=false)
            ax.text((x_cpu + x_g32) / 2, y_outer * 1.04,
                    @sprintf("\$\\sim %.0f\\times\$", sp32),
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color=COL_GPU32)
        end
    end

    ax.set_xticks(collect(x))
    ax.set_xticklabels(["{\\rm " * k * "}" for k in kernels], rotation=20, ha="right")
    ax.set_ylabel("{\\rm Time [ms]}")
    ax.set_title("{\\rm Convolution kernel timings}")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(false)

    y_max_data = maximum(cpu_med .+ cpu_iq)
    ax.set_ylim(nothing, y_max_data * (has_gpu32 ? 8.0 : 5.0))

    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "benchmark_convolutions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    println("Saved: benchmark_convolutions.png")
catch e
    println("Skipping convolution plot: ", e)
end

# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-tile breakdown + end-to-end timing
# ══════════════════════════════════════════════════════════════════════════════
try
    # read per-tile data (new format: step,cpu_ms,gpu64_ms,gpu32_ms)
    _, pt_rows = read_csv(joinpath(DATADIR, "pertile_timings.csv"))
    steps = [r[1] for r in pt_rows]
    cpu_ms = [parse(Float64, r[2]) for r in pt_rows]
    gpu_ms = [parse(Float64, r[3]) for r in pt_rows]  # gpu64 for the breakdown plot

    # read end-to-end data (new format: backend,precision,time_s,...)
    _, e2e_rows = read_csv(joinpath(DATADIR, "e2e_timings.csv"))

    # detect CSV format: new has "precision" as column 2
    has_precision_col = length(e2e_rows[1]) >= 8

    e2e = Dict{String, NamedTuple{(:time_s, :Nphi, :Natm, :Nlambda), NTuple{4, Float64}}}()
    for r in e2e_rows
        if has_precision_col
            key = r[1] == "cpu" ? "cpu" : r[1] * "_" * r[2]  # "gpu_float64", "gpu_float32"
            e2e[key] = (time_s=parse(Float64, r[3]), Nphi=parse(Float64, r[4]),
                        Natm=parse(Float64, r[5]), Nlambda=parse(Float64, r[6]))
        else
            e2e[r[1]] = (time_s=parse(Float64, r[2]), Nphi=parse(Float64, r[3]),
                         Natm=parse(Float64, r[4]), Nlambda=parse(Float64, r[5]))
        end
    end

    has_gpu = any(k -> startswith(k, "gpu"), keys(e2e)) && any(g -> g > 0.0, gpu_ms)
    if !has_gpu
        println("Skipping per-tile plot: no GPU data")
    else
        Nλ = Int(e2e["cpu"].Nlambda)
        Natm = Int(e2e["cpu"].Natm)

        cpu_total = sum(cpu_ms)
        gpu_total = sum(gpu_ms)
        cpu_pct = cpu_ms ./ cpu_total .* 100.0
        gpu_pct = gpu_ms ./ gpu_total .* 100.0

        pe = PythonPlot.pyimport("matplotlib.patheffects")
        bar_stroke = [pe.withStroke(linewidth=0.0, foreground="black")]

        fig, ax_brk = plt.subplots(figsize=(7, 3.2))

        step_labels = ["{\\rm Microturbulence}", "{\\rm Optical depth}",
                       "{\\rm Contribution fn.}", "{\\rm Macroturbulence}"]
        colors_cpu = ["#B8DFFB", COL_CPU_1T, COL_CPU_MT, "#1A7AB5"]
        colors_gpu = [COL_GPU64, "#F5A66E", "#FFD3A6", "#B8432E"]

        bar_h = 0.55
        y_cpu = 0
        y_gpu = 1
        pct_min_inside = 12

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

        # GPU row
        gpu_left = 0.0
        for (i, (label, pct, ms)) in enumerate(zip(step_labels, gpu_pct, gpu_ms))
            ax_brk.barh(y_gpu, pct, left=gpu_left, color=colors_gpu[i],
                        edgecolor="white", height=bar_h, zorder=3)
            if pct > pct_min_inside
                ax_brk.text(gpu_left + pct / 2, y_gpu, @sprintf("{\\rm %.2f ms}", ms),
                            ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold", zorder=4, path_effects=bar_stroke)
            end
            gpu_left += pct
        end

        # total time annotations
        ax_brk.text(-0.01, y_cpu - 0.22, @sprintf("{\\rm (%.1f ms)}", cpu_total),
                    ha="right", va="top", fontsize=7, color="#555555",
                    transform=ax_brk.get_yaxis_transform(), zorder=4)
        ax_brk.text(-0.01, y_gpu - 0.22, @sprintf("{\\rm (%.2f ms)}", gpu_total),
                    ha="right", va="top", fontsize=7, color="#555555",
                    transform=ax_brk.get_yaxis_transform(), zorder=4)

        # segment annotations
        ann_color = "#000000"
        ann_fs = 7
        min_sep = 15.0
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

        for (i, x_mid) in enumerate(cpu_x_mids)
            ax_brk.plot([x_mid, x_mid],
                        [y_cpu - bar_h / 2, y_cpu - bar_h / 2 - cpu_dy[i]],
                        color="#999999", lw=0.5, zorder=4)
        end
        for (i, (label, x_mid)) in enumerate(zip(step_labels, cpu_x_mids))
            ax_brk.text(x_mid, y_cpu - bar_h / 2 - cpu_dy[i],
                        label, ha="center", va="top", fontsize=ann_fs,
                        color=ann_color, bbox=ann_bbox, zorder=5)
        end

        # GPU annotations (above GPU bar)
        gpu_x_mids = Float64[]
        gpu_cum = 0.0
        for pct in gpu_pct
            push!(gpu_x_mids, gpu_cum + pct / 2)
            gpu_cum += pct
        end

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
        for (i, (label, x_mid)) in enumerate(zip(step_labels, gpu_x_mids))
            ax_brk.text(x_mid, y_gpu + bar_h / 2 + gpu_dy[i],
                        label, ha="center", va="bottom", fontsize=ann_fs,
                        color=ann_color, bbox=ann_bbox, zorder=5)
        end

        max_below = maximum(cpu_dy) + 0.5
        max_above = maximum(gpu_dy) + 0.7
        ax_brk.set_yticks([y_cpu, y_gpu])
        ax_brk.set_yticklabels(["{\\rm CPU}", "{\\rm GPU (Float64)}"])
        ax_brk.set_xlim(-0.2, 100.3)
        ax_brk.set_ylim(y_cpu - max_below, y_gpu + max_above)
        ax_brk.set_xlabel("{\\rm Fraction of per-tile time [\\%]}")
        ax_brk.set_title(@sprintf("{\\rm Per-tile breakdown (}\$N_\\lambda\${\\rm =%d, }\$N_{\\rm atm}\${\\rm =%d)}", Nλ, Natm))

        fig.tight_layout()
        fig.savefig(joinpath(PLOTDIR, "benchmark_pertile.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        println("Saved: benchmark_pertile.png")
    end
catch e
    println("Skipping per-tile/e2e plot: ", e)
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Threading scaling
# ══════════════════════════════════════════════════════════════════════════════
try
    _, rows = read_csv(joinpath(DATADIR, "threading_scaling.csv"))

    sorted_nt = [parse(Int, r[1]) for r in rows]
    median_s = [parse(Float64, r[2]) for r in rows]
    min_s = [parse(Float64, r[3]) for r in rows]
    max_s = [parse(Float64, r[4]) for r in rows]
    speedups = [parse(Float64, r[5]) for r in rows]

    # read Nϕ from e2e CSV if available
    Nϕ = 128
    try
        _, e2e_rows = read_csv(joinpath(DATADIR, "e2e_timings.csv"))
        # handle both old (col 3) and new (col 4) CSV formats
        Nϕ = length(e2e_rows[1]) >= 8 ? Int(parse(Float64, e2e_rows[1][4])) :
                                         Int(parse(Float64, e2e_rows[1][3]))
    catch; end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.errorbar(sorted_nt, speedups, yerr=0.0,
                 fmt="o-", color=COL_CPU_MT, lw=2, ms=6, capsize=3,
                 ecolor=COL_CPU_MT, label="{\\rm Measured}")
    ax1.plot([1, maximum(sorted_nt)], [1, maximum(sorted_nt)], "--",
             color="#999999", lw=1, label="{\\rm Ideal}")
    ax1.set_xlabel("{\\rm Number of threads}")
    ax1.set_ylabel("{\\rm Speedup}")
    ax1.set_title("{\\rm Threading scaling}")
    ax1.legend()
    ax1.set_xlim(0, maximum(sorted_nt) + 1)
    ax1.set_ylim(0, maximum(sorted_nt) + 1)

    ax2.errorbar(sorted_nt, median_s, yerr=0.0,
                 fmt="s-", color=COL_CPU_MT, lw=2, ms=6, capsize=3, ecolor=COL_CPU_MT)
    ax2.set_xlabel("{\\rm Number of threads}")
    ax2.set_ylabel("{\\rm Wall-clock time [s]}")
    ax2.set_title(@sprintf("{\\rm Disk integration (}\$N_\\phi\${\\rm =%d)}", Nϕ))

    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "benchmark_threading.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    println("Saved: benchmark_threading.png")
catch e
    println("Skipping threading plot: ", e)
end

# ══════════════════════════════════════════════════════════════════════════════
# 4. Nλ scaling (CPU 1T, CPU NT, GPU64, GPU32)
# ══════════════════════════════════════════════════════════════════════════════
try
    _, rows = read_csv(joinpath(DATADIR, "nlambda_scaling.csv"))

    # parse into per-series arrays
    series = Dict{String, Tuple{Vector{Int}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    for r in rows
        backend = r[1]
        threads = parse(Int, r[2])
        Nlambda = parse(Int, r[4])
        med = parse(Float64, r[5])
        mn = parse(Float64, r[6])
        mx = parse(Float64, r[7])

        if backend == "gpu_float32"
            key = "GPU (Float32)"
        elseif backend == "gpu_float64" || backend == "gpu"
            key = "GPU (Float64)"
        elseif threads == 1
            key = "CPU (1 thread)"
        else
            key = @sprintf("CPU (%d threads)", threads)
        end

        if !haskey(series, key)
            series[key] = (Int[], Float64[], Float64[], Float64[])
        end
        push!(series[key][1], Nlambda)
        push!(series[key][2], med)
        push!(series[key][3], mn)
        push!(series[key][4], mx)
    end

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # styling per series
    style = Dict(
        "CPU (1 thread)" => (color=COL_CPU_1T, marker="^", ls="--"),
        "GPU (Float64)"  => (color=COL_GPU64, marker="s", ls="-"),
        "GPU (Float32)"  => (color=COL_GPU32, marker="D", ls="-"),
    )
    # find the multi-thread key dynamically
    cpu_mt_key = ""
    for k in keys(series)
        if startswith(k, "CPU") && k != "CPU (1 thread)"
            cpu_mt_key = k
        end
    end
    if !isempty(cpu_mt_key)
        style[cpu_mt_key] = (color=COL_CPU_MT, marker="o", ls="-")
    end

    # plot order
    plot_order = filter(k -> haskey(series, k),
        ["CPU (1 thread)", cpu_mt_key, "GPU (Float64)", "GPU (Float32)"])

    for key in plot_order
        isempty(key) && continue
        Nλs, meds, mins, maxs = series[key]
        idx = sortperm(Nλs)
        Nλs, meds, mins, maxs = Nλs[idx], meds[idx], mins[idx], maxs[idx]
        s = style[key]
        yerr_lo = meds .- mins
        yerr_hi = maxs .- meds
        ax.errorbar(Nλs, meds, yerr=(yerr_lo, yerr_hi),
                    fmt=string(s.marker, s.ls), color=s.color, lw=2, ms=7,
                    capsize=3, ecolor=s.color, label="{\\rm " * key * "}")
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("\$N_\\lambda\$")
    ax.set_ylabel("{\\rm Wall-clock time [s]}")
    ax.set_title(@sprintf("{\\rm Performance vs. }\$N_\\lambda\$ {\\rm (}\$N_\\phi\${\\rm =%d)}", 128))
    ax.legend()

    ticker = PythonPlot.pyimport("matplotlib.ticker")
    for a in [ax.xaxis, ax.yaxis]
        a.set_major_formatter(ticker.ScalarFormatter())
        a.get_major_formatter().set_scientific(false)
    end
    ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
    ax.grid(true, which="minor", color="#EEEEEE", lw=0.3)

    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "benchmark_nlambda.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    println("Saved: benchmark_nlambda.png")
catch e
    println("Skipping Nλ scaling plot: ", e)
end

println()
println("DONE")
