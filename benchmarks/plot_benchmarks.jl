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
    # read Nλ from convolution metadata
    Nλ_conv = 0
    try
        _, meta = read_csv(joinpath(DATADIR, "convolution_meta.csv"))
        Nλ_conv = parse(Int, meta[1][1])
    catch; end
    if Nλ_conv > 0
        ax.set_title(@sprintf("{\\rm Convolution kernel timings (}\$N_\\lambda\${\\rm =%d)}", Nλ_conv))
    else
        ax.set_title("{\\rm Convolution kernel timings}")
    end
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
    # read per-tile data
    _, pt_rows = read_csv(joinpath(DATADIR, "pertile_timings.csv"))
    steps = [r[1] for r in pt_rows]
    # detect old (4-col: step,cpu,gpu64,gpu32) vs new (7-col: step,cpu_med,cpu_iqr,gpu64_med,...)
    has_iqr = length(pt_rows[1]) >= 7
    cpu_ms   = [parse(Float64, r[2]) for r in pt_rows]
    gpu64_ms = [parse(Float64, r[has_iqr ? 4 : 3]) for r in pt_rows]
    gpu32_ms = [parse(Float64, r[has_iqr ? 6 : 4]) for r in pt_rows]

    # read metadata (Nλ, Natm, etc.)
    _, meta_rows = read_csv(joinpath(DATADIR, "pertile_meta.csv"))
    Nλ = parse(Int, meta_rows[1][1])
    Natm = parse(Int, meta_rows[1][2])
    B_gpu = parse(Int, meta_rows[1][5])

    has_gpu = any(g -> g > 0.0, gpu64_ms)
    has_gpu32 = has_gpu && any(g -> g > 0.0, gpu32_ms)
    if !has_gpu
        println("Skipping per-tile plot: no GPU data")
    else

        cpu_total   = sum(cpu_ms)
        gpu64_total = sum(gpu64_ms)
        cpu_pct   = cpu_ms   ./ cpu_total   .* 100.0
        gpu64_pct = gpu64_ms ./ gpu64_total .* 100.0
        if has_gpu32
            gpu32_total = sum(gpu32_ms)
            gpu32_pct = gpu32_ms ./ gpu32_total .* 100.0
        end

        fig, ax_brk = plt.subplots(figsize=(7, has_gpu32 ? 7.0 : 4.0))

        step_labels = ["{\\rm Microturbulence}", "{\\rm Optical depth}",
                       "{\\rm Contribution fn.}", "{\\rm Macroturbulence}"]
        colors_cpu  = ["#B8DFFB", COL_CPU_1T, COL_CPU_MT, "#1A7AB5"]
        colors_g64  = ["#FFD3A6", "#F5A66E", COL_GPU64, "#B8432E"]
        colors_g32  = ["#66D9B2", COL_GPU32, "#007A59", "#004D38"]

        bar_h = 0.55
        y_cpu   = 0
        y_gpu64 = has_gpu32 ? 2.0 : 1.0
        y_gpu32 = has_gpu32 ? 4.0 : -1.0
        y_top   = has_gpu32 ? y_gpu32 : y_gpu64

        ax_brk.set_axisbelow(true)
        ax_brk.grid(true, axis="x", color="#DDDDDD", lw=0.5)
        ax_brk.grid(false, axis="y")

        # helper: draw one stacked bar row (no inside labels)
        function draw_row!(ax, y, pcts, colors)
            left = 0.0
            for (i, pct) in enumerate(pcts)
                ax.barh(y, pct, left=left, color=colors[i],
                        edgecolor="white", height=bar_h, zorder=3)
                left += pct
            end
        end

        draw_row!(ax_brk, y_cpu,   cpu_pct,   colors_cpu)
        draw_row!(ax_brk, y_gpu64, gpu64_pct, colors_g64)
        if has_gpu32
            draw_row!(ax_brk, y_gpu32, gpu32_pct, colors_g32)
        end

        # total time annotations (below each bar label)
        ax_brk.text(-0.01, y_cpu - 0.22, @sprintf("{\\rm (%.1f ms)}", cpu_total),
                    ha="right", va="top", fontsize=7, color="#555555",
                    transform=ax_brk.get_yaxis_transform(), zorder=4)
        ax_brk.text(-0.01, y_gpu64 - 0.22, @sprintf("{\\rm (%.2f ms)}", gpu64_total),
                    ha="right", va="top", fontsize=7, color="#555555",
                    transform=ax_brk.get_yaxis_transform(), zorder=4)
        if has_gpu32
            ax_brk.text(-0.01, y_gpu32 - 0.22, @sprintf("{\\rm (%.2f ms)}", gpu32_total),
                        ha="right", va="top", fontsize=7, color="#555555",
                        transform=ax_brk.get_yaxis_transform(), zorder=4)
        end

        # leader-line annotations: step name + timing for each bar
        ann_color = "#000000"
        ann_fs = 7
        min_sep = 15.0
        base_dy = 0.3
        stagger_dy = 0.35
        ann_bbox = Dict("boxstyle" => "square,pad=0.05", "facecolor" => "white",
                         "edgecolor" => "none", "alpha" => 1.0)

        fmt_cpu = ms -> @sprintf("{\\rm %.2f ms}", ms)
        fmt_gpu = ms -> @sprintf("{\\rm %.2f ms}", ms)

        # helper: compute segment midpoints and staggered offsets
        function calc_mids_dy(pcts)
            x_mids = Float64[]
            cum = 0.0
            for pct in pcts
                push!(x_mids, cum + pct / 2)
                cum += pct
            end
            dy = fill(base_dy, length(pcts))
            for i in 2:length(pcts)
                if abs(x_mids[i] - x_mids[i-1]) < min_sep
                    dy[i] = dy[i-1] + stagger_dy
                end
            end
            return x_mids, dy
        end

        # helper: draw leader-line annotations above a bar
        function annotate_above!(ax, y, pcts, vals, fmt_fn, show_names)
            x_mids, dy = calc_mids_dy(pcts)
            for (i, x_mid) in enumerate(x_mids)
                ax.plot([x_mid, x_mid],
                        [y + bar_h / 2, y + bar_h / 2 + dy[i]],
                        color="#999999", lw=0.5, zorder=4)
                label = show_names ? step_labels[i] * "\n" * fmt_fn(vals[i]) :
                                     fmt_fn(vals[i])
                ax.text(x_mid, y + bar_h / 2 + dy[i],
                        label, ha="center", va="bottom", fontsize=ann_fs,
                        color=ann_color, bbox=ann_bbox, zorder=5)
            end
            return maximum(dy)
        end

        # all annotations above their respective bars, all with step names
        max_cpu_dy = annotate_above!(ax_brk, y_cpu, cpu_pct, cpu_ms, fmt_cpu, true)
        max_g64_dy = annotate_above!(ax_brk, y_gpu64, gpu64_pct, gpu64_ms, fmt_gpu, true)

        if has_gpu32
            max_top_dy = annotate_above!(ax_brk, y_gpu32, gpu32_pct, gpu32_ms, fmt_gpu, true)
        else
            max_top_dy = max_g64_dy
        end

        max_below = 0.5
        max_above = max_top_dy + 1.0
        yticks = [y_cpu, y_gpu64]
        ylabels = ["{\\rm CPU}", "{\\rm GPU (Float64)}"]
        if has_gpu32
            push!(yticks, y_gpu32)
            push!(ylabels, "{\\rm GPU (Float32)}")
        end
        ax_brk.set_yticks(yticks)
        ax_brk.set_yticklabels(ylabels)
        ax_brk.set_xlim(-0.2, 100.3)
        ax_brk.set_ylim(y_cpu - max_below, y_top + max_above)
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

    # read Nϕ and Nλ from metadata
    Nϕ = 128
    Nλ_thread = 0
    try
        _, meta = read_csv(joinpath(DATADIR, "pertile_meta.csv"))
        Nλ_thread = parse(Int, meta[1][1])
        Nϕ = parse(Int, meta[1][3])
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
    if Nλ_thread > 0
        ax2.set_title(@sprintf("{\\rm Disk integration (}\$N_\\phi\${\\rm =%d, }\$N_\\lambda\${\\rm =%d)}", Nϕ, Nλ_thread))
    else
        ax2.set_title(@sprintf("{\\rm Disk integration (}\$N_\\phi\${\\rm =%d)}", Nϕ))
    end

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
