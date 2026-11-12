# Render the quadrature-benchmark CSVs (benchmarks/data/quadrature_*.csv, written by
# benchmark_quadrature.jl) into figures under docs/src/static/.
#
# A failed panel is reported and the script exits nonzero, so run_all.jl records it. Silently
# skipping leaves the previously committed figure in place, which publishes stale numbers.
#
# Run:  julia --project=. benchmarks/plot_quadrature.jl

using Printf, Statistics
using PythonPlot; plt = PythonPlot.pyplot

const PROJECT_DIR = dirname(@__DIR__)
const DATADIR = joinpath(PROJECT_DIR, "benchmarks", "data")
const PLOTDIR = joinpath(PROJECT_DIR, "docs", "src", "static")
!isdir(PLOTDIR) && mkpath(PLOTDIR)

stylefile = joinpath(PROJECT_DIR, "fig.mplstyle")
isfile(stylefile) && plt.style.use(stylefile)
plt.ioff()

const COL_DISK   = "#2A96D1"   # blue
const COL_QUAD   = "#D55E00"   # orange
const COL_HIRANO = "#009E73"   # green
const COL_TIME   = "#777777"   # grey (secondary axis)

const failures = String[]

function read_csv(path)
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    header = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end]]
    return header, rows
end

col(rows, j; T=Float64) = [parse(T, r[j]) for r in rows]

"""
    icol(header, name)

Column index by header name, so a schema change fails here rather than silently plotting the
wrong column. The CSVs and this script are edited at different times; positional indices have
already gone stale once.
"""
function icol(header, name)
    j = findfirst(==(name), strip.(header))
    j === nothing && error("column '$name' not in CSV header $(join(header, ','))")
    return j
end

function nolog_sci!(ax)
    ticker = PythonPlot.pyimport("matplotlib.ticker")
    for a in (ax.xaxis, ax.yaxis)
        a.set_major_formatter(ticker.ScalarFormatter())
        a.get_major_formatter().set_scientific(false)
    end
end

# run a panel, recording rather than swallowing a failure.
# `f` is first: `panel("x.png") do ... end` passes the closure as the first argument.
function panel(f, name)
    try
        f()
        println("Saved: ", name)
    catch e
        printstyled("FAILED: $name -- $e\n", color=:red)
        push!(failures, name)
    end
end

# ── 1. convergence vs Nμ ─────────────────────────────────────────────────────
panel("quadrature_convergence.png") do
    hn, nmu = read_csv(joinpath(DATADIR, "quadrature_nmu.csv"))
    x = col(nmu, icol(hn, "Nmu"))
    emax, emean = col(nmu, icol(hn, "formT_max")), col(nmu, icol(hn, "formT_mean"))
    tmin, tmed, tmax = col(nmu, icol(hn, "time_min_ms")), col(nmu, icol(hn, "time_med_ms")),
                       col(nmu, icol(hn, "time_max_ms"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, emax,  "o-",  color=COL_QUAD, lw=2, ms=6)
    ax.plot(x, emean, "s--", color=COL_QUAD, lw=1.6, ms=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("\$N_\\mu\$")
    ax.set_ylabel("{\\rm formation-}\$T\${\\rm\\ error\\ vs.\\ :disk\\ [K]}")
    ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
    nolog_sci!(ax)
    # tick the sampled Nμ values: the axis spans little more than a decade, where the log
    # locator labels its minor ticks and the labels collide
    ticker = PythonPlot.pyimport("matplotlib.ticker")
    ax.set_xticks(x)
    ax.set_xticklabels([@sprintf("\$%d\$", n) for n in x])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # cost on a secondary axis; error bars span the timing rounds
    axt = ax.twinx()
    axt.errorbar(x, tmed, yerr=(tmed .- tmin, tmax .- tmed), fmt="^:", color=COL_TIME,
                 lw=1.4, ms=5, capsize=3, ecolor=COL_TIME)
    axt.set_ylabel("{\\rm time\\ [ms]}", color=COL_TIME)
    axt.tick_params(axis="y", colors=COL_TIME)
    axt.grid(false)

    # combined legend (proxy handles: the lines live on different axes)
    Line2D = PythonPlot.pyimport("matplotlib.lines").Line2D
    leg = [Line2D([0], [0]; color=COL_QUAD, marker="o", ls="-",  label="{\\rm max}"),
           Line2D([0], [0]; color=COL_QUAD, marker="s", ls="--", label="{\\rm mean}"),
           Line2D([0], [0]; color=COL_TIME, marker="^", ls=":",  label="{\\rm time}")]
    ax.legend(handles=leg, loc="upper center", fontsize=9)
    ax.set_title("{\\rm Convergence vs.\\ }\$N_\\mu\$ {\\rm (}\$v\\sin i\${\\rm =15\\ km\\ s}\$^{-1}\${\\rm )}")

    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "quadrature_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
end

# ── 2. accuracy vs vsini (quadrature vs hirano) ──────────────────────────────
panel("quadrature_vsini.png") do
    h, rows = read_csv(joinpath(DATADIR, "quadrature_vsini.csv"))
    jv, jm, je = icol(h, "vsini_kms"), icol(h, "method"), icol(h, "formT_max")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stylemap = Dict("quadrature" => (COL_QUAD, "o-", "{\\rm :quadrature}"),
                    "hirano"     => (COL_HIRANO, "s--", "{\\rm :hirano}"))
    for m in ("quadrature", "hirano")
        sel = [r for r in rows if String(strip(r[jm])) == m]
        isempty(sel) && continue
        vk = [parse(Float64, r[jv]) for r in sel]
        emax = [parse(Float64, r[je]) for r in sel]
        c, fmt, lab = stylemap[m]
        ax.plot(vk, emax, fmt, color=c, lw=2, ms=6, label=lab)
    end
    ax.set_yscale("log")
    ax.set_xlabel("\$v\\sin i\$ {\\rm [km\\ s}\$^{-1}\${\\rm ]}")
    ax.set_ylabel("{\\rm worst-pixel formation-}\$T\${\\rm\\ error\\ vs.\\ :disk\\ [K]}")
    ax.set_title("{\\rm Accuracy vs.\\ }\$v\\sin i\$ {\\rm (}\$N_\\mu\${\\rm =32)}")
    ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "quadrature_vsini.png"), dpi=150, bbox_inches="tight")
    plt.close()
end

# ── 3. accuracy vs wavelength sampling ───────────────────────────────────────
panel("quadrature_grid.png") do
    h, rows = read_csv(joinpath(DATADIR, "quadrature_grid.csv"))
    jd, jm = icol(h, "dlambda_A"), icol(h, "method")
    jmax, jmean = icol(h, "formT_max"), icol(h, "formT_mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (m, c, lab) in (("quadrature", COL_QUAD, ":quadrature"),
                        ("hirano", COL_HIRANO, ":hirano"))
        sel = [r for r in rows if String(strip(r[jm])) == m]
        isempty(sel) && continue
        dλ = [parse(Float64, r[jd]) for r in sel]
        idx = sortperm(dλ)
        ax.plot(dλ[idx], [parse(Float64, r[jmax]) for r in sel][idx], "o-", color=c, lw=2,
                ms=6, label="{\\rm " * lab * " (max)}")
        ax.plot(dλ[idx], [parse(Float64, r[jmean]) for r in sel][idx], "s--", color=c,
                lw=1.5, ms=5, alpha=0.75, label="{\\rm " * lab * " (mean)}")
    end
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()                     # finer sampling to the right, i.e. "more effort"
    # tick the sampled Δλ values: the log majors give only 1e-2 over this narrow range, which
    # leaves the axis unreadable against the values quoted in the docs
    dλ_ticks = sort(unique([parse(Float64, r[jd]) for r in rows]))
    ticker = PythonPlot.pyimport("matplotlib.ticker")
    ax.set_xticks(dλ_ticks)
    ax.set_xticklabels([@sprintf("%g", d) for d in dλ_ticks])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("\$\\Delta\\lambda\$ {\\rm [\\AA]}")
    ax.set_ylabel("{\\rm formation-}\$T\${\\rm\\ error\\ vs.\\ :disk\\ [K]}")
    ax.set_title("{\\rm Accuracy vs.\\ wavelength sampling (}\$v\\sin i\${\\rm =15\\ km\\ s}\$^{-1}\${\\rm , }\$N_\\mu\${\\rm =32)}")
    ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "quadrature_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
end

# ── 4. speed scaling vs Nλ (method × device) ─────────────────────────────────
panel("quadrature_scaling.png") do
    h, rows = read_csv(joinpath(DATADIR, "quadrature_scaling.csv"))
    jm, jd, jn = icol(h, "method"), icol(h, "device"), icol(h, "Nlambda")
    jlo, jmed, jhi = icol(h, "time_min_ms"), icol(h, "time_med_ms"), icol(h, "time_max_ms")
    colormap = Dict("disk" => COL_DISK, "quadrature" => COL_QUAD, "hirano" => COL_HIRANO)
    devsty = Dict("cpu" => ("--", "o"), "gpu" => ("-", "s"))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for meth in ("disk", "quadrature", "hirano"), dev in ("cpu", "gpu")
        sel = [r for r in rows if String(strip(r[jm])) == meth && String(strip(r[jd])) == dev]
        isempty(sel) && continue
        nl = [parse(Int, r[jn]) for r in sel]
        idx = sortperm(nl)
        med = [parse(Float64, r[jmed]) for r in sel][idx]
        lo  = [parse(Float64, r[jlo])  for r in sel][idx]
        hi  = [parse(Float64, r[jhi])  for r in sel][idx]
        ls, mk = devsty[dev]
        # error bars span the timing rounds: where they overlap between two methods, the
        # measurement does not resolve them
        ax.errorbar(nl[idx], med, yerr=(med .- lo, hi .- med), fmt=string(mk, ls),
                    color=colormap[meth], lw=2, ms=6, capsize=3, ecolor=colormap[meth],
                    label="{\\rm " * meth * " (" * uppercase(dev) * ")}")
    end
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("\$N_\\lambda\$")
    ax.set_ylabel("{\\rm wall-clock time\\ [ms]}")
    ax.set_title("{\\rm Speed vs.\\ }\$N_\\lambda\$ {\\rm (}\$N_\\phi\${\\rm =128, }\$N_\\mu\${\\rm =32)}")
    ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
    ax.grid(true, which="minor", color="#EEEEEE", lw=0.3)
    nolog_sci!(ax)
    # tick the sampled Nλ values: the minor tick labels collide on this narrow log range
    nl_ticks = sort(unique([parse(Int, r[jn]) for r in rows]))
    ticker = PythonPlot.pyimport("matplotlib.ticker")
    ax.set_xticks(nl_ticks)
    ax.set_xticklabels([@sprintf("\$%d\$", n) for n in nl_ticks])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "quadrature_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
end

if isempty(failures)
    println("DONE")
else
    printstyled("FAILED panels: ", join(failures, ", "), "\n", color=:red)
    exit(1)
end
