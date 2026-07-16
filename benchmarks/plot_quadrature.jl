# Render the quadrature-benchmark CSVs (benchmarks/data/quadrature_*.csv, written by
# benchmark_quadrature.jl) into figures under docs/src/static/.
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

function read_csv(path)
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    header = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end]]
    return header, rows
end

col(rows, j; T=Float64) = [parse(T, r[j]) for r in rows]

function nolog_sci!(ax)
    ticker = PythonPlot.pyimport("matplotlib.ticker")
    for a in (ax.xaxis, ax.yaxis)
        a.set_major_formatter(ticker.ScalarFormatter())
        a.get_major_formatter().set_scientific(false)
    end
end

# ── 1. convergence vs Nμ and N_az ────────────────────────────────────────────
try
    _, nmu = read_csv(joinpath(DATADIR, "quadrature_nmu.csv"))
    _, naz = read_csv(joinpath(DATADIR, "quadrature_naz.csv"))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    function panel!(ax, x, emax, emean, t, xlabel)
        ax.plot(x, emax,  "o-",  color=COL_QUAD, lw=2, ms=6, label="{\\rm max}")
        ax.plot(x, emean, "s--", color=COL_QUAD, lw=1.6, ms=5, label="{\\rm mean}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("{\\rm formation-}\$T\${\\rm\\ error\\ vs.\\ :disk\\ [K]}")
        ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
        nolog_sci!(ax)
        axt = ax.twinx()
        axt.plot(x, t, "^:", color=COL_TIME, lw=1.4, ms=5, label="{\\rm time}")
        axt.set_ylabel("{\\rm time\\ [ms]}", color=COL_TIME)
        axt.tick_params(axis="y", colors=COL_TIME)
        axt.grid(false)
        return axt
    end

    axtL = panel!(axL, col(nmu, 1), col(nmu, 2), col(nmu, 3), col(nmu, 5), "\$N_\\mu\$")
    axtR = panel!(axR, col(naz, 1), col(naz, 2), col(naz, 3), col(naz, 5), "\$N_{\\rm az}\$")
    axL.set_title("{\\rm Convergence vs.\\ }\$N_\\mu\$ {\\rm (}\$N_{\\rm az}\${\\rm =256)}")
    axR.set_title("{\\rm Convergence vs.\\ }\$N_{\\rm az}\$ {\\rm (}\$N_\\mu\${\\rm =16)}")
    # combined legend (proxy handles — the two lines live on different axes)
    Line2D = PythonPlot.pyimport("matplotlib.lines").Line2D
    leg = [Line2D([0], [0]; color=COL_QUAD, marker="o", ls="-",  label="{\\rm max}"),
           Line2D([0], [0]; color=COL_QUAD, marker="s", ls="--", label="{\\rm mean}"),
           Line2D([0], [0]; color=COL_TIME, marker="^", ls=":",  label="{\\rm time}")]
    axL.legend(handles=leg, loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "quadrature_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    println("Saved: quadrature_convergence.png")
catch e
    println("Skipping convergence plot: ", e)
end

# ── 2. accuracy vs vsini (quadrature vs hirano) ──────────────────────────────
try
    _, rows = read_csv(joinpath(DATADIR, "quadrature_vsini.csv"))
    methods = unique([String(r[2]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stylemap = Dict("quadrature" => (COL_QUAD, "o-", "{\\rm :quadrature}"),
                    "hirano"     => (COL_HIRANO, "s--", "{\\rm :hirano}"))
    for m in ("quadrature", "hirano")
        sel = [r for r in rows if String(r[2]) == m]
        isempty(sel) && continue
        vk = [parse(Float64, r[1]) for r in sel]
        emax = [parse(Float64, r[3]) for r in sel]
        c, fmt, lab = stylemap[m]
        ax.plot(vk, emax, fmt, color=c, lw=2, ms=6, label=lab)
    end
    ax.set_yscale("log")
    ax.set_xlabel("\$v\\sin i\$ {\\rm [km\\ s}\$^{-1}\${\\rm ]}")
    ax.set_ylabel("{\\rm worst-pixel formation-}\$T\${\\rm\\ error\\ vs.\\ :disk\\ [K]}")
    ax.set_title("{\\rm Accuracy vs.\\ }\$v\\sin i\$")
    ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "quadrature_vsini.png"), dpi=150, bbox_inches="tight")
    plt.close()
    println("Saved: quadrature_vsini.png")
catch e
    println("Skipping vsini plot: ", e)
end

# ── 3. speed scaling vs Nλ (method × device) ─────────────────────────────────
try
    _, rows = read_csv(joinpath(DATADIR, "quadrature_scaling.csv"))
    colormap = Dict("disk" => COL_DISK, "quadrature" => COL_QUAD, "hirano" => COL_HIRANO)
    devsty = Dict("cpu" => ("--", "o"), "gpu" => ("-", "s"))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for meth in ("disk", "quadrature", "hirano"), dev in ("cpu", "gpu")
        sel = [r for r in rows if String(r[1]) == meth && String(r[2]) == dev]
        isempty(sel) && continue
        nl = [parse(Int, r[3]) for r in sel]; t = [parse(Float64, r[4]) for r in sel]
        idx = sortperm(nl)
        ls, mk = devsty[dev]
        ax.plot(nl[idx], t[idx], string(mk, ls), color=colormap[meth], lw=2, ms=6,
                label="{\\rm " * meth * " (" * uppercase(dev) * ")}")
    end
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("\$N_\\lambda\$")
    ax.set_ylabel("{\\rm wall-clock time\\ [ms]}")
    ax.set_title("{\\rm Speed vs.\\ }\$N_\\lambda\$")
    ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
    ax.grid(true, which="minor", color="#EEEEEE", lw=0.3)
    nolog_sci!(ax)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(joinpath(PLOTDIR, "quadrature_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
    println("Saved: quadrature_scaling.png")
catch e
    println("Skipping scaling plot: ", e)
end

println("DONE")
