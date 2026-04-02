using Revise
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Printf

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pylist
np = pyimport("numpy")
plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
plt.ioff()

# convert Julia arrays to numpy for matplotlib compatibility
py(x::AbstractVector) = np.asarray(collect(Float64, x))

plotdir = joinpath(FT.moddir, "docs", "src", "static")
mkpath(plotdir)

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]

# cut on species
linelist = linelist[specs .== "Fe I"]

# get the Fe I 6301 & 6302 lines (just cuz)
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

# stellar params
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0
Δλ = 0.0025
u1 = 0.43
u2 = 0.31
Nϕ = 128

star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H,
                    vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# ── helper ────────────────────────────────────────────────────────────────────
function format_max_resid(resids)
    v = maximum(abs.(resids))
    if v == 0.0
        return "\$0\$"
    end
    e = floor(Int, log10(v))
    m = v / 10.0^e
    return @sprintf("\$%.1f \\times 10^{%d}\$", m, e)
end

# consistent palette (matches plot_benchmarks.jl)
const COL_GPU64 = "#D55E00"  # orange
const COL_GPU32 = "#009E73"  # green

# ── compute spectra ───────────────────────────────────────────────────────────
function run_comparison(; convolve::Bool, label::String)
    kw = Dict(:Δλ => Δλ, :Nϕ => Nϕ, :showprogress => false, :ne_warn_thresh => Inf)
    if convolve
        kw[:convolve] = true
        kw[:u1] = u1
        kw[:u2] = u2
    end

    println("Computing CPU Float64 ($label)...")
    res_cpu = calc_formation_temp(star, linelist; use_gpu=false, kw...)

    println("Computing GPU Float64 ($label)...")
    res_gpu64 = calc_formation_temp(star, linelist; use_gpu=true,
                                    gpu_precision=Float64, kw...)

    println("Computing GPU Float32 ($label)...")
    res_gpu32 = calc_formation_temp(star, linelist; use_gpu=true,
                                    gpu_precision=Float32, kw...)

    return res_cpu, res_gpu64, res_gpu32
end

# ── plotting ──────────────────────────────────────────────────────────────────
function make_plot(res_cpu, res_gpu64, res_gpu32; title_str, filename)
    # convert everything to Float64 then numpy
    wavs = Float64.(res_cpu.wavs)
    flux_cpu = Float64.(res_cpu.flux)
    flux_64 = Float64.(res_gpu64.flux)
    flux_32 = Float64.(res_gpu32.flux)
    ft_cpu = Float64.(res_cpu.form_temps)
    ft_64 = Float64.(res_gpu64.form_temps)
    ft_32 = Float64.(res_gpu32.form_temps)

    λ0 = mean(wavs)

    # edge mask for formation temps
    edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
    interior = (edge_px + 1):(length(wavs) - edge_px)

    # residuals
    flux_resid_64 = flux_64 .- flux_cpu
    flux_resid_32 = flux_32 .- flux_cpu
    ft_resid_64 = ft_64 .- ft_cpu
    ft_resid_32 = ft_32 .- ft_cpu

    symlog = pyimport("matplotlib.scale")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
    ax_flux = axes[0, 0]
    ax_ft = axes[0, 1]
    ax_fr = axes[1, 0]
    ax_ftr = axes[1, 1]

    ms = 3.0
    xl = (6300.0, 6304.0)

    # row 1: spectra
    ax_flux.plot(py(wavs), py(flux_cpu), "-", color="k", lw=1.5,
                 label="{\\rm CPU (Float64)}", zorder=0)
    ax_flux.scatter(py(wavs), py(flux_64), marker="s",
                    alpha=0.7, c=COL_GPU64, s=ms,
                    label="{\\rm GPU (Float64)}", zorder=1)
    ax_flux.scatter(py(wavs), py(flux_32), marker="^",
                    alpha=0.7, c=COL_GPU32, s=ms,
                    label="{\\rm GPU (Float32)}", zorder=2)
    ax_flux.set_ylabel("{\\rm Normalized Flux}")
    ax_flux.set_xlim(xl...)

    ax_ft.plot(py(wavs), py(ft_cpu), "-", color="k", lw=1.5, zorder=0)
    ax_ft.scatter(py(wavs), py(ft_64), marker="s",
                  alpha=0.7, c=COL_GPU64, s=ms, zorder=1)
    ax_ft.scatter(py(wavs), py(ft_32), marker="^",
                  alpha=0.7, c=COL_GPU32, s=ms, zorder=2)
    ax_ft.set_ylabel("{\\rm Formation Temperature [K]}")
    ax_ft.set_xlim(xl...)

    # row 2: residuals (both precisions overplotted, symlog y-scale)
    ax_fr.scatter(py(wavs), py(flux_resid_64), s=4, marker="s", c=COL_GPU64, alpha=0.8,
                  label="{\\rm GPU64}")
    ax_fr.scatter(py(wavs), py(flux_resid_32), s=4, marker="^", c=COL_GPU32, alpha=0.8,
                  label="{\\rm GPU32}")
    ax_fr.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_fr.set_yscale("symlog", linthresh=1e-6)
    ax_fr.set_ylabel("{\\rm CPU \$-\$ GPU}")
    ax_fr.set_xlabel("{\\rm Wavelength [\\AA]}")
    ax_fr.legend(fontsize=9)
    ax_fr.set_xlim(xl...)
    fr_ext = max(maximum(abs.(flux_resid_64)), maximum(abs.(flux_resid_32)))
    ax_fr.set_ylim(-1.5 * fr_ext, 1.5 * fr_ext)

    ax_ftr.scatter(py(wavs[interior]), py(ft_resid_64[interior]), s=4, marker="s",
                   c=COL_GPU64, alpha=0.8, label="{\\rm GPU64}")
    ax_ftr.scatter(py(wavs[interior]), py(ft_resid_32[interior]), s=4, marker="^",
                   c=COL_GPU32, alpha=0.8, label="{\\rm GPU32}")
    ax_ftr.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_ftr.set_yscale("symlog", linthresh=0.01)
    ax_ftr.set_ylabel("{\\rm CPU \$-\$ GPU}")
    ax_ftr.set_xlabel("{\\rm Wavelength [\\AA]}")
    ax_ftr.legend(fontsize=9)
    ax_ftr.set_xlim(xl...)
    ftr_ext = max(maximum(abs.(ft_resid_64[interior])), maximum(abs.(ft_resid_32[interior])))
    ax_ftr.set_ylim(-1.5 * ftr_ext, 1.5 * ftr_ext)

    # hide x tick labels on top row
    ax_flux.tick_params(labelbottom=false)
    ax_ft.tick_params(labelbottom=false)

    # legend on top row
    leg = ax_flux.legend(loc="lower left", mode="expand", ncol=3, fontsize=11,
                         bbox_to_anchor=pylist([0, 1.02, 2.12, 0.2]), handletextpad=0.3,
                         title=title_str, title_fontsize=13)
    for lh in leg.legend_handles
        lh._sizes = pylist([20.0])
    end

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(joinpath(plotdir, filename), bbox_inches="tight", dpi=150)
    plt.close()
    println("Saved: ", joinpath(plotdir, filename))
end

# ── run ───────────────────────────────────────────────────────────────────────
println("="^60)
println("GPU PRECISION COMPARISON")
println("="^60)
println()

# convolution path (Hirano)
res_cpu_c, res_gpu64_c, res_gpu32_c = run_comparison(convolve=true, label="convolve")
println()

# disk integration path
res_cpu_d, res_gpu64_d, res_gpu32_d = run_comparison(convolve=false, label="disk integration")
println()

# make plots
make_plot(res_cpu_c, res_gpu64_c, res_gpu32_c;
          title_str=@sprintf("{\\rm Hirano convolution (}\$N_\\lambda\${\\rm =%d, }\$\\Delta\\lambda\${\\rm =%.3f \\AA)}", length(res_cpu_c.wavs), Δλ),
          filename="gpu_precision_convolve.png")

make_plot(res_cpu_d, res_gpu64_d, res_gpu32_d;
          title_str=@sprintf("{\\rm Disk integration (}\$N_\\phi\${\\rm =%d, }\$N_\\lambda\${\\rm =%d, }\$\\Delta\\lambda\${\\rm =%.3f \\AA)}", Nϕ, length(res_cpu_d.wavs), Δλ),
          filename="gpu_precision_diskint.png")

# print summary statistics
println()
println("="^60)
println("SUMMARY")
println("="^60)
for (label, rc, r64, r32) in [("Convolve", res_cpu_c, res_gpu64_c, res_gpu32_c),
                                ("Disk int", res_cpu_d, res_gpu64_d, res_gpu32_d)]
    λ0 = mean(Float64.(rc.wavs))
    edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
    interior = (edge_px + 1):(length(rc.wavs) - edge_px)

    println()
    println("  $label:")
    @printf("    GPU64 flux   max|resid| = %.2e   mean = %.2e\n",
            maximum(abs.(Float64.(r64.flux) .- Float64.(rc.flux))),
            mean(abs.(Float64.(r64.flux) .- Float64.(rc.flux))))
    @printf("    GPU32 flux   max|resid| = %.2e   mean = %.2e\n",
            maximum(abs.(Float64.(r32.flux) .- Float64.(rc.flux))),
            mean(abs.(Float64.(r32.flux) .- Float64.(rc.flux))))
    @printf("    GPU64 Tform  max|resid| = %.2f K   mean = %.2f K  (interior)\n",
            maximum(abs.(Float64.(r64.form_temps[interior]) .- Float64.(rc.form_temps[interior]))),
            mean(abs.(Float64.(r64.form_temps[interior]) .- Float64.(rc.form_temps[interior]))))
    @printf("    GPU32 Tform  max|resid| = %.2f K   mean = %.2f K  (interior)\n",
            maximum(abs.(Float64.(r32.form_temps[interior]) .- Float64.(rc.form_temps[interior]))),
            mean(abs.(Float64.(r32.form_temps[interior]) .- Float64.(rc.form_temps[interior]))))
end

println()
println("DONE")
