"""
Compare CPU (Float64), GPU (Float64), and GPU (Float32) spectra for both
the convolution and numerical disk integration paths.

Produces two plots:
    scripts/plots/gpu_precision_convolve.pdf
    scripts/plots/gpu_precision_diskint.pdf

Usage:
    julia --project=. scripts/gpu_precision_comparison.jl
"""
using Revise
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Printf

# plotting
import PythonPlot; plt = PythonPlot
plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
plt.ioff()

plotdir = joinpath(@__DIR__, "plots")
mkpath(plotdir)

# ── setup ─────────────────────────────────────────────────────────────────────
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist_fe = linelist[specs .== "Fe I"]
wls_all = [l.wl * 1e8 for l in linelist_fe]
idx_start = findfirst(x -> x >= 6298.0, wls_all)
idx_end   = findfirst(x -> x >= 6304.0, wls_all)
linelist = linelist_fe[idx_start:idx_end]

Teff  = 5777.0
logg  = 4.44
Fe_H  = 0.0
vsini = 2100.0
ζ_RT  = 3400.0
ξ     = 850.0
Δλ    = 0.01
u1    = 0.43
u2    = 0.31
Nϕ    = 64

star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H,
                    vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# ── helper ────────────────────────────────────────────────────────────────────
function format_max_resid(resids)
    v = maximum(abs.(resids))
    if v == 0.0
        return "\$0\$"
    end
    exp = floor(Int, log10(v))
    mantissa = v / 10.0^exp
    return @sprintf("\$%.1f \\times 10^{%d}\$", mantissa, exp)
end

colors = ["#56B4E9", "#E69F00", "#009E73"]

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
    wavs = res_cpu.wavs
    λ0 = mean(wavs)

    # edge mask for formation temps
    edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
    interior = (edge_px + 1):(length(wavs) - edge_px)

    # residuals
    flux_resid_64 = res_gpu64.flux .- res_cpu.flux
    flux_resid_32 = Float64.(res_gpu32.flux) .- res_cpu.flux
    ft_resid_64   = res_gpu64.form_temps .- res_cpu.form_temps
    ft_resid_32   = Float64.(res_gpu32.form_temps) .- res_cpu.form_temps

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=true)
    ax_flux     = axes[0, 0]
    ax_ft       = axes[0, 1]
    ax_fr64     = axes[1, 0]
    ax_ftr64    = axes[1, 1]
    ax_fr32     = axes[2, 0]
    ax_ftr32    = axes[2, 1]

    ms = 3.0

    # row 1: spectra
    ax_flux.plot(wavs, res_cpu.flux, "-", color="k", lw=1.5,
                 label="{\\rm CPU (Float64)}", zorder=0)
    ax_flux.scatter(Float64.(res_gpu64.wavs), res_gpu64.flux, marker="s",
                    alpha=0.7, c=colors[1], s=ms,
                    label="{\\rm GPU (Float64)}", zorder=1)
    ax_flux.scatter(Float64.(res_gpu32.wavs), Float64.(res_gpu32.flux), marker="^",
                    alpha=0.7, c=colors[2], s=ms,
                    label="{\\rm GPU (Float32)}", zorder=2)
    ax_flux.set_ylabel("{\\rm Normalized flux}")

    ax_ft.plot(wavs, res_cpu.form_temps, "-", color="k", lw=1.5, zorder=0)
    ax_ft.scatter(Float64.(res_gpu64.wavs), res_gpu64.form_temps, marker="s",
                  alpha=0.7, c=colors[1], s=ms, zorder=1)
    ax_ft.scatter(Float64.(res_gpu32.wavs), Float64.(res_gpu32.form_temps), marker="^",
                  alpha=0.7, c=colors[2], s=ms, zorder=2)
    ax_ft.set_ylabel("{\\rm Formation temp [K]}")

    # row 2: GPU Float64 residuals
    ax_fr64.scatter(wavs, flux_resid_64, s=4, marker="s", c=colors[1], alpha=0.8)
    ax_fr64.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_fr64.set_ylabel("{\\rm CPU \$-\$ GPU64 flux}")

    # inset histogram
    ax_fr64_h = ax_fr64.inset_axes([1.0015, 0, 0.06, 1], sharey=ax_fr64)
    ax_fr64_h.hist(flux_resid_64, bins="auto", density=true, histtype="step",
                   orientation="horizontal", color=colors[1])
    _style_hist_inset(ax_fr64_h)

    ax_ftr64.scatter(wavs[interior], ft_resid_64[interior], s=4, marker="s",
                     c=colors[1], alpha=0.8)
    ax_ftr64.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_ftr64.set_ylabel("{\\rm CPU \$-\$ GPU64 \$T_{\\rm form}\$ [K]}")

    ax_ftr64_h = ax_ftr64.inset_axes([1.0015, 0, 0.06, 1], sharey=ax_ftr64)
    ax_ftr64_h.hist(ft_resid_64[interior], bins="auto", density=true, histtype="step",
                    orientation="horizontal", color=colors[1])
    _style_hist_inset(ax_ftr64_h)

    # row 3: GPU Float32 residuals
    ax_fr32.scatter(wavs, flux_resid_32, s=4, marker="^", c=colors[2], alpha=0.8)
    ax_fr32.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_fr32.set_ylabel("{\\rm CPU \$-\$ GPU32 flux}")
    ax_fr32.set_xlabel("{\\rm Wavelength [\\AA]}")

    ax_fr32_h = ax_fr32.inset_axes([1.0015, 0, 0.06, 1], sharey=ax_fr32)
    ax_fr32_h.hist(flux_resid_32, bins="auto", density=true, histtype="step",
                   orientation="horizontal", color=colors[2])
    _style_hist_inset(ax_fr32_h)

    ax_ftr32.scatter(wavs[interior], ft_resid_32[interior], s=4, marker="^",
                     c=colors[2], alpha=0.8)
    ax_ftr32.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_ftr32.set_ylabel("{\\rm CPU \$-\$ GPU32 \$T_{\\rm form}\$ [K]}")
    ax_ftr32.set_xlabel("{\\rm Wavelength [\\AA]}")

    ax_ftr32_h = ax_ftr32.inset_axes([1.0015, 0, 0.06, 1], sharey=ax_ftr32)
    ax_ftr32_h.hist(ft_resid_32[interior], bins="auto", density=true, histtype="step",
                    orientation="horizontal", color=colors[2])
    _style_hist_inset(ax_ftr32_h)

    # annotation: max residuals
    ax_fr64.text(0.02, 0.95, "{\\rm |max| =\\ }" * format_max_resid(flux_resid_64),
                 transform=ax_fr64.transAxes, va="top", fontsize=10)
    ax_ftr64.text(0.02, 0.95, "{\\rm |max| =\\ }" * format_max_resid(ft_resid_64[interior]) * " {\\rm K}",
                  transform=ax_ftr64.transAxes, va="top", fontsize=10)
    ax_fr32.text(0.02, 0.95, "{\\rm |max| =\\ }" * format_max_resid(flux_resid_32),
                 transform=ax_fr32.transAxes, va="top", fontsize=10)
    ax_ftr32.text(0.02, 0.95, "{\\rm |max| =\\ }" * format_max_resid(ft_resid_32[interior]) * " {\\rm K}",
                  transform=ax_ftr32.transAxes, va="top", fontsize=10)

    # legend on top row
    leg = ax_flux.legend(loc="lower left", mode="expand", ncol=3, fontsize=11,
                         bbox_to_anchor=(0, 1.02, 2.12, 0.2), handletextpad=0.3)
    for lh in leg.legendHandles
        lh._sizes = [20.0]
    end

    fig.suptitle(title_str, y=1.01, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(joinpath(plotdir, filename), bbox_inches="tight", dpi=150)
    plt.close(fig)
    println("Saved: ", joinpath(plotdir, filename))
end

function _style_hist_inset(ax)
    ax.tick_params(axis="both", labelleft=false, labelbottom=false)
    ax.set_xticks([])
    ax.get_yaxis().set_visible(false)
    for spine in ["left", "bottom", "top", "right"]
        ax.spines[spine].set_visible(false)
    end
    ax.grid(false)
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
          filename="gpu_precision_convolve.pdf")

make_plot(res_cpu_d, res_gpu64_d, res_gpu32_d;
          title_str=@sprintf("{\\rm Disk integration (}\$N_\\phi\${\\rm =%d, }\$N_\\lambda\${\\rm =%d, }\$\\Delta\\lambda\${\\rm =%.3f \\AA)}", Nϕ, length(res_cpu_d.wavs), Δλ),
          filename="gpu_precision_diskint.pdf")

# print summary statistics
println()
println("="^60)
println("SUMMARY")
println("="^60)
for (label, rc, r64, r32) in [("Convolve", res_cpu_c, res_gpu64_c, res_gpu32_c),
                                ("Disk int", res_cpu_d, res_gpu64_d, res_gpu32_d)]
    λ0 = mean(rc.wavs)
    edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
    interior = (edge_px + 1):(length(rc.wavs) - edge_px)

    println()
    println("  $label:")
    @printf("    GPU64 flux   max|resid| = %.2e   mean = %.2e\n",
            maximum(abs.(r64.flux .- rc.flux)),
            mean(abs.(r64.flux .- rc.flux)))
    @printf("    GPU32 flux   max|resid| = %.2e   mean = %.2e\n",
            maximum(abs.(Float64.(r32.flux) .- rc.flux)),
            mean(abs.(Float64.(r32.flux) .- rc.flux)))
    @printf("    GPU64 Tform  max|resid| = %.2f K   mean = %.2f K  (interior)\n",
            maximum(abs.(r64.form_temps[interior] .- rc.form_temps[interior])),
            mean(abs.(r64.form_temps[interior] .- rc.form_temps[interior])))
    @printf("    GPU32 Tform  max|resid| = %.2f K   mean = %.2f K  (interior)\n",
            maximum(abs.(Float64.(r32.form_temps[interior]) .- rc.form_temps[interior])),
            mean(abs.(Float64.(r32.form_temps[interior]) .- rc.form_temps[interior])))
end

println()
println("DONE")
