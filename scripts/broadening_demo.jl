#=
Reproduce a Gray-textbook-style figure showing how macroturbulent broadening
alters the emergent line profile of a solar-type star.

The 6252-6254 A region (canonical Fe I lines) is plotted at thermal-only
broadening and at several increasing values of the macroturbulent velocity ζ.
=#

using Revise
using FormationTemps; FT = FormationTemps
using Korg

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyconvert
using LaTeXStrings
mpl = plt.matplotlib
mpl.use("Agg")
mpl.style.use(joinpath(FT.moddir, "fig.mplstyle"))

# output directory
plotdir = joinpath(pwd(), "figures")
!isdir(plotdir) && mkdir(plotdir)

# ── style toggle
dark_mode = false
fg = dark_mode ? "white" : "black"   # spines, ticks, axis labels
bg = dark_mode ? "#111111" : "white" # canvas background
cmap_name = dark_mode ? "plasma" : "viridis"  # curves coloured by ζ

# ── linelist: keep all species within window, line cores at ~6252.5 and ~6254.3
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
λ_lo, λ_hi = 6251.0, 6255.0
window = [l for l in linelist if λ_lo <= l.wl * 1e8 <= λ_hi]
@info "lines in window" n_total=length(linelist) n_window=length(window)

# ── macroturbulence sweep (m/s); micro = rot = 0 throughout
v_macros = [0.0, 6000.0, 12000.0, 18000.0]

# ── compute each spectrum
results = map(v_macros) do ζ
    star = FT.StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0,
                           vsini=0.0, v_macro=ζ, v_micro=0.0)
    FT.calc_formation_temp(star, window;
                           Δλ=0.005, minλ=λ_lo, maxλ=λ_hi,
                           convolve=false, Nϕ=128, showprogress=false)
end

# ── plot
fig, ax = plt.subplots(figsize=(7.5, 4.5))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

# colour each curve by its ζ value via a shared normalisation
cmap = plt.get_cmap(cmap_name)
norm = mpl.colors.Normalize(vmin=minimum(v_macros), vmax=maximum(v_macros))
linewidths = range(1.4, 2.0, length=length(results))

for (i, r) in enumerate(results)
    ax.plot(r.wavs, r.flux,
            color=cmap(norm(v_macros[i])),
            linestyle="-",
            linewidth=linewidths[i])
end

ax.set_xlim(λ_lo + 0.5, λ_hi - 0.3)
ax.set_ylim(0.0, 1.1)
ax.set_xlabel(L"{\rm Wavelength\ [\AA]}", color=fg)
ax.set_ylabel(L"F/F_{\rm c}", color=fg)
ax.tick_params(colors=fg, which="both")
for spine in ax.spines.values()
    spine.set_color(fg)
end

# colorbar keyed to ζ; ticks at the actual sampled values, labelled in km/s
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(L"{\rm Macroturbulence}\ {\rm [km\ s}^{-1}{\rm ]}", color=fg)
cbar.set_ticks(v_macros)
cbar.set_ticklabels([latexstring(round(v / 1000.0; digits=1)) for v in v_macros])
cbar.ax.tick_params(colors=fg, which="both")
cbar.outline.set_edgecolor(fg)

fig.tight_layout()
suffix = dark_mode ? "_dark" : ""
outfile = joinpath(plotdir, "broadening_demo$(suffix).pdf")
fig.savefig(outfile, bbox_inches="tight", facecolor=fig.get_facecolor())
@info "wrote figure" path=outfile
