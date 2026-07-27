using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using ProgressMeter

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pyconvert
using LaTeXStrings
mpl = plt.matplotlib

# matplotlib backend
mpl.use("QtAgg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")
# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]

# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# set parameters
Teff = 5777.0
logg = 4.44
A_X = Korg.asplund_2020_solar_abundances
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0

# consolidate 
star_props = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# high-level formation temperature calculation
form_temp_result_conv = FT.calc_formation_temp(star_props, linelist; Δλ=0.01, method=:hirano, u1=0.43, u2=0.31)
form_temp_result_int = FT.calc_formation_temp(star_props, linelist; Δλ=0.01, method=:disk, Nϕ=128)

# parse out results
wavs = form_temp_result_conv.wavs
flux_conv = form_temp_result_conv.flux
temp_conv = form_temp_result_conv.form_temps
flux_int = form_temp_result_int.flux
temp_int = form_temp_result_int.form_temps

# plot flux and form temp with residual panels + zoom column
min_depth = 0.02
is_min = falses(length(flux_conv))
for i in 2:length(flux_conv)-1
    depth = 1.0 - flux_conv[i]
    is_min[i] = depth > min_depth && flux_conv[i] < flux_conv[i-1] && flux_conv[i] < flux_conv[i+1]
end
min_idx_all = findall(is_min)
min_idx = min_idx_all
if length(min_idx) >= 2
    min_wavs = wavs[min_idx]
    min_flux = flux_conv[min_idx]
    depth_sort = sortperm(min_flux)
    min_idx = min_idx[depth_sort]
    min_wavs = min_wavs[depth_sort]
    n_select = max(1, Int(ceil(length(min_idx) * 0.10)))
    min_idx = min_idx[1:n_select]
    min_wavs = min_wavs[1:n_select]
    min_sort = sortperm(min_wavs)
    min_idx = min_idx[min_sort]
    min_wavs = min_wavs[min_sort]
    if length(min_wavs) == 1
        zoom_center = min_wavs[1]
    else
        min_spacing = similar(min_wavs)
        min_spacing[1] = abs(min_wavs[2] - min_wavs[1])
        min_spacing[end] = abs(min_wavs[end] - min_wavs[end-1])
        for i in 2:length(min_wavs)-1
            min_spacing[i] = min(min_wavs[i] - min_wavs[i-1], min_wavs[i+1] - min_wavs[i])
        end
        zoom_center = min_wavs[argmax(min_spacing)]
    end
else
    sorted_idx = sortperm(flux_conv)
    zoom_center = wavs[sorted_idx[2]]
end
cont_level = 0.95
center_idx = argmin(abs.(wavs .- zoom_center))
left_idx = findlast(i -> flux_conv[i] >= cont_level, 1:center_idx)
right_idx = findfirst(i -> flux_conv[i] >= cont_level, center_idx:length(wavs))
left_idx = isnothing(left_idx) ? 1 : left_idx
right_idx = isnothing(right_idx) ? length(wavs) : center_idx + right_idx - 1
left_span = zoom_center - wavs[left_idx]
right_span = wavs[right_idx] - zoom_center
half_span = max(left_span, right_span)
zoom_min = zoom_center - half_span
zoom_max = zoom_center + half_span

fig, axes = plt.subplots(nrows=4, ncols=2, sharex="col", figsize=(10.5, 9.0),
                         gridspec_kw=Dict("height_ratios" => [4, 1, 4, 1],
                                          "width_ratios" => [3, 2]))
ax1 = axes[0, 0]
ax1r = axes[1, 0]
ax2 = axes[2, 0]
ax2r = axes[3, 0]
ax1z = axes[0, 1]
ax1rz = axes[1, 1]
ax2z = axes[2, 1]
ax2rz = axes[3, 1]

ax1.plot(wavs, flux_int, label=L"{\rm Integration}", c="k")
ax1.plot(wavs, flux_conv, label=L"{\rm Convolution}", c="tab:blue", alpha=0.8)
ax2.plot(wavs, temp_int, label=L"{\rm Integration}", c="k")
ax2.plot(wavs, temp_conv, label=L"{\rm Convolution}", c="tab:blue", alpha=0.8)

ax1z.plot(wavs, flux_int, c="k")
ax1z.plot(wavs, flux_conv, c="tab:blue", alpha=0.8)
ax2z.plot(wavs, temp_int, c="k")
ax2z.plot(wavs, temp_conv, c="tab:blue", alpha=0.8)

ax1z.yaxis.tick_right()
ax1rz.yaxis.tick_right()
ax2z.yaxis.tick_right()
ax2rz.yaxis.tick_right()
ax1z.yaxis.set_label_position("right")
ax1rz.yaxis.set_label_position("right")
ax2z.yaxis.set_label_position("right")
ax2rz.yaxis.set_label_position("right")

ax1r.plot(wavs, flux_int .- flux_conv, color="k")
ax2r.plot(wavs, temp_int .- temp_conv, color="k")
ax1rz.plot(wavs, flux_int .- flux_conv, color="k")
ax2rz.plot(wavs, temp_int .- temp_conv, color="k")

ax1r.axhline(0.0, color="0.5", linewidth=0.8)
ax2r.axhline(0.0, color="0.5", linewidth=0.8)
ax1rz.axhline(0.0, color="0.5", linewidth=0.8)
ax2rz.axhline(0.0, color="0.5", linewidth=0.8)

ax1.axvspan(zoom_min, zoom_max, facecolor="none", edgecolor="0.3", linewidth=1.0)
ax1r.axvspan(zoom_min, zoom_max, facecolor="none", edgecolor="0.3", linewidth=1.0)
ax2.axvspan(zoom_min, zoom_max, facecolor="none", edgecolor="0.3", linewidth=1.0)
ax2r.axvspan(zoom_min, zoom_max, facecolor="none", edgecolor="0.3", linewidth=1.0)

ax1z.set_xlim(zoom_min, zoom_max)
ax1rz.set_xlim(zoom_min, zoom_max)
ax2z.set_xlim(zoom_min, zoom_max)
ax2rz.set_xlim(zoom_min, zoom_max)

ax2r.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax2rz.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"{\rm Normalized\ Flux}")
ax1r.set_ylabel(L"{\rm Flux\ Error}")
ax2.set_ylabel(L"T_{1/2}\ {\rm [K]}")
ax2r.set_ylabel(L"T_{1/2}\ {\rm Error\ [K]}")
ax1.legend(bbox_to_anchor=(0.33, 1.02, 1.05, 0.2), loc="lower center",
           mode="expand", borderaxespad=0, ncol=2)
fig.tight_layout()
fig.savefig(joinpath(FT.moddir, "docs", "src", "static", "convolution_vs_integration.png"),
            bbox_inches="tight")
plt.show()
