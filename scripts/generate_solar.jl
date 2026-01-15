using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
# mpl.style.use("tableau-colorblind10")

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
form_temp_result_conv = FT.calc_formation_temp(star_props, linelist; Δλ=0.01, convolve=true, u1=0.43, u2=0.31)

# parse out results
wavs = form_temp_result_conv.wavs
flux_conv = form_temp_result_conv.flux
temp_conv = form_temp_result_conv.form_temps

# plot flux and form temp
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, figsize=(7.2, 7.2))
ax1.plot(wavs, flux_conv)
ax2.plot(wavs, temp_conv)
ax2.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"{\rm Normalized\ Flux}")
ax2.set_ylabel(L"T_{1/2}\ {\rm [K]}")
fig.tight_layout()
plt.show()