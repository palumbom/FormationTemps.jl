using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, NPZ, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                            \\usepackage{mathrsfs}")

# read in khaled's
kfile = joinpath(FT.datdir, "Sun_SME.npz")
df = npzread(kfile)

# parse out 
flux_k = df["flux"]
temp_k = df["T1o2"]
wave_k = df["wave"]
cfunc_k = df["cont_abs"]

# read in mine 
mfile = joinpath(FT.datdir, "solar_temps.jld2")
df = load(mfile)

# parse out 
λs_korg = df["λs_korg"]
zs = df["zs"]
Ts = df["Ts"]
τ_500 = df["τ_500"]
μs = df["μs"]
intensities = df["intensities"]
cfuncs_int = df["cfuncs_int"]
continuum_int = df["continuum_int"]
cfuncs_int_cont = df["cfuncs_int_cont"]
flux = df["flux"]
cfunc_flux = df["cfunc_flux"]
continuum_flux = df["continuum_flux"]
cfunc_flux_cont = df["cfunc_flux_cont"]
form_temps_intensity = df["form_temps_intensity"]
form_temps_flux = df["form_temps_flux"]

# compare 
plt.plot(wave_k, flux_k)
plt.plot(λs_korg, flux ./ continuum_flux)
plt.ylim(0.0, 1.1)
plt.show()