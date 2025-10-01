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
flux_m = df["flux"]
temp_k = df["form_temps_flux"]
wave_k = df["λs_korg"]
cfunc_k = df["cfunc_flux"]