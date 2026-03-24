using Revise, Anemoi
using CSV, DataFrames, Statistics
using Korg, GRASS, CUDA, Adapt
using BenchmarkTools
using LinearAlgebra
using NPZ, HDF5
using EchelleCCFs: λ_air_to_vac, λ_vac_to_air
import PythonPlot; plt = PythonPlot

# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

# data path 
outpath = joinpath(homedir(), "ceph", "formation_temps")
figpath = joinpath(outpath, "figures")

# get khaleds
khaled = CSV.read(joinpath(outpath, "khaleds.csv"), DataFrame)
khaled = coalesce.(khaled, NaN)
λs_khaled = λ_air_to_vac.(khaled.wave)
flux_khaled = khaled.flux
temps_khaled = Array(khaled.T1o2)

# get mine
fname = joinpath(outpath, "line_formation_by_mu.h5")
λs_mine, flux_mine, temps_mine = h5open(fname, "r") do file
    λs_mine = read(file["vac_wavs"])
    flux_mine = read(file["flux"])[:,1]
    temps_mine = read(file["form_temps"])[:,1]
    return λs_mine, flux_mine, temps_mine
end

# get the continuum
fname = joinpath(outpath, "continuum_by_mu.h5")
λs_cont, flux_cont = h5open(fname, "r") do file
    λs_cont = read(file["vac_wavs"])
    flux_cont = read(file["flux"])[:,1]
    return λs_cont, flux_cont
end

# interpolate 
itp = Korg.CubicSpline(λs_cont, flux_cont)
flux_cont_new = itp.(λs_mine)

# read my linelist
linelist = Korg.read_linelist(joinpath(Anemoi.datdir, "linelist.h5"))
my_wls = [l.wl * 1e8 for l in linelist]

# read Khaled's linelist 
valdlist = Korg.read_linelist(joinpath(Anemoi.datdir, "Sun_VALD.lin"), format="vald")
vald_wls = [l.wl * 1e8 for l in valdlist]

# get the intersections
common = intersect(my_wls, vald_wls)
my_indices = findall(x -> x in common, my_wls)
vald_indices = findall(x -> x in common, vald_wls)

# get one linelist 
linelist_common = linelist[my_indices]
wls = [l.wl * 1e8 for l in linelist_common]

# calc the formation temps
# allocate for lines in list 
avg_temp_50 = zeros(length(wls))
avg_temp_80 = zeros(length(wls))

# loop over lines in the list
for i in eachindex(wls)
    # get the index of the lines location
    idx_wav = findfirst(x -> x .>= wls[i], λs_khaled)

    # get the flux slice
    flux_slice = flux_khaled
    temps_slice = temps_khaled

    # refine the location of the minimum
    min_idx = argmin(flux_slice[idx_wav - 5:idx_wav+5]) + (idx_wav - 5) - 1

    # get a view of the line 
    idxl = clamp(min_idx - 50, firstindex(flux_slice), lastindex(flux_slice))
    idxr = clamp(min_idx + 50, firstindex(flux_slice), lastindex(flux_slice))
    wavs_view = view(λs_khaled, idxl:idxr)
    flux_view = view(flux_slice, idxl:idxr)

    # # do a rough continuum normalization
    flux_norm = flux_view ./ maximum(flux_view)

    # get the depth
    bot = minimum(flux_norm)
    depth = 1.0 - bot

    # get the wing indices 
    idxl_50, idxr_50 = GRASS.find_wing_index(0.5 * depth + bot, flux_norm, min=argmin(flux_norm))
    idxl_80, idxr_80 = GRASS.find_wing_index(0.8 * depth + bot, flux_norm, min=argmin(flux_norm))

    # make the relative indices absolute
    idxl_50 += idxl
    idxl_80 += idxl
    idxr_50 += idxl - 1
    idxr_80 += idxl - 1

    # now get the avg formation coords 
    avg_temp_50[i] = mean(view(temps_slice, idxl_50:idxr_50))
    avg_temp_80[i] = mean(view(temps_slice, idxl_80:idxr_80))
end

# read in my form temps 
fname = joinpath(outpath, "avg_line_formation_by_mu.h5")
vac_wavs, avg_temp_50_mine, avg_temp_80_mine, avg_temp_50_integrated_mine, avg_temp_80_integrated_mine = h5open(fname, "r") do file
    vac_wavs = read(file["vac_wavs"])[:,1][my_indices]
    avg_temp_50_mine = read(file["avg_temp_50"])[:,1][my_indices]
    avg_temp_80_mine = read(file["avg_temp_80"])[:,1][my_indices]
    avg_temp_50_integrated_mine = read(file["avg_temp_integrated_50"])[my_indices]
    avg_temp_80_integrated_mine = read(file["avg_temp_integrated_80"])[my_indices]
    return vac_wavs, avg_temp_50_mine, avg_temp_80_mine, avg_temp_50_integrated_mine, avg_temp_80_integrated_mine
end

# plot em 
mosaic = """
         AAA
         AAA
         BBB
         """

fig = plt.figure(layout="constrained")
ax_dict = fig.subplot_mosaic(mosaic)
ax_dict["A"].scatter(wls, avg_temp_50, c="tab:blue", s=5, label="khaled")
ax_dict["A"].scatter(vac_wavs, avg_temp_50_integrated_mine, c="tab:orange", s=5, label="korg")
ax_dict["B"].scatter(vac_wavs, avg_temp_50_integrated_mine .- avg_temp_50, c="k", s=5)
ax_dict["B"].set_xlabel("Wavelength (Å)")
ax_dict["B"].set_ylabel("Residual (K)")
ax_dict["A"].set_ylabel("Formation Temp (K)")
ax_dict["A"].legend()
fig.savefig(joinpath(figpath, "compare_khaled_v_korg.pdf"), bbox_inches="tight")
plt.clf(); plt.close()
