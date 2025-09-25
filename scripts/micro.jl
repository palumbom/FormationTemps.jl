using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics, NaNMath
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                            \\usepackage{mathrsfs}")

# python interpolation for matplotlib stuff
interp1d = pyimport("scipy.interpolate").interp1d

# set colormaps
img_cmap = "viridis"
μ_cmap = "autumn"
vmic_cmap = "autumn"

# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

# make plotdir
plotdir = joinpath(pwd(), "figures")
!isdir(plotdir) && mkdir(plotdir)

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]

# cut on species
linelist = linelist[specs .== "Fe I"]

# get the Fe I 6301 & 6302 lines (just cuz)
wls = [l.wl for l in linelist] 
idx1 = findfirst(x -> x * 1e8 .>= 6301, wls)
idx2 = findfirst(x -> x * 1e8 .>= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])


# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# make the wavelength grid
λs_korg = range(first(wls) - 5.0, last(wls) + 5.0, step=0.005)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
marcs_atm = get_marcs_atm(5777.0, 4.44, A_X, n_layers=168 * 3)
τ_500 = Korg.get_tau_refs(marcs_atm)
zs = Korg.get_zs(marcs_atm)
Ts = Korg.get_temps(marcs_atm)
ne = Korg.get_electron_number_densities(marcs_atm)
nd = Korg.get_number_densities(marcs_atm)

# make my atmosphere 
atm_gpu = FT.AtmosphereGPU(marcs_atm)
zs = atm_gpu.zs
Ts = atm_gpu.Ts
τ5000 = atm_gpu.τs

# synthesis to get the alphas
αs = zeros(length(atm_gpu.zs), length(λs_korg))
FT.compute_alpha!(αs, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# make array of vmics 
mic_min = 0.0
vmics = range(mic_min, 8000.0 + mic_min, step=800.0)

# loop over vmics
μs = 1.0
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs))
cfuncs = zeros(length(zs)-1, length(λs_korg), length(vmics))
cfuncs_flux = zeros(length(zs)-1, length(λs_korg), length(vmics))
intensities = zeros(length(λs_korg), length(vmics))
fluxes = zeros(length(λs_korg), length(vmics))

for i in eachindex(vmics)
    σ_v .= vmics[i]
    cfuncs[:,:,i] .= FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs, μ_v, σ_v)
    intensities[:,i] .= dropdims(sum(view(cfuncs,:,:,i), dims=1), dims=1)

    cfuncs_flux[:,:,i] = FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v)
    fluxes[:,i] = 2π .* dropdims(sum(view(cfuncs_flux,:,:,i), dims=1), dims=1) 
end

cum_cfuncs_norm = cumsum(cfuncs, dims=1) 
cum_cfuncs_norm ./= maximum(cum_cfuncs_norm, dims=1)
cum_cfuncs_flux_norm = cumsum(cfuncs_flux, dims=1) 
cum_cfuncs_flux_norm ./= maximum(cum_cfuncs_flux_norm, dims=1)

form_temps_int = zeros(length(λs_korg), length(vmics))
form_temps_flux = zeros(length(λs_korg), length(vmics))

for i in eachindex(λs_korg)
    for j in eachindex(vmics)
        local xs1 = view(cum_cfuncs_norm, :, i, j)
        local xs2 = view(cum_cfuncs_flux_norm, :, i, j)
        local itp1 = FT.linear_interp(xs1, elav(Ts))
        local itp2 = FT.linear_interp(xs2, elav(Ts))
        form_temps_int[i, j] = itp1(0.5)
        form_temps_flux[i, j] = itp2(0.5)
    end
end

# get colormaps
cmap = plt.get_cmap(vmic_cmap)
# norm = mpl.colors.Normalize(vmin=minimum(vmics), vmax=maximum(vmics) + 50.0)
norm = mpl.colors.Normalize(vmin=0.0, vmax=8000.0)
colors = cmap(norm(vmics))

# do some plotting 
fig, ax1 = plt.subplots()
for i in eachindex(vmics)
    ax1.plot(λs_korg, form_temps_flux[:,i], c=colors[i,:])
end

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"v_{\rm mic}\ {\rm[km\ s}^{-1}{\rm ]}")
cbar.set_ticklabels(latexstring.(cbar.get_ticks() ./ 1000.0))

ax1.set_xlim(first(wls) - 0.75, last(wls) + 0.75)
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
fig.savefig(joinpath(plotdir, "vmic.pdf"), bbox_inches="tight")
plt.show()