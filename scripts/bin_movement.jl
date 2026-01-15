using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, NPZ, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib
using ProgressMeter

# matplotlib config
plt.ioff()
mpl.style.use(FT.moddir * "fig.mplstyle")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                            \\usepackage{mathrsfs}")

# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

# make plotdir
plotdir = joinpath(pwd(), "figures")
!isdir(plotdir) && mkdir(plotdir)

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist][16000:17000]
specs = [string(l.species) for l in linelist]

# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# make the wavelength grid
buffer = 1.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.01)
cont_idx = findfirst(x -> x .>= wls[2] + 0.1, λs_korg)#findfirst(x -> x .>= 6301.3, λs_korg)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
marcs_atm = FT.get_marcs_atm(5777.0, 4.44, A_X, n_layers=168 * 3)
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
αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 240
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# loop over mus 
μs = range(0.1, 1.0, step=0.1)
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0
cfuncs = zeros(length(zs)-1, length(λs_korg), length(μs))
cfuncs_cum = zeros(length(zs)-1, length(λs_korg), length(μs))
intensities = zeros(length(λs_korg), length(μs))
continuum = zeros(length(λs_korg), length(μs))

μ_cmap = "autumn"
cmap = plt.get_cmap(μ_cmap)
# norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=maximum(μs))
norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=1.075)
colors = cmap(norm(μs))

for i in eachindex(μs)
    cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    cfuncs[:,:,i] .= Array(cfunc_intensity_struct.cfunc_dt)
    cfuncs_cum[:,:,i] .= Array(FT.get_cum_cfunc(cfunc_intensity_struct))
    intensities[:,i] .= Array(FT.get_intensity(cfunc_intensity_struct))

    cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    continuum[:,i] .= Array(FT.get_intensity(cfunc_intensity_cont))
end
 
# get flux and flux cfunc
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux_disk_integrated = Array(FT.get_flux(cfunc_flux_struct))
cfunc_flux = Array(cfunc_flux_struct.cfunc_dt)
cfunc_flux_cum = Array(FT.get_cum_cfunc(cfunc_flux_struct))

# now get cumulative contribution functions
cum_cfuncs_norm = cfuncs_cum
cum_cfunc_flux_norm = cfunc_flux_cum

# now compute the formation temperature
form_temps_intensity = zeros(length(λs_korg), length(μs))
form_temps_flux = zeros(length(λs_korg))

for i in eachindex(λs_korg)
    local xs = view(cum_cfunc_flux_norm, :, i)
    local itp = FT.linear_interp(xs, elav(Ts))
    form_temps_flux[i] = itp(0.5)
end

for i in eachindex(λs_korg)
    for j in eachindex(μs)
        local xs = view(cum_cfuncs_norm, :, i, j)
        local itp = FT.linear_interp(xs, elav(Ts))
        form_temps_intensity[i,j] = itp(0.5)
    end
end

# overplot the intensity and flux formation temperure spectra 
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[3,1])
ax1.plot(λs_korg, form_temps_flux, c="k", label=L"{\rm Flux}")
# plt.plot(λs_korg, form_temps_intensity[:,end], c=colors[end,:], label=L"{\rm Disk\ Center\ Intensity}")
ax1.plot(λs_korg, form_temps_intensity[:,end], c="gold", label=L"{\rm Disk\ Center\ Intensity}")
ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
ax2.plot(λs_korg, form_temps_flux .- form_temps_intensity[:,end], c="k", label=L"{\rm Disk\ Center\ Intensity}")
ax2.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
ax2.set_ylabel(L"{\rm Difference\ [K]}")
fig.savefig("figures/bin_form_temp_spectra.pdf")
plt.close()

# split the bins 
x1 = form_temps_flux
x2 = @view form_temps_intensity[:,end]
nbins = 4
bin_edges_flux = range(floor(Int, minimum(x1)), ceil(Int, maximum(x1)); length=nbins+1)
bin_edges_intc = range(floor(Int, minimum(x2)), ceil(Int, maximum(x2)); length=nbins+1)

bins_flux = clamp.(searchsortedlast.(Ref(bin_edges_flux), x1), 1, length(bin_edges_flux)-1)
bins_intc = clamp.(searchsortedlast.(Ref(bin_edges_intc), x2), 1, length(bin_edges_intc)-1)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
ax1.plot(λs_korg, bins_flux, c="k", label=L"{\rm Flux}")
# ax1.plot(λs_korg, bins_intc, c=colors[end,:], label=L"{\rm Disk\ Center\ Intensity}")
ax1.plot(λs_korg, bins_intc, c="gold", label=L"{\rm Disk\ Center\ Intensity}")
ax2.plot(λs_korg, bins_flux .- bins_intc, c="k")
ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
ax2.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"{\rm Bin\ Assignment}")
ax2.set_ylabel(L"{\rm Difference}")
fig.savefig("figures/bin_movement.pdf")
plt.close()