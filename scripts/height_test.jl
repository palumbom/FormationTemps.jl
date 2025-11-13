using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using ProgressMeter
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

AF = AbstractFloat
AA = AbstractArray

# make the wavelength grid
λs_korg = range(3000.0, 6000.0, step=0.1)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
marcs_atm = FT.get_marcs_atm(5777.0, 4.44, A_X, n_layers=56)
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

# # synthesis to get the alphas
# αs = zeros(length(atm_gpu.zs), length(λs_korg))
# αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
# FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), [], atm_gpu, A_X)

sol = synthesize(marcs_atm, [], A_X, λs_korg; vmic=1.2, tau_scheme="bezier", 
                 hydrogen_lines=false, use_MHD_for_hydrogen_lines=false)
αs = deepcopy(sol.alpha)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 2400
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

μ_v_mac = CUDA.zeros(Float64, length(zs)-1)
σ_v_mac = CUDA.zeros(Float64, length(zs)-1)

cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

cfunc_intensity = FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, 1.0, μ_v_rot, σ_v_mic)
intensity = dropdims(sum(cfunc_flux_stationary, dims=1), dims=1)

# get the formation temperature for a stationary star
cfunc_flux_stationary = 2π .* FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_stationary = dropdims(sum(cfunc_flux_stationary, dims=1), dims=1)

cum_cfunc_flux_stationary = cumsum(cfunc_flux_stationary, dims=1)
cum_cfunc_flux_stationary ./= maximum(cum_cfunc_flux_stationary, dims=1)

cum_cfunc_intensity = cumsum(cfunc_intensity, dims=1)
cum_cfunc_intensity ./= maximum(cum_cfunc_intensity, dims=1)


form_height = zeros(length(λs_korg))
form_temp = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cum_cfunc_flux_stationary, :, i)
    # xs = view(cum_cfunc_intensity, :, i)
    itp = FT.linear_interp(xs, elav(zs))
    form_height[i] = itp(0.5)

    xs = view(cum_cfunc_flux_stationary, :, i)
    # xs = view(cum_cfunc_intensity, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp[i] = itp(0.5)
end

# plt.plot(λs_korg, flux_stationary)

plt.plot(λs_korg, form_height)
plt.show()