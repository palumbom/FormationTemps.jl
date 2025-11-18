using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using ProgressMeter
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib
plt.ioff()

AF = AbstractFloat
AA = AbstractArray

# make the wavelength grid
λs_korg = range(3000.0, 6000.0, step=0.1)
# λs_korg = range(6301.0, 6303.0, step=0.1)

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
Npad = 240
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

# get intensity stuff
cfunc_intensity, cfunc_intensity_cum, intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, 1.0, μ_v_rot, σ_v_mic)

# get flux stuff
cfunc_flux_stationary, cfunc_flux_cum, flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)

# plt.plot(λs_korg, sol.flux)
# plt.plot(λs_korg, flux_stationary)
# plt.show()

# formation heights
form_height = zeros(length(λs_korg))
form_temp = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cfunc_flux_cum, :, i)
    # xs = view(cum_cfunc_intensity, :, i)
    itp = FT.linear_interp(xs, elav(zs))
    form_height[i] = itp(0.5)

    xs = view(cfunc_flux_cum, :, i)
    # xs = view(cum_cfunc_intensity, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp[i] = itp(0.5)
end

# plt.plot(λs_korg, flux_stationary)

plt.plot(λs_korg, form_height)
plt.xlabel("Wavelength")
plt.ylabel("Formation Height")
plt.legend()
plt.show()