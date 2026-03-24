using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using ProgressMeter
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
import PythonPlot; plt = PythonPlot
mpl = plt.matplotlib
plt.ioff()

AF = AbstractFloat
AA = AbstractArray

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
λs_korg = range(2000.0, 6400.0, step=0.05)
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

# synthesis to get the alphas
αs = zeros(length(atm_gpu.zs), length(λs_korg))
αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
# FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), [], atm_gpu, A_X)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# sol = synthesize(marcs_atm, [], A_X, λs_korg; vmic=1.2, tau_scheme="bezier", 
#                  hydrogen_lines=false, use_MHD_for_hydrogen_lines=false)
# αs = deepcopy(sol.alpha)

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
cfunc_int = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, 1.0, μ_v_rot, σ_v_mic)
cfunc_int_cum = Array(FT.get_cum_cfunc(cfunc_int))
intensity = Array(FT.get_intensity(cfunc_int))

# get flux stuff
cfunc_flux = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
cfunc_flux_cum = Array(FT.get_cum_cfunc(cfunc_flux))
flux = Array(FT.get_flux(cfunc_flux))

# formation heights
form_height = zeros(length(λs_korg))
form_temp = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cfunc_flux_cum, :, i)
    # xs = view(cfunc_int_cum, :, i)
    itp = FT.linear_interp(xs, elav(zs))
    form_height[i] = itp(0.5)

    xs = view(cfunc_flux_cum, :, i)
    # xs = view(cfunc_int_cum, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp[i] = itp(0.5)
end

# plot it
plt.plot(λs_korg, form_height)
plt.xlabel("Wavelength")
plt.ylabel("Formation Height")
plt.legend()
plt.show()
