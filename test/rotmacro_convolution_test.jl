using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf, JLD2
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using ProgressMeter
using PyPlot, PyCall; mpl = plt.matplotlib
plt.ioff()

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# set colormaps
img_cmap = "viridis"
μ_cmap = "autumn"

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
buffer = 0.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.0023334)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

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
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 500
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

μ_v_mac = CUDA.zeros(Float64, length(zs)-1)
σ_v_mac = CUDA.zeros(Float64, length(zs)-1)

cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

# get the formation temperature for a stationary star
cfunc_flux_stationary = 2π .* FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_stationary = dropdims(sum(cfunc_flux_stationary, dims=1), dims=1)

cfunc_flux_cont_stationary = 2π .* FT.calc_flux_cfunc(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_cont_stationary = dropdims(sum(cfunc_flux_cont_stationary, dims=1), dims=1)

# set some params
vsini = 4200.0
u1 = 0.4
u2 = 0.26
ζ_rt = 1200.0

# do rotmacro gpu
cfunc_flux_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_stationary, vsini, ζ_rt, u1, u2))
# cfunc_flux_convolution = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, cfunc_flux_stationary, vsini, u1))
# cfunc_flux_convolution = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, λs_korg, cfunc_flux_stationary, ζ_rt))
flux_convolution = dropdims(sum(cfunc_flux_convolution, dims=1), dims=1)
cfunc_flux_cont_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_cont_stationary, vsini, ζ_rt, u1, u2))
# cfunc_flux_cont_convolution = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, cfunc_flux_cont_stationary, vsini, u1))
# cfunc_flux_cont_convolution = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, λs_korg, cfunc_flux_cont_stationary, ζ_rt))
flux_cont_convolution = dropdims(sum(cfunc_flux_cont_convolution, dims=1), dims=1)

# normalize
flux_norm_convolution_gpu = flux_convolution ./ flux_cont_convolution
plt.plot(flux_norm_convolution_gpu .- flux_stationary./flux_cont_stationary, label="GPU")

# do rotmacro cpu
cfunc_flux_convolution = Array(FT.convolve_hirano_rotmacro(λs_korg, cfunc_flux_stationary, vsini, ζ_rt, u1, u2))
# cfunc_flux_convolution = Array(FT.convolve_gray_rotation(λs_korg, cfunc_flux_stationary, vsini, u1))
# cfunc_flux_convolution = Array(FT.convolve_gray_rt_macro(λs_korg, cfunc_flux_stationary, ζ_rt))
flux_convolution = dropdims(sum(cfunc_flux_convolution, dims=1), dims=1)
cfunc_flux_cont_convolution = Array(FT.convolve_hirano_rotmacro(λs_korg, cfunc_flux_cont_stationary, vsini, ζ_rt, u1, u2))
# cfunc_flux_cont_convolution = Array(FT.convolve_gray_rotation(λs_korg, cfunc_flux_cont_stationary, vsini, u1))
# cfunc_flux_cont_convolution = Array(FT.convolve_gray_rt_macro(λs_korg, cfunc_flux_cont_stationary, ζ_rt))
flux_cont_convolution = dropdims(sum(cfunc_flux_cont_convolution, dims=1), dims=1)

# normalize
flux_norm_convolution_cpu = flux_convolution ./ flux_cont_convolution
plt.plot(flux_norm_convolution_cpu .- flux_stationary./flux_cont_stationary, label="CPU")

@show extrema(flux_norm_convolution_gpu .- flux_norm_convolution_cpu)
plt.legend()
plt.show()

#= # get disk stuff 
ρstar = 1.0
istar = 90.0
v0 = vsini
Nϕ = 32
μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

# flatten, move to cpu
μs_cpu = Array(μs)
dA_cpu = Array(dA)
z_rot_cpu = Array(z_rot)

# allocate for output
flux_integration = zeros(length(λs_korg))
flux_cont_integration = zeros(length(λs_korg))

cfunc_int_i_gpu = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))
cfunc_int_cont_i_gpu = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))
λs_gpu = CuArray(λs_korg)

# do the disk integration
idx = findall(μs_cpu .> 0.0)
@showprogress for i in idx
    # set the rotational velocity
    μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

    # get the intensity contribution function
    cfunc_int_i = FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)
    cfunc_int_cont_i = FT.calc_intensity_cfunc(αs_cont, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)

    # convolve the cfunc with RT macroturbulence
    copyto!(cfunc_int_i_gpu, cfunc_int_i)
    copyto!(cfunc_int_cont_i_gpu, cfunc_int_cont_i)
    cfunc_int_i_mac = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, λs_gpu, cfunc_int_i_gpu, ζ_rt))
    cfunc_int_cont_i_mac = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, λs_gpu, cfunc_int_cont_i_gpu, ζ_rt))

    # cfunc_int_i_mac = FT.convolve_gray_rt_macro(λs_korg, cfunc_int_i, ζ_rt)
    # cfunc_int_cont_i_mac = FT.convolve_gray_rt_macro(λs_korg, cfunc_int_cont_i, ζ_rt)

    # add to the flux integral
    flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
    flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[i]
end

# convert units on disk integration
flux_integration .*= 1e-8
flux_cont_integration .*= 1e-8

# normalize
flux_norm_integration = flux_integration ./ flux_cont_integration

# plt.plot(λs_korg, circshift(flux_norm_convolution_gpu, 1))
# plt.plot(λs_korg, circshift(flux_norm_convolution_cpu, 1))
# plt.plot(λs_korg, flux_norm_integration)
plt.scatter(λs_korg, flux_norm_integration .- circshift(flux_norm_convolution_cpu, -1), s=5)
plt.scatter(λs_korg, flux_norm_integration .- circshift(flux_norm_convolution_gpu, -1), s=5)
plt.show() =#