using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using FFTW
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib
using ProgressMeter
plt.ioff()

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# python interpolation for matplotlib stuff
interp1d = pyimport("scipy.interpolate").interp1d

ncolors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#A6761D", "#66A61E"]

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

# set steps
steps = range(0.0001, 0.01, step=0.0001)

# allocate memory
αs_error = zeros(length(steps))
rot_error = zeros(length(steps))
rt_error = zeros(length(steps))
rotmacro_error  = zeros(length(steps))
flux_error = zeros(length(steps))

# loop 
@showprogress for i in eachindex(steps)
    # get wavelength grid
    buffer = 0.5
    λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=steps[i])
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

    # get the formation temperature for a stationary star
    cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
    cfunc_flux_cum = Array(FT.get_cum_cfunc(cfunc_flux_stationary))
    flux_stationary = Array(FT.get_flux(cfunc_flux_stationary))

    cfunc_flux_cont_stationary = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)
    cfunc_flux_cont_cum = Array(FT.get_cum_cfunc(cfunc_flux_cont_stationary))
    flux_cont_stationary = Array(FT.get_flux(cfunc_flux_cont_stationary))

    flux_norm = flux_stationary ./ flux_cont_stationary

    # compare microturbulence
    αs_cpu_new = FT.convolve_wavelength_axis(λs_korg, αs, Array(μ_v_rot), Array(σ_v_mic))
    αs_gpu_new = FT.convolve_wavelength_axis_gpu(cmem, CuArray(λs_korg), CuArray(αs), μ_v_rot, σ_v_mic)

    αs_error[i] = maximum(abs.((Array(αs_gpu_new) .- αs_cpu_new) ./ αs_cpu_new))

    # set some params
    vsini = 4200.0
    u1 = 0.4
    u2 = 0.26
    ζ_rt = 1200.0

    # compare rotation
    tbc = Array(cfunc_flux_stationary.cfunc_dt)
    cfunc_flux_gray_rot_cpu = FT.convolve_gray_rotation(λs_korg, tbc, vsini, u1)
    cfunc_flux_gray_rot_gpu = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, tbc, vsini, u1))

    rot_error[i] = maximum(abs.((cfunc_flux_gray_rot_cpu .- cfunc_flux_gray_rot_gpu) ./ cfunc_flux_gray_rot_cpu))

    # compare RT
    cfunc_flux_gray_rt_cpu = FT.convolve_gray_rt_macro(λs_korg, tbc, ζ_rt)
    cfunc_flux_gray_rt_gpu = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt))

    rt_error[i] = maximum(abs.((cfunc_flux_gray_rt_cpu .- cfunc_flux_gray_rt_gpu) ./ cfunc_flux_gray_rt_cpu))

    # compare hirano
    cfunc_flux_hirano_cpu = FT.convolve_hirano_rotmacro(λs_korg, tbc, vsini, ζ_rt, u1, u2)
    cfunc_flux_hirano_gpu = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini, ζ_rt, u1, u2))

    rotmacro_error[i] = maximum(abs.((cfunc_flux_hirano_cpu .- cfunc_flux_hirano_gpu) ./ cfunc_flux_hirano_cpu))

    # do some flux 
    flux_gray_cpu = sum(cfunc_flux_hirano_cpu, dims=1)'
    flux_gray_gpu = sum(cfunc_flux_hirano_gpu, dims=1)'
    flux_error[i] = maximum(abs.((flux_gray_gpu .- flux_gray_cpu) ./ flux_gray_cpu))
end

plt.scatter(steps, αs_error, s=2, label="alpha")
plt.scatter(steps, rot_error, s=2, label="rot")
plt.scatter(steps, rt_error, s=2, label="rt")
plt.scatter(steps, rotmacro_error, s=2, label="hirano")
plt.scatter(steps, flux_error, s=2, label="flux")
plt.legend()
plt.show()