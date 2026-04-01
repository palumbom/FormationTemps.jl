using Revise
using FormationTemps; FT = FormationTemps
using Korg, LsqFit
using HDF5, Printf, JLD2
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport
mpl = plt.matplotlib
using ProgressMeter
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
idx1 = findfirst(x -> x * FT.CM_TO_ANGSTROM .>= 6301, wls)
idx2 = findfirst(x -> x * FT.CM_TO_ANGSTROM .>= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

# re-get values
wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# make the wavelength grid
buffer = 0.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.002)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
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

cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)

# get the formation temperature for a stationary star
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_stationary = Array(FT.get_flux(cfunc_flux_struct)')
cum_cfunc_flux_stationary = Array(FT.get_cum_cfunc(cfunc_flux_struct))

cfunc_flux_cont_struct = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_cont_stationary = Array(FT.get_flux(cfunc_flux_cont_struct)')

form_temp_stationary = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cum_cfunc_flux_stationary, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_stationary[i] = itp(0.5)
end

# set broadening
vsini = 2100.0
vmac = 3400.0

# get disk stuff 
ρstar = 1.0
istar = 90.0
v0 = vsini
Nϕ = 32
μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

# flatten, move to cpu
idx = findall(x -> x .> zero(eltype(μs)), Array(μs))
μs_cpu = Array(μs)[idx]
dA_cpu = Array(dA)[idx]
z_rot_cpu = Array(z_rot)[idx]

flux_test = CUDA.zeros(Float64, length(λs_korg))
flux_integration = CUDA.zeros(Float64, length(λs_korg))
cfunc_test = CUDA.zeros(Float64, length(atm_gpu.zs) - 1, length(λs_korg))
cfunc_integration = CUDA.zeros(Float64, length(atm_gpu.zs) - 1, length(λs_korg))

@showprogress for i in eachindex(μs_cpu)
    # set the rotational velocity
    μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

    # get intensity stuff
    cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)
    tbc = cfunc_intensity_struct.cfunc_dt

    flux_test .+= FT.get_intensity(cfunc_intensity_struct) .* dA_cpu[i]
    cfunc_test .+= tbc .* dA_cpu[i]

    cfunc_int_i_mac = FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, tbc, vmac)
    cfunc_integration .+= cfunc_int_i_mac .* dA_cpu[i]
    flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
end

# convolve flux_test
flux_test_mac = FT.convolve_iso_rt_macro(λs_korg, Array(flux_test), vmac)
cfunc_test_mac =  FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, cfunc_test, vmac)
flux_new_test_mac = Array(sum(cfunc_test_mac, dims=1))'

plt.plot(Array(flux_test_mac))
plt.plot(Array(flux_new_test_mac))
plt.plot(Array(flux_integration))
plt.show()
