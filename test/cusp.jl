using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using FFTW
using CSV, DataFrames, Statistics
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport
mpl = plt.matplotlib
plt.ioff()

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

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

# make the wavelength grid
buffer = 2.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.001)
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
Npad = 5000
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

μ_v_mac = CUDA.zeros(Float64, length(zs)-1)
σ_v_mac = CUDA.zeros(Float64, length(zs)-1)

cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
tbc = cfunc_flux_stationary.cfunc_dt
flux_stationary = Array(FT.get_flux(cfunc_flux_stationary))

# set rotational and macroturbulence
vsini = 2100.0
ζ_rt = 3400.0

# set limb darkening
u1 = 0.4
u2 = 0.0

# set some mus
μs = 1.0

# compare RT aniso
cfunc_flux_gray_rt_cpu = FT.convolve_rt_macro(λs_korg, Array(tbc), ζ_rt, μs)
# cfunc_flux_gray_rt_cpu = FT.convolve_iso_rt_macro(λs_korg, Array(tbc), ζ_rt)
# cfunc_flux_gray_rt_cpu = FT.convolve_gray_rotation(λs_korg, Array(tbc), vsini, u1)
cfunc_flux_gray_rt_gpu = Array(FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt, μs))
# cfunc_flux_gray_rt_gpu = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt))
# cfunc_flux_gray_rt_gpu = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, tbc, vsini, u1))

# get flux
flux1 = 2π .* dropdims(sum(cfunc_flux_gray_rt_cpu, dims=1), dims=1)
flux2 = 2π .* dropdims(Array(sum(cfunc_flux_gray_rt_gpu, dims=1)), dims=1)

# get errors
rt_aniso_errosr = (cfunc_flux_gray_rt_cpu .- cfunc_flux_gray_rt_gpu) ./ cfunc_flux_gray_rt_cpu
flux_err = 100 .* ((flux1 .- flux2) ./ flux1)

plt.plot(λs_korg, flux_err)
plt.show()

# # plot
# plt.plot(λs_korg, flux_stationary)
# plt.plot(λs_korg, flux1)
# plt.plot(λs_korg, flux2)
# plt.show()
