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
buffer = 0.1
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.00005)
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
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

μ_v_mac = CUDA.zeros(Float64, length(zs)-1)
σ_v_mac = CUDA.zeros(Float64, length(zs)-1)

cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

# set rotational and macroturbulence 
vsini = 2100.0
ζ_rt = 1400.0

# set limb darkening
u1 = 0.4
u2 = 0.0

# set some mus 
μs = range(0.1, 1.0, step=0.1)

# get v grid
xs = λs_korg

N = length(xs)
λ0 = mean(xs)
vs = FT.c_ms .* (xs .- λ0) ./ λ0
Δv = (last(vs) - first(vs)) / (N - 1)
dv = diff(vs)

# get iso rt 
iso_rt_macro_kernel = FT.gray_iso_rt_macro_kernel(vs, ζ_rt)

# get stellar grid
ρstar = 1.0
istar = 90.0
v0 = 0.0
Nϕ = 256
μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

# flatten, move to cpu
idx = findall(x -> x .> 0.0, Array(μs))
μs_cpu = Array(μs)[idx]
dA_cpu = Array(dA)[idx]
z_rot_cpu = Array(z_rot)[idx]

# allocate for total
int_kernel = zeros(length(vs))

# loop over disk
for i in eachindex(μs_cpu)
    # aniso kernel for mu
    aniso_rt_macro_kernel = FT.rt_macro_kernel(vs, ζ_rt, μs_cpu[i])
    # plt.plot(vs, aniso_rt_macro_kernel)

    # add to total 
    int_kernel .+= aniso_rt_macro_kernel .* dA_cpu[i]
end

int_kernel ./= π

plt.plot(vs, iso_rt_macro_kernel)
plt.plot(vs, int_kernel)
# plt.plot(vs, iso_rt_macro_kernel ./ int_kernel)
plt.show()
