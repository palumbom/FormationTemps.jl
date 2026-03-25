using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using FFTW
using CSV, DataFrames, Statistics
using ProgressMeter

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pyconvert
using LaTeXStrings
mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

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
buffer = 0.5
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

# get the formation temperature for a stationary star
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_stationary = Array(FT.get_flux(cfunc_flux_struct))
cum_cfunc_flux_stationary = Array(FT.get_cum_cfunc(cfunc_flux_struct))

form_temp_stationary = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cum_cfunc_flux_stationary, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_stationary[i] = itp(0.5)
end

# set rotational and macroturbulence 
vsini = 2100.0
ζ_rt = 1400.0

# set limb darkening
u1 = 0.4
u2 = 0.0

xs = λs_korg
ys = Array(cfunc_flux_struct.cfunc_dt)
intres = range(50, 1000, step=50)
intres = 10_000

# for i in intres
N = length(xs)
λ0 = mean(xs)
vs = FT.c_ms .* (xs .- λ0) ./ λ0
Δv = (last(vs) - first(vs)) / (N - 1)
dv = diff(vs)

# hirano kernel no rot
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, 0.0, ζ_rt; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_no_rot = k_ctr ./ sum(k_ctr)

# hirano kernel no mac
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, vsini, 0.0; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_no_macro = k_ctr ./ sum(k_ctr)

# hirano rotmacro
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_rot_macro = k_ctr ./ sum(k_ctr)

# get the gray rt kernel and rotation kernel
rt_macro_kernel = FT.gray_iso_rt_macro_kernel(vs, ζ_rt)
gray_rot_kernel = FT.gray_rot_kernel(vs, vsini, u1)

# get isotropic gaussian
σ_g(x) = x * (ζ_rt / FT.c_ms)
g(x, n) = exp(-((x - n) / σ_g(x))^2.0)

# offset the kernel by the velocity
λ0 = mean(λs_korg)
λc = λ0

# sample the kernel
gaussian = g.(λs_korg, λc)
gaussian ./= sum(gaussian)

# now get contribution functions + flux
cfunc_flux_hirano_norot = FT.convolve_hirano_rotmacro(xs, ys, 0.0, ζ_rt, u1, u2, intres=intres)
cfunc_flux_hirano_nomacro = FT.convolve_hirano_rotmacro(xs, ys, vsini, 0.0, u1, u2, intres=intres)
cfunc_flux_hirano_rotmacro = FT.convolve_hirano_rotmacro(xs, ys, vsini, ζ_rt, u1, u2, intres=intres)

cfunc_flux_rotgray = FT.convolve_gray_rotation(xs, ys, vsini, u1)
cfunc_flux_macrogray = FT.convolve_iso_rt_macro(xs, ys, ζ_rt)

flux_hirano_norot = dropdims(sum(cfunc_flux_hirano_norot, dims=1), dims=1)
flux_hirano_nomacro = dropdims(sum(cfunc_flux_hirano_nomacro, dims=1), dims=1)

flux_rotgray = dropdims(sum(cfunc_flux_rotgray, dims=1), dims=1)
flux_macrogray = dropdims(sum(cfunc_flux_macrogray, dims=1), dims=1)

# overplot the kernels
fig, ax1 = plt.subplots()
ax1.plot(vs .- 9600, gray_rot_kernel ./ maximum(gray_rot_kernel), c=ncolors[1], label=L"\mathrm{Rotation}")
ax1.plot(vs .- 3200, rt_macro_kernel ./ maximum(rt_macro_kernel), c=ncolors[2], ls="--", label=L"\mathrm{Isotropic\ RT}")
ax1.plot(vs .+ 3200, hirano_rot_macro ./ maximum(hirano_rot_macro), c=ncolors[3], ls="-.", label=L"\mathrm{Rotation\ \&\ RT}")
ax1.plot(vs .+ 9600, gaussian ./ maximum(gaussian), c=ncolors[7], ls=":", label=L"\mathrm{Isotropic\ Gaussian}")
#  ax1.legend(loc="upper left", fontsize=12)
ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
ax1.set_xlabel(L"\Delta v\ {\rm [m\ s}^{-1} {\rm ]}")
ax1.set_ylabel(L"{\rm Normalized\ Kernel}")
ax1.set_xlim(-13500, 13500)
#  ax1.set_ylim(-0.1, 1.2)
plt.savefig("figures/kernels.pdf", bbox_inches="tight")
plt.clf(); plt.close()
