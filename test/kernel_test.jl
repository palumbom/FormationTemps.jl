using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using FFTW
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib
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

# make the wavelength grid
buffer = 0.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.0005)
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
cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
cfunc_flux_cum = Array(FT.get_cum_cfunc(cfunc_flux_stationary))
flux_stationary = Array(FT.get_flux(cfunc_flux_stationary))

form_temp_stationary = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cfunc_flux_cum, :, i)
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
ys = Array(cfunc_flux_stationary.cfunc_dt)
intres = 1024

N = length(xs)
λ0 = mean(xs)
vs = FT.c_ms .* (xs .- λ0) ./ λ0
Δv = (last(vs) - first(vs)) / (N - 1)
dv = diff(vs)

shift = -1

# hirano kernel no rot
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, 0.0, ζ_rt; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_no_rot = circshift(k_ctr ./ sum(k_ctr), shift)

# hirano kernel no mac
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, vsini, 0.0; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_no_macro = circshift(k_ctr ./ sum(k_ctr), shift)

# hirano rotmacro
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_rot_macro = circshift(k_ctr ./ sum(k_ctr), shift)

# get the gray rt kernel and rotation kernel
rt_macro_kernel = FT.gray_rt_macro_kernel(vs, ζ_rt)
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

# plot the RT case
plt.close("all")
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
ax1.plot(λs_korg, rt_macro_kernel, label="gray")
ax1.plot(λs_korg, hirano_no_rot, label="hirano")
ax2.scatter(λs_korg, hirano_no_rot .- rt_macro_kernel, c="tab:blue", s=2)
ax1.set_xlim(6301.8, 6302.2)
ax1.legend()
ax1.set_title("Macro Only")
plt.show()

# plot the vsini case
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
ax1.plot(λs_korg, gray_rot_kernel, label="gray")
ax1.plot(λs_korg, hirano_no_macro, label="hirano")
ax2.scatter(λs_korg, hirano_no_macro .- gray_rot_kernel, c="tab:blue", s=2)
ax1.set_xlim(6301.8, 6302.2)
ax1.set_title("Rotation Only")
ax1.legend()
plt.show()

# now get contribution functions + flux
cfunc_flux_hirano_norot = FT.convolve_hirano_rotmacro(xs, ys, 0.0, ζ_rt, u1, u2, intres=intres)
cfunc_flux_hirano_nomacro = FT.convolve_hirano_rotmacro(xs, ys, vsini, 0.0, u1, u2, intres=intres)
cfunc_flux_hirano_rotmacro = FT.convolve_hirano_rotmacro(xs, ys, vsini, ζ_rt, u1, u2, intres=intres)

cfunc_flux_rotgray = FT.convolve_gray_rotation(xs, ys, vsini, u1)
cfunc_flux_macrogray = FT.convolve_gray_rt_macro(xs, ys, ζ_rt)

flux_hirano_norot = dropdims(sum(cfunc_flux_hirano_norot, dims=1), dims=1)
flux_hirano_nomacro = dropdims(sum(cfunc_flux_hirano_nomacro, dims=1), dims=1)

flux_rotgray = dropdims(sum(cfunc_flux_rotgray, dims=1), dims=1)
flux_macrogray = dropdims(sum(cfunc_flux_macrogray, dims=1), dims=1)

# plot the RT case
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
ax1.plot(λs_korg, flux_macrogray, label="gray")
ax1.plot(λs_korg, flux_hirano_norot, label="hirano")
ax2.scatter(λs_korg, 100 .* (flux_hirano_norot .- flux_macrogray) ./ flux_hirano_norot, c="tab:blue", s=2)
ax1.legend()
ax1.set_title("Macro Only")
plt.show()

# plot the vsini case
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
ax1.plot(λs_korg, flux_rotgray, label="gray")
ax1.plot(λs_korg, flux_hirano_nomacro, label="hirano")
ax2.scatter(λs_korg, 100 .* (flux_hirano_nomacro .- flux_rotgray) ./ flux_hirano_nomacro, c="tab:blue", s=2)
ax1.legend()
ax1.set_title("Rotation Only")
plt.show()
