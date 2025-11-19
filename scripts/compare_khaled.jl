using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, NPZ, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib
using ProgressMeter

# matplotlib backend
plt.ioff()
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                            \\usepackage{mathrsfs}")

# read in khaled's
kfile = joinpath(FT.datdir, "Sun_SME.npz")
df = npzread(kfile)

# parse out 
flux_k = df["flux"]
temp_k = df["T1o2"]
wave_k = Korg.vacuum_to_air.(df["wave"])
cfunc_k = df["cont_abs"]

# get index
idx1 = findfirst(x -> x .>= 6301, wave_k)
idx2 = findfirst(x -> x .>= 6303, wave_k)

# slice it
flux_k = flux_k[idx1:idx2]
temp_k = temp_k[idx1:idx2]
wave_k = wave_k[idx1:idx2]
cfunc_k = cfunc_k[idx1:idx2, :]

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
λs_korg = range(first(wls) - 0.5, last(wls) + 0.5, step=0.005)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
marcs_atm = FT.get_marcs_atm(5770.0, 4.0, A_X, n_layers=56)
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

# get the absorption coeffs
αs = zeros(length(atm_gpu.zs), length(λs_korg))
αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 400
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# broadening
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 850.0
μ_v_rot = CUDA.zeros(Float64, length(zs))

# memory for convolution
cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

# params for convolution
@load joinpath(FT.datdir, "ld_coeffs.jld2") u1 u2
vsini = 1630.0
vmac = 3980.0

# get intensity stuff
cfunc_int = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, 1.0, μ_v, σ_v)
cfunc_int_cum = Array(FT.get_cum_cfunc(cfunc_int))
intensity = Array(FT.get_intensity(cfunc_int))

cfunc_int_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, 1.0, μ_v, σ_v)
cfunc_int_cont_cum = Array(FT.get_cum_cfunc(cfunc_int_cont))
intensity_cont = Array(FT.get_intensity(cfunc_int_cont))

# get flux
cfunc_flux = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
cfunc_flux_cum = Array(FT.get_cum_cfunc(cfunc_flux))
flux = Array(FT.get_flux(cfunc_flux))

cfunc_flux_cont = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v)
cfunc_flux_cont_cum = Array(FT.get_cum_cfunc(cfunc_flux_cont))
flux_cont = Array(FT.get_flux(cfunc_flux_cont))

# convolve
tbc = cfunc_flux.cfunc_dt
cfunc_flux_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini, vmac, u1, u2))
flux_convolution = 2π .* dropdims(sum(cfunc_flux_convolution, dims=1), dims=1)

# convolve
tbc = cfunc_flux_cont.cfunc_dt
cfunc_flux_cont_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini, vmac, u1, u2))
flux_cont_convolution = 2π .* dropdims(sum(cfunc_flux_cont_convolution, dims=1), dims=1)

# normalize
flux_norm = flux ./ flux_cont
flux_norm_conv = flux_convolution ./ flux_cont_convolution

# get disk stuff 
ρstar = 1.0
istar = 90.0
v0 = vsini
Nϕ = 24
μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)
μs_cpu = Array(μs)
dA_cpu = Array(dA)
z_rot_cpu = Array(z_rot)

# allocate for output
flux_integration = zeros(length(λs_korg))
flux_cont_integration = zeros(length(λs_korg))
cfunc_flux_integration = zeros(length(zs)-1, length(λs_korg))

# do the disk integration
@showprogress for i in eachindex(μs_cpu)
    μs_cpu[i] <= 0.0 && continue

    # set the rotational velocity
    μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

    # get the intensity contribution function
    cfunc_intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v, σ_v)
    cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v, σ_v)

    # convolve the cfunc with RT macroturbulence
    tbc = cfunc_intensity.cfunc_dt
    cfunc_int_i_mac = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, λs_korg, tbc, vmac))

    # convolve the cfunc with RT macroturbulence
    tbc = cfunc_intensity_cont.cfunc_dt
    cfunc_int_cont_i_mac = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, λs_korg, tbc, vmac))

    # add to the flux integral
    flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
    flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[i]
    cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]
end

# normalize
flux_norm_int = flux_integration ./ flux_cont_integration

# now get cumulative cfuncs 
cum_cfunc_intensity = Array(FT.get_cum_cfunc(cfunc_int))
cum_cfunc_flux = Array(FT.get_cum_cfunc(cfunc_flux))

cum_cfunc_conv = cumsum(cfunc_flux_convolution, dims=1)
cum_cfunc_conv ./= maximum(cum_cfunc_conv, dims=1)
cum_cfunc_int = cumsum(cfunc_flux_integration, dims=1)
cum_cfunc_int ./= maximum(cum_cfunc_int, dims=1)

# loop over wavelength
form_temp_intensity = zeros(length(λs_korg))
form_temp_flux = zeros(length(λs_korg))
form_temp_integration = zeros(length(λs_korg))
form_temp_convolution = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cum_cfunc_intensity, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_intensity[i] = itp(0.5)

    xs = view(cum_cfunc_flux, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_flux[i] = itp(0.5)

    xs = view(cum_cfunc_conv, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_convolution[i] = itp(0.5)

    xs = view(cum_cfunc_int, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_integration[i] = itp(0.5)
end

# compare 
plt.plot(wave_k, temp_k, label="Khaled")
plt.plot(λs_korg, form_temp_intensity, label="Intensity")
plt.plot(λs_korg, form_temp_flux, label="Stationary Flux")
plt.plot(λs_korg, form_temp_integration, label="Int. Flux.")
plt.plot(λs_korg, form_temp_convolution, label="Conv. Flux")
plt.legend()
plt.show()

# plt.plot(wave_k, flux_k)
# plt.plot(λs_korg, flux_norm_conv)
# plt.plot(λs_korg, flux_norm_int)
# plt.show()