using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                            \\usepackage{mathrsfs}")

# python interpolation for matplotlib stuff
interp1d = pyimport("scipy.interpolate").interp1d

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
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.01)
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
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0

# get the nominal answer
cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
cfunc_flux_cum_stationary = Array(FT.get_cum_cfunc(cfunc_flux_stationary))
flux_stationary = Array(FT.get_flux(cfunc_flux_stationary))

# get disk stuff 
ρstar = 1.0
istar = 90.0
v0 = 0.0
Nϕ = 128
μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

# flatten, move to cpu
idx = findall(x -> x .> 0.0, Array(μs))
μs_cpu = Array(μs)[idx]
dA_cpu = Array(dA)[idx]
z_rot_cpu = Array(z_rot)[idx]

# allocate for output
ints = zeros(length(λs_korg), length(μs_cpu))
flux_rotating = zeros(length(λs_korg))
cfunc_flux_rotating = zeros(length(zs)-1, length(λs_korg))

@showprogress for i in eachindex(μs_cpu)
    # set rotational velocity
    μ_v .= z_rot_cpu[i] .* FT.c_ms

    # get the cfunc and stuff
    cfunc_intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v, σ_v)
    
    # tabulate
    ints[:,i] .= Array(FT.get_intensity(cfunc_intensity))

    # add to disk integration
    flux_rotating .+= ints[:, i] .* dA_cpu[i]
    cfunc_flux_rotating .+= Array(cfunc_intensity.cfunc_dt .* dA_cpu[i])
end

# convert units
flux_rotating .*= 1e-8
cfunc_flux_rotating .*= 1e-8

# now get cumulative cfuncs 
cum_cfunc_flux_rotating = cumsum(cfunc_flux_rotating, dims=1)
cum_cfunc_flux_rotating ./= maximum(cum_cfunc_flux_rotating, dims=1)

# loop over wavelength
form_temp_stationary = zeros(length(λs_korg))
form_temp_rotating = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cfunc_flux_cum_stationary, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_stationary[i] = itp(0.5)

    xs = view(cum_cfunc_flux_rotating, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_rotating[i] = itp(0.5)
end

# overplot the flux
@show extrema((flux_rotating - flux_stationary) ./ flux_stationary)
@show extrema((form_temp_rotating - form_temp_stationary) ./ form_temp_stationary)

fig, ax1 = plt.subplots()
ax1.plot(λs_korg, flux_stationary, c="k", ls="--", label="Stationary")
ax1.plot(λs_korg, flux_rotating, c="tab:blue", label="Solid Body Rotation")
ax1.set_xlabel("Wavelength")
ax1.set_ylabel("Flux")
ax1.legend()
fig.savefig("figures/flux_rotation.pdf", bbox_inches="tight")
plt.clf(); plt.close();

# overplot the temperature
fig, ax1 = plt.subplots()
ax1.plot(λs_korg, form_temp_stationary, c="k", ls="--", label="Stationary")
ax1.plot(λs_korg, form_temp_rotating, c="tab:blue", label="Solid Body Rotation")
ax1.set_xlabel("Wavelength")
ax1.set_ylabel("Formation Temperature")
ax1.legend()
fig.savefig("figures/temp_rotation.pdf", bbox_inches="tight")
plt.clf(); plt.close();