using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
# mpl.style.use("tableau-colorblind10")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[1:5000]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]

# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# make the wavelength grid
# λs_korg = range(3500, 7000.0, step=0.01)
λs_korg = range(first(wls) - 0.5, last(wls) + 0.5, step=0.01)
println(length(λs_korg))

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

# get the absorption coeffs
αs = zeros(length(atm_gpu.zs), length(λs_korg))
αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# loop over mus 
μs = range(0.1, 1.0, length=10)
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0
cfuncs_int = zeros(length(zs)-1, length(λs_korg), length(μs))
cfuncs_int_cont = zeros(length(zs)-1, length(λs_korg), length(μs))
intensities = zeros(length(λs_korg), length(μs))
continuum_int = zeros(length(λs_korg), length(μs))

for i in eachindex(μs)
    cfuncs_int[:,:,i] .= FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    intensities[:,i] .= dropdims(sum(view(cfuncs_int,:,:,i), dims=1), dims=1)

    cfuncs_int_cont[:,:,i] .= FT.calc_intensity_cfunc(αs_cont, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    continuum_int[:,i] .= dropdims(sum(view(cfuncs_int_cont,:,:,i), dims=1), dims=1)
end
 
# get cfunc for flux
cfunc_flux = FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux = 2π .* dropdims(sum(cfunc_flux, dims=1), dims=1)

# get disk integrated continuum
cfunc_flux_cont = FT.calc_flux_cfunc(αs_cont, atm_gpu, gpu_mem, cmem, σ_v)
continuum_flux = 2π .* dropdims(sum(cfunc_flux_cont, dims=1), dims=1)

# now get cumulative contribution functions
cum_cfuncs_int_norm = cumsum(cfuncs_int, dims=1) 
cum_cfuncs_int_norm ./= maximum(cum_cfuncs_int_norm, dims=1)
cum_cfunc_flux_norm = cumsum(cfunc_flux, dims=1) 
cum_cfunc_flux_norm ./= maximum(cum_cfunc_flux_norm, dims=1)

# now compute the formation temperature
form_temps_intensity = zeros(length(λs_korg), length(μs))
form_temps_flux = zeros(length(λs_korg))

# loop over wavelength
Ts_elav = elav(Ts)
for i in eachindex(λs_korg)
    # flux form temp
    local xs_flux = view(cum_cfunc_flux_norm, :, i)
    local itp_flux = FT.linear_interp(xs_flux, Ts_elav)
    form_temps_flux[i] = itp_flux(0.5)

    # loop over disk position
    for j in eachindex(μs)
        # intensity form temp
        local xs_int = view(cum_cfuncs_int_norm, :, i, j)
        local itp_int = FT.linear_interp(xs_int, Ts_elav)
        form_temps_intensity[i,j] = itp_int(0.5)
    end
end

# write it out 
zs = Array(zs)
Ts = Array(Ts)
jldsave(joinpath(FT.datdir, "solar_temps.jld2"); 
        λs_korg, zs, Ts, τ_500, μs,
        intensities, cfuncs_int, 
        continuum_int, cfuncs_int_cont,
        flux, cfunc_flux, 
        continuum_flux, cfunc_flux_cont,
        form_temps_intensity, form_temps_flux)


# sanity check against Korg
korg_res = Korg.synthesize(marcs_atm, linelist, A_X, λs_korg, 
                           vmic=0.0,  tau_scheme="bezier", 
                           mu_values=μs, hydrogen_lines=false)
korg_flux = korg_res.flux
korg_cntm = korg_res.cntm
korg_mu = korg_res.mu_grid
korg_int = collect(korg_res.intensity')


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.2, 4.8))
ax1.plot(λs_korg, intensities[:,end])
ax1.plot(λs_korg, korg_int[:,end])

ax2.plot(λs_korg, flux ./ continuum_flux, label="me")
ax2.plot(λs_korg, korg_flux ./ korg_cntm, label="Korg")
ax2.legend()
plt.show()
