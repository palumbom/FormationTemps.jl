using Revise, Anemoi
using FormationTemps; FT = FormationTemps
using Korg, GRASS
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(GRASS.moddir * "fig.mplstyle")

# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

# output directory
outdir = abspath("/mnt/ceph/users/mpalumbo/data_for_ryan")

# get the model atmosphere
Teff = 5777
logg = 4.44
A_X = deepcopy(Korg.asplund_2020_solar_abundances)
marcs_atm = Korg.interpolate_marcs(Teff, logg, A_X)
τ_500 = Korg.get_tau_5000s(marcs_atm)
zs = Korg.get_zs(marcs_atm)
Ts = Korg.get_temps(marcs_atm)
ne = Korg.get_electron_number_densities(marcs_atm)
nd = Korg.get_number_densities(marcs_atm)

# get the line list
valdfile = joinpath(FT.datdir, "full_linelist.lin")
linelist = Korg.read_linelist(valdfile)
linelist = linelist[154_500:155_500]

# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# make the wavelength grid
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=0.005)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# make my atmosphere 
atm_gpu = AtmosphereGPU(marcs_atm)
zs = atm_gpu.zs
Ts = atm_gpu.Ts
τ5000 = atm_gpu.τs

# synthesis to get the alphas
αs = zeros(length(atm_gpu.zs), length(λs_korg))
Anemoi.compute_alpha!(αs, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = Anemoi.ConvolutionMemory(Nλ, Natm, Npad)

# loop over mus 
μs = range(0.1, 1.0, length=10)
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0
cfuncs = zeros(length(zs)-1, length(λs_korg), length(μs))
intensities = zeros(length(λs_korg), length(μs))

for i in eachindex(μs)
    cfuncs[:,:,i] .= Anemoi.calculate_cfunc(αs, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    intensities[:,i] .= dropdims(sum(view(cfuncs,:,:,i), dims=1), dims=1)
end
 
# get disk integrated cfunc
cfunc_flux = Anemoi.calculate_cfunc_disk_integrated(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux_disk_integrated = 2π .* dropdims(sum(cfunc_flux, dims=1), dims=1)

# now get cumulative contribution functions
cum_cfuncs_norm = cumsum(cfuncs, dims=1) 
cum_cfuncs_norm ./= maximum(cum_cfuncs_norm, dims=1)
cum_cfunc_flux_norm = cumsum(cfunc_flux, dims=1) 
cum_cfunc_flux_norm ./= maximum(cum_cfunc_flux_norm, dims=1)

# now compute the formation temperature
form_temps_intensity = zeros(length(λs_korg), length(μs))
form_heights_intensity = zeros(length(λs_korg), length(μs))
form_tau_intensity = zeros(length(λs_korg), length(μs))
form_temps_flux = zeros(length(λs_korg))
form_heights_flux = zeros(length(λs_korg))
form_tau_flux = zeros(length(λs_korg))

for i in eachindex(λs_korg)
    local xs = view(cum_cfunc_flux_norm, :, i)
    local itp = GRASS.linear_interp(xs, elav(Ts))
    form_temps_flux[i] = itp(0.5)

    local itp = GRASS.linear_interp(xs, elav(zs))
    form_heights_flux[i] = itp(0.5)

    local itp = GRASS.linear_interp(xs, elav(τ5000))
    form_tau_flux[i] = itp(0.5)
end

for i in eachindex(λs_korg)
    for j in eachindex(μs)
        local xs = view(cum_cfuncs_norm, :, i, j)
        local itp = GRASS.linear_interp(xs, elav(Ts))
        form_temps_intensity[i,j] = itp(0.5)

        local itp = GRASS.linear_interp(xs, elav(zs))
        form_heights_intensity[i,j] = itp(0.5)

        local itp = GRASS.linear_interp(xs, elav(τ5000))
        form_tau_intensity[i,j] = itp(0.5)
    end
end

cmap = plt.get_cmap("autumn")
norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=maximum(μs))
cs = cmap(norm(μs))

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8.6, 12.8))
ax1, ax2, ax3 = axs
for i in eachindex(μs)
    mu_val = μs[i]
    ax1.plot(λs_korg, form_temps_intensity[:,i],  c=cs[i,:], label=L"\mu = %$mu_val")
    ax2.plot(λs_korg, form_heights_intensity[:,i] ./ 1e7,  c=cs[i,:], label=L"\mu = %$mu_val")
    ax3.plot(λs_korg, form_tau_intensity[:,i],  c=cs[i,:], label=L"\mu = %$mu_val")
end
ax1.plot(λs_korg, form_temps_flux, c="k", label=L"{\rm Flux}")
ax2.plot(λs_korg, form_heights_flux ./ 1e7, c="k", label=L"{\rm Flux}")
ax3.plot(λs_korg, form_tau_flux, c="k", label=L"{\rm Flux}")

ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax3.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
ax2.set_ylabel(L"z_{1/2}\ {\rm [Mm]}")
ax3.set_ylabel(L"\tau^{\rm ref}_{1/2}")
ax2.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
# ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

idx1 = findfirst(x -> x .>= first(wls) - 1.25, λs_korg)
idx2 = findfirst(x -> x .>= last(wls) + 1.25, λs_korg)
ax1.set_xlim(λs_korg[idx1], λs_korg[idx2])

fig.tight_layout()
fig.savefig(joinpath(outdir, "form_temps.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# write out data
data = Dict("air_wavs" => collect(λs_korg), 
            "mus" => collect(μs),
            "intensity" => intensities,
            "flux" => flux_disk_integrated,
            "cfunc_intensity" => cfuncs, 
            "cfunc_flux" => cfunc_flux,
            "form_temps_intensity" => form_temps_intensity,
            "form_heights_intensity" => form_heights_intensity,
            "form_tau_refs_intensity" => form_tau_intensity,
            "form_temps_flux" => form_temps_flux,
            "form_heights_flux" => form_heights_flux,
            "form_tau_refs_flux" => form_tau_flux)            

fname = joinpath(outdir, "formation_metrics_for_ryan.h5")
h5open(fname, "w") do file
    for (k, v) in data
        write(file, k, v)
    end
end