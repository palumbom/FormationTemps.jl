using NPZ
using Korg
using GRASS
using PyPlot
using FITSIO
using Anemoi
using Statistics
using Downloads
using Interpolations
using SpecialFunctions
using ImageFiltering
using CSV, HDF5, DataFrames
using EchelleCCFs: λ_air_to_vac, λ_vac_to_air

AF = AbstractFloat
AA = AbstractArray

outpath = joinpath(homedir(), "ceph", "formation_temps")

# make the wavelength grid
λs_korg = range(3000.0, 1e5, step=0.01)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the model atmosphere
marcs_atm = Korg.interpolate_marcs(5777, 4.44, A_X)
atmosphere = marcs_atm

# parse out atmosphere "coordinates"
τ_500 = Korg.get_tau_5000s(atmosphere)
zs = Korg.get_zs(atmosphere)
temps = Korg.get_temps(atmosphere)

# get the elavs
temps_middle = elav(temps)
zs_middle = elav(zs)
τ_500_middle = elav(τ_500)

# do the line synthesis
vmic = 1.2  # CHANGE ME
sol = synthesize(atmosphere, [], A_X, λs_korg; vmic=vmic, tau_scheme="bezier", 
                 hydrogen_lines=false, use_MHD_for_hydrogen_lines=false)
flux = sol.flux

# get the absorption coefficients
αs = deepcopy(sol.alpha)

# compute the source function
sfunc = Korg.blackbody.(temps, (λs_korg .* 1e-8)')

# make grid of mus 
μs = range(1.0, 0.1, step=-0.05)

# allocate 
τs = similar(αs)
flux_out = zeros(length(λs_korg), length(μs))
form_temps = zeros(length(λs_korg), length(μs))
form_heights = zeros(length(λs_korg), length(μs))
form_tau_ref = zeros(length(λs_korg), length(μs))

# loop over mu
for i in eachindex(μs)
    # compute the optical depths
    τs .= 0.0
    ss = zs ./ μs[i]
    for i in axes(αs, 2)
        Korg.RadiativeTransfer.compute_tau_bezier!(view(τs, :, i), ss, view(αs, :, i))
    end

    # get the contribution functions and flux 
    cfunc = elav(sfunc, dims=1) .* elav(exp.(-τs), dims=1)
    cfunc .*= (diff(τs, dims=1) ./ diff(ss .* -1.0))
    cfunc_cum = cumsum(cfunc, dims=1)
    flux_out[:,i] .= vec(sum(cfunc .* diff(ss .* -1.0), dims=1))

    # compute the formation temps
    for j in eachindex(λs_korg)
        norm_cfunc = view(cfunc_cum, :, j) ./ maximum(view(cfunc_cum, :, j))
        temp_func = GRASS.linear_interp(norm_cfunc, temps_middle)
        form_temps[j,i] = temp_func(0.5)

        height_func = GRASS.linear_interp(norm_cfunc, zs_middle)
        form_heights[j,i] = height_func(0.5)

        tau_func = GRASS.linear_interp(norm_cfunc, τ_500_middle)
        form_tau_ref[j,i] = tau_func(0.5)
    end
end

# do disk-integrated 
τs .= 0.0
ss = zs ./ 1.0
for i in axes(αs, 2)
    Korg.RadiativeTransfer.compute_tau_bezier!(view(τs, :, i), ss, view(αs, :, i))
end

# get the contribution functions and flux 
cfunc = elav(sfunc, dims=1) .* elav(SpecialFunctions.expint.(2, τs), dims=1) .* diff(τs, dims=1)
cfunc_cum = cumsum(cfunc, dims=1)
flux_integrated = 2π .* vec(sum(elav(cfunc, dims=1), dims=1))

# compute the formation temps
form_temps_integrated = zeros(length(λs_korg))
form_heights_integrated = zeros(length(λs_korg))
form_tau_ref_integrated = zeros(length(λs_korg))
for j in eachindex(λs_korg)
    norm_cfunc = view(cfunc_cum, :, j) ./ maximum(view(cfunc_cum, :, j))
    temp_func = GRASS.linear_interp(norm_cfunc, temps_middle)
    form_temps_integrated[j] = temp_func(0.5)

    height_func = GRASS.linear_interp(norm_cfunc, zs_middle)
    form_heights_integrated[j] = height_func(0.5)

    tau_func = GRASS.linear_interp(norm_cfunc, τ_500_middle)
    form_tau_ref_integrated[j] = tau_func(0.5)
end

# write to h5 
data = Dict("vac_wavs" => collect(λs_korg), 
            "mus" => collect(μs),
            "flux" => flux_out, 
            "form_temps" => form_temps,
            "form_heights" => form_heights,
            "form_tau_refs" => form_tau_ref,
            "flux_integrated" => flux_integrated, 
            "form_temps_integrated" => form_temps_integrated,
            "form_heights_integrated" => form_heights_integrated,
            "form_tau_refs_integrated" => form_tau_ref_integrated)            

fname = joinpath(outpath, "continuum_formation_by_mu.h5")
h5open(fname, "w") do file
    for (k, v) in data
        write(file, k, v)
    end
end