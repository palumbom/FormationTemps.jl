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
# mpl.style.use("tableau-colorblind10")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                            \\usepackage{mathrsfs}")

# python interpolation for matplotlib stuff
interp1d = pyimport("scipy.interpolate").interp1d

# set colormaps
img_cmap = "viridis"
μ_cmap = "autumn"
seq_cmap = "Set3"
ncolors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#A6761D", "#66A61E"]


# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

function get_marcs_atm(Teff::T, logg::T, A_X::AA{T,1}; n_layers::Int=240) where T<:AF
    # get the model atmosphere
    marcs_atm = Korg.interpolate_marcs(Teff, logg, A_X)
    τ_500 = Korg.get_tau_refs(marcs_atm)
    zs = Korg.get_zs(marcs_atm)
    Ts = Korg.get_temps(marcs_atm)
    ne = Korg.get_electron_number_densities(marcs_atm)
    nd = Korg.get_number_densities(marcs_atm)

    # interpolate in zs 
    itp_τs = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(τ_500))
    itp_Ts = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(Ts))
    itp_ne = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(ne))
    itp_nd = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(nd))

    zs_new = range(last(zs), first(zs), length=n_layers)
    τs_new = reverse(itp_τs.(zs_new))
    Ts_new = reverse(itp_Ts.(zs_new))
    ne_new = reverse(itp_ne.(zs_new))
    nd_new = reverse(itp_nd.(zs_new))
    zs_new = reverse(collect(zs_new))

    ls = Array{Korg.PlanarAtmosphereLayer{Float64, Float64, Float64, Float64, Float64}}(undef, length(zs_new))
    for i in eachindex(zs_new)
        ls[i] = Korg.PlanarAtmosphereLayer(τs_new[i], zs_new[i], Ts_new[i], ne_new[i], nd_new[i])
    end
    return Korg.PlanarAtmosphere(ls, 5000.0 / 1e8)
end

# make plotdir
plotdir = joinpath(pwd(), "figures")
!isdir(plotdir) && mkdir(plotdir)

#= # get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist][17500:end]
specs = [string(l.species) for l in linelist]

# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# make the wavelength grid
λs_korg = range(first(wls) - 5.0, last(wls) + 5.0, step=0.005)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
marcs_atm = get_marcs_atm(5777.0, 4.44, A_X, n_layers=168 * 3)
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
FT.compute_alpha!(αs, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# get disk integrated cfunc
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0
cfunc_flux = FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux = 2π .* dropdims(sum(cfunc_flux, dims=1), dims=1)

# get formation temperature
cum_cfunc_flux_norm = cumsum(cfunc_flux, dims=1) 
cum_cfunc_flux_norm ./= maximum(cum_cfunc_flux_norm, dims=1)
form_temps_flux = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    local xs = view(cum_cfunc_flux_norm, :, i)
    local itp = FT.linear_interp(xs, elav(Ts))
    form_temps_flux[i] = itp(0.5)
end =#

# get indices with similar formation temperatures
ftemp = 5000.0
idx = isapprox.(ftemp, form_temps_flux, atol=0.1e1)

# find the lines they are nearest
λs_interest = view(λs_korg, idx)
idx_idx = findall(idx)
idx_wls = [FT.searchsortednearest(wls, i) for i in λs_interest]
wls_interest = wls[idx_wls]

# format species name
specs_interest = string.(species[idx_wls])
specs_interest_latex = latexstring.(specs_interest)
for i in eachindex(specs_interest)
    parts = split(specs_interest[i])
    part3 = string(round(wls_interest[i], digits=1))
    specs_interest_latex[i] = L"{\rm %$(parts[1])\, %$(parts[2])\, %$part3\, \AA}"
end

# get colors 
cmap = plt.get_cmap(seq_cmap)
norm = mpl.colors.Normalize(vmin=1, vmax=length(wls_interest))
colors = cmap(norm(1:length(wls_interest)))

# make figure objects
fig, ax1 = plt.subplots(figsize=(9.2,4.8))

# iterate over lines
xticks = zeros(length(idx_wls))
for i in eachindex(idx_wls)
    # isolate the lines
    buffer = 25
    idx_λs = findfirst(x -> x .>= wls[idx_wls[i]], λs_korg)

    # get an offset 
    offset = 0.3 * (i - 1)
    xticks[i] = offset

    # plot the lines
    λs_view = view(λs_korg, idx_λs-buffer:idx_λs+buffer) .- wls[idx_wls[i]] .+ offset
    flux_view = view(flux, idx_λs-buffer:idx_λs+buffer)
    temp_view = view(form_temps_flux, idx_λs-buffer:idx_λs+buffer)
    # ax1.plot(λs_view, temp_view, zorder=0, c=colors[i,:])
    ax1.plot(λs_view, temp_view, zorder=0, c=ncolors[i])

    # get data for scatter
    xscatter = [λs_korg[idx_idx[i]]] .- wls[idx_wls[i]] .+ offset
    yscatter = [form_temps_flux[idx_idx[i]]]
    ax1.scatter(xscatter, yscatter, c="k", zorder=1)
end
ax1.set_xticks(xticks)
ax1.set_xticklabels(specs_interest_latex, rotation=45, ha="right")
# ax1.set_xticks([])

# ax1.set_xlabel(L"{\rm Air\ Wavelength\ +\ Offset\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
fig.tight_layout()
fig.savefig(joinpath(plotdir, "line_lineup.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# get views of cfuncs at indices of interest
cfuncs_sim = view(cfunc_flux, :, idx)
cfuncs_cum_sim = view(cum_cfunc_flux_norm, :, idx)

# get exponent for units
max_val = maximum(abs.(cfuncs_sim))
exponent = floor(Int, log10(max_val))

# plot each curve
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9.2, 4.8), sharex=true)
for i in eachindex(idx_wls)
    ax1.plot(elav(Ts), cfuncs_sim[:,i] / 10^exponent, c=ncolors[i])
    ax2.plot(elav(Ts), cfuncs_cum_sim[:,i], c=ncolors[i], label=specs_interest_latex[i])
end

ax1.set_xlabel(L"{\rm Temperature\ [K]}")
ax2.set_xlabel(L"{\rm Temperature\ [K]}")

ax1.set_ylabel(L"\mathscr{C}_{\nu}(t_\nu)\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1}]}")
ax2.set_ylabel(L"{\rm Normalized\ Cumulative\ Flux\ Cont.\ Fn.}")
ax2.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

fig.subplots_adjust(wspace=0.25)
fig.savefig(joinpath(plotdir, "mean_comparison.pdf"), bbox_inches="tight")
plt.clf(); plt.close()