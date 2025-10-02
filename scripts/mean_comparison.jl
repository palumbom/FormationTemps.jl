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

# animation
using PyCall; animation = pyimport("matplotlib.animation");
pe = pyimport("matplotlib.patheffects");

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

# make plotdir
plotdir = joinpath(pwd(), "figures")
framedir = joinpath(plotdir, "cont_frames")
!isdir(plotdir) && mkdir(plotdir)
!isdir(framedir) && mkdir(framedir)

function synth_given_linelist(linelist; δλ=0.005)
    # re-get values
    wls = [l.wl * 1e8 for l in linelist]
    log_gf =  [l.log_gf for l in linelist]
    species =  [l.species for l in linelist]
    E_lower =  [l.E_lower for l in linelist]
    gamma_rad =  [l.gamma_rad for l in linelist]
    gamma_stark =  [l.gamma_stark for l in linelist]

    # make the wavelength grid
    λs_korg = range(first(wls) - 5.0, last(wls) + 5.0, step=δλ)

    # get some abundances
    A_X = Korg.asplund_2020_solar_abundances

    # get the atmosphere
    marcs_atm = FT.get_marcs_atm(5777.0, 4.44, A_X, n_layers=168 * 3)
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
    end
    return λs_korg, cfunc_flux, flux, cum_cfunc_flux_norm, form_temps_flux, Ts
end

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist][16000:end]
wls = [l.wl * 1e8 for l in linelist]
species = [l.species for l in linelist]

# do the synthesis
λs_korg, cfunc_flux, flux, cum_cfunc_flux_norm, form_temps_flux, Ts = synth_given_linelist(linelist)

# get indices with similar formation temperatures
ftemp = 4750.0
atol = 0.5e1
all_idx = findall(isapprox.(ftemp, form_temps_flux, atol=atol))
# idx = all_idx[sort(rand(1:length(all_idx), 10))]
idx = all_idx[1:min(10, length(all_idx))]

# find the lines they are nearest
λs_interest = view(λs_korg, idx)
idx_wls = [FT.searchsortednearest(wls, i) for i in λs_interest]
wls_interest = wls[idx_wls]

# redo the synthesis just with these lines on a finer grid
δλ = 0.0005
linelist = linelist[idx_wls]
λs_korg, cfunc_flux, flux, cum_cfunc_flux_norm, form_temps_flux, Ts = synth_given_linelist(linelist, δλ=δλ)
wls = [l.wl * 1e8 for l in linelist]
species =  [l.species for l in linelist]

# format species name
specs_interest = string.(species)
specs_interest_latex = latexstring.(specs_interest)
for i in eachindex(specs_interest)
    parts = split(specs_interest[i])
    part3 = string(round(wls_interest[i], digits=1))
    specs_interest_latex[i] = L"{\rm %$(parts[1])\, %$(parts[2])\, %$part3\, \AA}"
end

# get views of the lines
wavs_list = []
flux_list = [] 
temp_list = []
cfunc_list = []
cfunc_cum_list = [] 

buffer = ceil(Int, 0.25 / mean(diff(λs_korg)))
offset_scale = 0.65
for i in eachindex(wls)
    # isolate the lines
    idx_λs = findfirst(x -> x .>= wls[i], λs_korg)

    # get an offset 
    offset = offset_scale * (i - 1)

    # take views
    λs_view = view(λs_korg, idx_λs-buffer:idx_λs+buffer) .- wls[i] .+ offset
    flux_view = view(flux, idx_λs-buffer:idx_λs+buffer)
    temp_view = view(form_temps_flux, idx_λs-buffer:idx_λs+buffer)
    cfunc_view = view(cfunc_flux, :, idx_λs-buffer:idx_λs+buffer)
    cfunc_cum_view = view(cum_cfunc_flux_norm, :, idx_λs-buffer:idx_λs+buffer)

    # push 
    push!(wavs_list, collect(λs_view))
    push!(flux_list, collect(flux_view))
    push!(temp_list, collect(temp_view))
    push!(cfunc_list, collect(cfunc_view))
    push!(cfunc_cum_list, collect(cfunc_cum_view))
end

# find temperatures to loop over
min_temp = maximum(minimum.(temp_list))
max_temp = minimum(maximum.(temp_list))
ftemps = range(ceil(min_temp+1), floor(max_temp-1), length=50)

# get colors 
cmap = plt.get_cmap(seq_cmap)
norm = mpl.colors.Normalize(vmin=1, vmax=length(wls_interest))
colors = cmap(norm(1:length(wls_interest)))

# loop over ftemps
for j in eachindex(ftemps)
    # make figure objects
    fig, ax1 = plt.subplots(figsize=(9.2,4.8))

    # horizontal line
    ax1.axhline(ftemps[j], ls="--", c="k", alpha=0.9)

    # iterate over lines
    the_xticks = zeros(length(idx_wls))
    for i in eachindex(idx_wls)
        # get the index of the minimum
        idx_min = argmin(temp_list[i])

        # get the index of the temperature for each line
        tdiffs = abs.(temp_list[i][idx_min+1:end] .- ftemps[j])
        this_idx = argmin(tdiffs) .+ idx_min

        # plot the lines
        ax1.plot(wavs_list[i], temp_list[i], zorder=0, c=ncolors[i])

        # get an offset 
        offset = offset_scale * (i - 1)
        the_xticks[i] = offset

        # continue if broke
        isnothing(this_idx) && continue

        # get data for scatter
        xscatter = [wavs_list[i][this_idx]] #.+ offset
        yscatter = [temp_list[i][this_idx]]
        ax1.scatter(xscatter, yscatter, c="k", zorder=1)
    end
    ax1.set_xticks(the_xticks)
    ax1.set_xticklabels(specs_interest_latex, rotation=45, ha="right")

    # ax1.set_xlabel(L"{\rm Air\ Wavelength\ +\ Offset\ [\AA]}")
    ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
    fig.tight_layout()
    fig.savefig(joinpath(framedir, "line_lineup_$j.png"), bbox_inches="tight")
    plt.clf(); plt.close()

    # now do each contribution slice
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9.2, 4.8), sharex=true)

    # get exponent for units
    max_val = maximum(abs.(vcat(cfunc_list...)))
    exponent = floor(Int, log10(max_val))
    the_ymin = 0.0
    the_ymax = max_val / 10^exponent + 0.5 

    for i in eachindex(idx_wls)
        # get the index of the minimum
        idx_min = argmin(temp_list[i])

        # get the index of the temperature for each line
        tdiffs = abs.(temp_list[i][idx_min+1:end] .- ftemps[j])
        this_idx = argmin(tdiffs) .+ idx_min

        # get views of cfuncs at indices of interest
        cfuncs_sim = cfunc_list[i][:, this_idx] # view(cfunc_flux, :, idx)
        cfuncs_cum_sim = cfunc_cum_list[i][:, this_idx] # view(cum_cfunc_flux_norm, :, idx)

        ax1.plot(elav(Ts), cfuncs_sim / 10^exponent, c=ncolors[i])
        ax2.plot(elav(Ts), cfuncs_cum_sim, c=ncolors[i], label=specs_interest_latex[i])
        ax1.axvline(ftemps[j], ls="--", c="k", alpha=0.9)
        ax2.axvline(ftemps[j], ls="--", c="k", alpha=0.9)
    end

    ax1.set_xlabel(L"{\rm Temperature\ [K]}")
    ax2.set_xlabel(L"{\rm Temperature\ [K]}")

    ax1.set_ylabel(L"\mathscr{C}_{\nu}(t_\nu)\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1}]}")
    ax2.set_ylabel(L"{\rm Normalized\ Cumulative\ Flux\ Cont.\ Fn.}")
    ax2.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    ax1.set_ylim(the_ymin, the_ymax)

    fig.subplots_adjust(wspace=0.25)
    fig.savefig(joinpath(framedir, "cont_comparison_$j.png"), bbox_inches="tight")
    plt.clf(); plt.close()
end