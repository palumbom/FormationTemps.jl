using Revise
using FormationTemps; FT = FormationTemps
using Korg
using DSP: conv
using Random
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using ProgressMeter

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pyconvert
using LaTeXStrings
mpl = plt.matplotlib

# matplotlib backend
mpl.use("QtAgg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# animation
animation = pyimport("matplotlib.animation");
pe = pyimport("matplotlib.patheffects");

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

# minima of absorption lines in f(λ). irregular λ ok.
# returns (indices, subpixel λ via 3-pt quadratic)
function find_line_minima(λ::AbstractVector, f::AbstractVector;
                          σ=2.0, depth=0.01, win=25, minsep=5)
    N = length(f)
    @assert length(λ) == N && N ≥ 5

    g = collect(f)
    if σ > 0
        half = max(2, Int(ceil(3σ)))
        x = -half:half
        k = exp.(-(x.^2)./(2σ^2)); k ./= sum(k)
        gp = vcat(fill(g[1], half), g, fill(g[end], half))
        conv_full = conv(gp, k)
        g = collect(@view conv_full[2half+1:2half+N])  # same length
    end

    d1 = similar(g); d2 = similar(g)
    for i in 2:N-1
        dλm = λ[i]-λ[i-1]
        dλp = λ[i+1]-λ[i]
        s1 = (g[i]-g[i-1])/dλm
        s2 = (g[i+1]-g[i])/dλp
        d1[i] = (s1*dλp + s2*dλm)/(dλm+dλp)  # weighted slope
        a = (s2 - s1)/((dλm+dλp)/2)
        d2[i] = 2a
    end
    d1[1] = (g[2]-g[1])/(λ[2]-λ[1])
    d1[end] = (g[end]-g[end-1])/(λ[end]-λ[end-1])
    d2[1] = d2[2]; d2[end] = d2[end-1]

    cand = Int[]
    for i in 2:N-1
        if d1[i-1] < 0 && d1[i+1] > 0 && d2[i] > 0
            push!(cand, i)
        end
    end

    keep = Int[]
    for i in cand
        lo = max(1, i-win); hi = min(N, i+win)
        localmax = maximum(@view g[lo:hi])
        if localmax - g[i] ≥ depth*localmax
            push!(keep, i)
        end
    end

    sort!(keep, by=i->g[i])         # deeper first
    chosen = Int[]; taken = falses(N)
    for i in keep
        lo = max(1, i-minsep); hi = min(N, i+minsep)
        if !any(taken[lo:hi])
            push!(chosen, i); taken[i] = true
        end
    end
    sort!(chosen)

    λhat = similar(λ, length(chosen))
    for (j,i) in pairs(chosen)
        if 2 ≤ i ≤ N-1
            x1,x2,x3 = λ[i-1], λ[i], λ[i+1]
            y1,y2,y3 = g[i-1], g[i], g[i+1]
            denom = (x1-x2)*(x1-x3)*(x2-x3)
            if denom == 0
                λhat[j] = λ[i]
            else
                a = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2))/denom
                b = (x3^2*(y1-y2) + x2^2*(y3-y1) + x1^2*(y2-y3))/denom
                λhat[j] = a != 0 ? -b/(2a) : λ[i]
            end
        else
            λhat[j] = λ[i]
        end
    end

    return chosen, λhat
end


# read pre-computed 1D formation temperature spectrum
cephdir = abspath("/mnt/home/mpalumbo/ceph/")
outfile_1d = joinpath(cephdir, "formation_temps", "temp_spectrum_air_1D.h5")
λ_min = 5500.0  # Å
λ_max = 6400.0  # Å

λs_korg = Float64[]
flux = Float64[]
form_temps_flux = Float64[]
cfunc_chunks = Matrix{Float64}[]
line_centers_all = Float64[]

h5open(outfile_1d, "r") do h5
    Ts = read(h5["model_atmosphere"]["Ts"])

    chunk_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5))))
    for cn in chunk_names
        g = h5[cn]
        wavs_chunk = read(g["wavs"])

        # skip chunks entirely outside the wavelength range
        (last(wavs_chunk) < λ_min || first(wavs_chunk) > λ_max) && continue

        keep = (wavs_chunk .>= λ_min) .& (wavs_chunk .<= λ_max)
        append!(λs_korg, wavs_chunk[keep])
        append!(flux, read(g["flux"])[keep])
        append!(form_temps_flux, read(g["temp"])[keep])
        push!(cfunc_chunks, read(g["cfunc"])[:, keep])
        append!(line_centers_all, read(g["line_centers"]))
    end

    global Ts = Ts
end
cfunc_flux = hcat(cfunc_chunks...)
@info "read $(length(λs_korg)) pixels in [$(λ_min), $(λ_max)] Å"

# build cumulative contribution function (normalized per wavelength)
cum_cfunc_flux_norm = cumsum(cfunc_flux, dims=1)
cum_cfunc_flux_norm ./= maximum(cum_cfunc_flux_norm, dims=1)

# read linelist for species labels (filter to same range)
linelist = Korg.read_linelist(joinpath(cephdir, "formation_temps", "Sun_VALD_BIG.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
wls = [l.wl * 1e8 for l in linelist]
in_range = (wls .>= λ_min) .& (wls .<= λ_max)
linelist = linelist[in_range]
wls = wls[in_range]
species = [l.species for l in linelist]
E_lower = [l.E_lower for l in linelist]

# target formation temperature and tolerance
ftemp = 4800.0
atol = 25.0
isolated_only = true
min_sep = 0.5  # minimum separation in Å between T₁/₂ minima (not raw linelist)

all_idx = findall(isapprox.(ftemp, form_temps_flux, atol=atol))

# find line minima in the T₁/₂ curve
# scale win/minsep to the actual pixel size (parameters were tuned for Δλ≈0.01)
Δλ_actual = mean(diff(λs_korg))
pix_per_angstrom = round(Int, 1.0 / Δλ_actual)
min_idx, λsub = find_line_minima(λs_korg, form_temps_flux;
                                  σ=2.0,
                                  depth=0.02,
                                  win=max(40, round(Int, 0.5 * pix_per_angstrom)),
                                  minsep=max(7, round(Int, 0.1 * pix_per_angstrom)))
@info "found $(length(min_idx)) T₁/₂ minima (Δλ=$(round(Δλ_actual, digits=4)) Å, $(pix_per_angstrom) pix/Å)"

# group candidate pixels by their nearest line minimum
line_to_pixels = Dict{Int, Vector{Int}}()
for i in all_idx
    li = min_idx[FT.searchsortednearest(i, min_idx)]
    push!(get!(line_to_pixels, li, Int[]), i)
end
@info "$(length(line_to_pixels)) unique minima have candidate pixels nearby"

# for each unique line, pick the red-wing pixel closest to ftemp
line_ids = sort(collect(keys(line_to_pixels)))
best_pixel = Dict{Int, Int}()
global n_no_red = 0
for li in line_ids
    pixels = line_to_pixels[li]
    red_pixels = filter(p -> p > li, pixels)
    if isempty(red_pixels)
        delete!(line_to_pixels, li)
        global n_no_red += 1
        continue
    end
    _, bi = findmin(abs.(form_temps_flux[red_pixels] .- ftemp))
    best_pixel[li] = red_pixels[bi]
end
filter!(li -> haskey(best_pixel, li), line_ids)
@info "$(length(line_ids)) lines with red-wing pixels ($n_no_red dropped for no red-wing match)"

# filter to minima that are well separated from their neighbors in the T₁/₂ curve
if isolated_only
    min_sep_pix = round(Int, min_sep * pix_per_angstrom)
    min_λs = λs_korg[min_idx]  # wavelengths of all detected minima
    n_before_iso = length(line_ids)
    filter!(li -> begin
        # find this minimum's position among all minima
        mi = findfirst(==(li), min_idx)
        isnothing(mi) && return false
        left_ok = mi == 1 || (min_idx[mi] - min_idx[mi-1]) >= min_sep_pix
        right_ok = mi == length(min_idx) || (min_idx[mi+1] - min_idx[mi]) >= min_sep_pix
        left_ok && right_ok
    end, line_ids)
    best_pixel = Dict(li => best_pixel[li] for li in line_ids)
    @info "$(length(line_ids)) / $n_before_iso isolated candidate lines (min_sep=$(min_sep) Å)"
end

# filter to lines with E_lower > 0 (exclude ground-state resonance lines)
filter!(li -> begin
    nearest_ll = FT.searchsortednearest(wls, λs_korg[li])
    E_lower[nearest_ll] > 0
end, line_ids)
best_pixel = Dict(li => best_pixel[li] for li in line_ids)
@info "$(length(line_ids)) candidate lines after E_lower > 0 filter"

# filter out lines whose cfunc peaks at the top of the atmosphere (truncated by model boundary)
edge_thresh = 0.1  # max allowed cfunc at top layer, as fraction of peak
n_before_edge = length(line_ids)
filter!(li -> begin
    col = cfunc_flux[:, best_pixel[li]]
    peak = maximum(col)
    peak > 0 && col[1] / peak < edge_thresh
end, line_ids)
best_pixel = Dict(li => best_pixel[li] for li in line_ids)
@info "$(length(line_ids)) / $n_before_edge lines after top-of-atmosphere cfunc filter"

# extract cfunc at each line's best pixel, normalize to unit area for shape comparison
line_cfuncs = Dict{Int, Vector{Float64}}()
for li in line_ids
    col = cfunc_flux[:, best_pixel[li]]
    s = sum(col)
    line_cfuncs[li] = s > 0 ? col ./ s : col
end

# farthest-point sampling: select k lines with maximally diverse cfunc shapes
function select_diverse(line_ids, line_cfuncs, k)
    n = length(line_ids)
    k = min(k, n)

    # pairwise L2 distance
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        d = sqrt(sum((line_cfuncs[line_ids[i]] .- line_cfuncs[line_ids[j]]).^2))
        D[i,j] = d
        D[j,i] = d
    end

    # seed with the most distant pair
    _, ij = findmax(D)
    chosen = [ij[1], ij[2]]

    while length(chosen) < k
        best_ci = 0
        best_d = -Inf
        for c in 1:n
            c in chosen && continue
            min_d = minimum(D[c, s] for s in chosen)
            if min_d > best_d
                best_d = min_d
                best_ci = c
            end
        end
        push!(chosen, best_ci)
    end
    return sort(chosen)
end

nlines = 10
@info "$(length(all_idx)) pixels within atol=$atol of ftemp=$ftemp; $(length(line_ids)) unique lines"
diverse_ci = select_diverse(line_ids, line_cfuncs, nlines)
selected_line_ids = [line_ids[ci] for ci in diverse_ci]

# map selected lines back to linelist entries
idx_wls = [FT.searchsortednearest(wls, λs_korg[li]) for li in selected_line_ids]
wls_interest = wls[idx_wls]

# format species names
specs_interest = string.(species[idx_wls])
specs_interest_latex = latexstring.(specs_interest)
for i in eachindex(specs_interest)
    parts = split(specs_interest[i])
    wl_str = string(round(wls_interest[i], digits=1))
    spec_str = join(parts, "\\, ")
    specs_interest_latex[i] = L"{\rm %$spec_str\, %$wl_str\, \AA}"
end

# extract windows around each selected line
wavs_list = []
flux_list = []
temp_list = []
cfunc_list = []
cfunc_cum_list = []
best_local_idx = Int[]  # local index of the pixel that matched ftemp in each window

# buffer extends until dT/dλ flattens on both sides of the line
# smooth the T₁/₂ curve to avoid noise triggering early flattening
smooth_win = ceil(Int, 0.05 / mean(diff(λs_korg)))  # ~0.05 Å smoothing
dT_global = diff(form_temps_flux)
flat_thresh = 0.3  # K per pixel — derivative below this counts as flat
min_buffer = ceil(Int, 0.3 / mean(diff(λs_korg)))
max_buffer = ceil(Int, 3.0 / mean(diff(λs_korg)))  # hard cap at ±3 Å
gap = 0.08  # Å gap between adjacent cutouts
Nλ_total = length(λs_korg)
Δλ_mean = mean(diff(λs_korg))

# first pass: compute each line's buffer
buffers = Int[]
for i in eachindex(selected_line_ids)
    idx_λs = findfirst(x -> x >= wls_interest[i], λs_korg)
    bp = best_pixel[selected_line_ids[i]]

    # walk redward: stop when derivative flattens OR turns negative (entering next line)
    red_buf = min_buffer
    for k in min_buffer:min(max_buffer, Nλ_total - idx_λs - 1)
        lo = max(1, idx_λs + k - smooth_win)
        hi = min(length(dT_global), idx_λs + k + smooth_win)
        avg_dT = mean(view(dT_global, lo:hi))
        if avg_dT < flat_thresh  # flat or descending into next line
            red_buf = k
            break
        end
    end
    # walk blueward: stop when derivative flattens OR turns positive (entering next line)
    # note: blueward walk goes left, so rising wing has negative dT/dλ
    blue_buf = min_buffer
    for k in min_buffer:min(max_buffer, idx_λs - 2)
        lo = max(1, idx_λs - k - smooth_win)
        hi = min(length(dT_global), idx_λs - k + smooth_win)
        avg_dT = mean(view(dT_global, lo:hi))
        if avg_dT > -flat_thresh  # flat or descending into next line
            blue_buf = k
            break
        end
    end

    buffer = max(min_buffer, red_buf, blue_buf, ceil(Int, 1.2 * abs(bp - idx_λs)))
    buffer = min(buffer, idx_λs - 1, Nλ_total - idx_λs)
    push!(buffers, buffer)
end

# compute cumulative offsets so each cutout is placed after the previous one's right edge
offsets = zeros(length(selected_line_ids))
for i in 2:length(selected_line_ids)
    prev_half = buffers[i-1] * Δλ_mean  # right half of previous window
    curr_half = buffers[i] * Δλ_mean    # left half of current window
    offsets[i] = offsets[i-1] + prev_half + curr_half + gap
end

# second pass: extract windows using computed buffers and offsets
for i in eachindex(selected_line_ids)
    idx_λs = findfirst(x -> x >= wls_interest[i], λs_korg)
    buffer = buffers[i]

    win_start = idx_λs - buffer
    λs_view = view(λs_korg, win_start:idx_λs+buffer) .- wls_interest[i] .+ offsets[i]
    flux_view = view(flux, win_start:idx_λs+buffer)
    temp_view = view(form_temps_flux, win_start:idx_λs+buffer)
    cfunc_view = view(cfunc_flux, :, win_start:idx_λs+buffer)
    cfunc_cum_view = view(cum_cfunc_flux_norm, :, win_start:idx_λs+buffer)

    push!(wavs_list, collect(λs_view))
    push!(flux_list, collect(flux_view))
    push!(temp_list, collect(temp_view))
    push!(cfunc_list, collect(cfunc_view))
    push!(cfunc_cum_list, collect(cfunc_cum_view))
    push!(best_local_idx, best_pixel[selected_line_ids[i]] - win_start + 1)
end

# find temperatures to loop over
min_temp = maximum(minimum.(temp_list))
max_temp = minimum(maximum.(temp_list))
ftemps = range(ceil(min_temp+1), floor(max_temp-1), length=50)

# get colors
cmap = plt.get_cmap(seq_cmap)
norm = mpl.colors.Normalize(vmin=1, vmax=length(wls_interest))
colors = cmap(norm(1:length(wls_interest)))

# find the pixel on the wings (non-negative dT/dλ) closest to a target temperature
function find_nearest_pixel(temps, target)
    N = length(temps)
    dT = diff(temps)
    best_idx = 0
    best_diff = Inf
    for i in 1:N-1
        dT[i] >= 0 || continue
        d = abs(temps[i] - target)
        if d < best_diff
            best_diff = d
            best_idx = i
        end
    end
    # fallback: if no non-negative-derivative pixel found, use global closest
    return best_idx == 0 ? argmin(abs.(temps .- target)) : best_idx
end

# temperature at which to save the PDF
save_temp = 5000.0
save_atol = 10.0

# loop over ftemps
for j in eachindex(ftemps)
    is_save = abs(ftemps[j] - save_temp) <= save_atol

    # make figure objects
    fig, ax1 = plt.subplots(figsize=(9.2,4.8))

    # horizontal line
    ax1.axhline(ftemps[j], ls="--", c="k", alpha=0.9)

    # iterate over lines
    the_xticks = zeros(length(idx_wls))
    for i in eachindex(idx_wls)
        # plot the lines
        ax1.plot(wavs_list[i], temp_list[i], zorder=0, c=ncolors[i])
        the_xticks[i] = offsets[i]

        this_idx = find_nearest_pixel(temp_list[i], ftemps[j])
        if !is_save && abs(temp_list[i][this_idx] - ftemps[j]) > atol
            continue
        end

        ax1.scatter([wavs_list[i][this_idx]], [temp_list[i][this_idx]], c="k", zorder=1)
    end
    ax1.set_xticks(the_xticks)
    ax1.set_xticklabels(specs_interest_latex, rotation=45, ha="right")

    ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
    fig.tight_layout()
    fig.savefig(joinpath(framedir, "line_lineup_$j.png"), bbox_inches="tight")
    if is_save
        fig.savefig("figures/line_lineup.pdf", bbox_inches="tight")
    end
    plt.clf(); plt.close()

    # now do each contribution slice
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.2, 4.8), sharex=true,
                             gridspec_kw=Dict("width_ratios" => [1, 1]))
    ax1 = axes[0]
    ax2 = axes[1]

    # get exponent for units
    max_val = maximum(maximum(abs.(c)) for c in cfunc_list)
    exponent = floor(Int, log10(max_val))
    the_ymin = 0.0
    the_ymax = max_val / 10^exponent + 0.5
    frame_ymax = 0.0

    for i in eachindex(idx_wls)
        this_idx = find_nearest_pixel(temp_list[i], ftemps[j])
        if !is_save && abs(temp_list[i][this_idx] - ftemps[j]) > atol
            continue
        end

        # get views of cfuncs at indices of interest
        cfuncs_sim = cfunc_list[i][:, this_idx]
        cfuncs_cum_sim = cfunc_cum_list[i][:, this_idx]

        frame_ymax = max(frame_ymax, maximum(cfuncs_sim) / 10^exponent)

        ax1.plot(elav(Ts), cfuncs_sim / 10^exponent, c=ncolors[i])
        ax2.plot(elav(Ts), cfuncs_cum_sim, c=ncolors[i], label=specs_interest_latex[i])
        ax1.axvline(ftemps[j], ls="--", c="k", alpha=0.9)
        ax2.axvline(ftemps[j], ls="--", c="k", alpha=0.9)
    end

    ax1.set_xlabel(L"{\rm Temperature\ [K]}")
    ax2.set_xlabel(L"{\rm Temperature\ [K]}")

    ax1.set_ylabel(L"\mathscr{C}_{\nu}(t_\nu)\ dt_\nu\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-4} \ \AA\ ^{-1}]}")
    ax2.set_ylabel(L"{\rm Normalized\ Cumulative\ Flux\ Cont.\ Fn.}")
    ax2.legend(loc="lower right", fontsize="small")

    # fixed ylim for animation frames; tight ylim for saved PDF
    ax1.set_ylim(the_ymin, the_ymax)

    fig.tight_layout()
    fig.savefig(joinpath(framedir, "cont_comparison_$j.png"), bbox_inches="tight")
    if is_save
        @show ftemps[j]
        ax1.set_ylim(the_ymin, frame_ymax * 1.05)
        fig.savefig("figures/cont_comparison.pdf", bbox_inches="tight")
    end
    plt.clf(); plt.close()
end
