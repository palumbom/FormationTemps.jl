"""
Wavelength-window chunked computation of formation temperatures.
"""

"""
    _select_lines_for_window(linelist, λ_start, λ_stop, wing_padding)

Return a `view` of `linelist` containing lines whose centers (in Å) fall within
`[λ_start - wing_padding, λ_stop + wing_padding]`.  Uses binary search on the
sorted linelist for O(log N) per chunk.
"""
function _select_lines_for_window(linelist, wl_cm::Vector{Float64},
                                   λ_start::T, λ_stop::T, wing_padding::T) where T<:AF
    wl_lo = (λ_start - wing_padding) * ANGSTROM_TO_CM
    wl_hi = (λ_stop  + wing_padding) * ANGSTROM_TO_CM
    i_lo = searchsortedfirst(wl_cm, wl_lo)
    i_hi = searchsortedlast(wl_cm,  wl_hi)
    return view(linelist, i_lo:i_hi)
end

"""
    calc_formation_temp_chunked(star, linelist; chunk_width=50.0, wing_padding=30.0,
                                 overlap=5.0, Δλ=0.001, buffer=2.0,
                                 stitch_mode=:midpoint, callback=nothing, kwargs...)

Compute formation temperatures over an arbitrarily large wavelength range by
processing fixed-width wavelength chunks.

Each chunk selects lines within `[λ_start - wing_padding, λ_stop + wing_padding]`,
ensuring that broad line wings contribute correctly at chunk boundaries.

# Arguments
- `chunk_width`: width of each wavelength chunk in Å.
- `wing_padding`: extra range (Å) beyond each chunk edge for linelist selection.
  30 Å is conservative enough for the strongest lines (H-alpha, Ca II).
- `overlap`: overlap width (Å) between adjacent chunks for smooth stitching.
- `stitch_mode`: `:midpoint` (cut at midpoint, default) or `:blend` (linear blend
  in overlap region).
- `callback`: optional `(chunk_idx, result::FormTempResult, ll_chunk) -> nothing`
  called after each chunk, where `ll_chunk` is the padded linelist view used for
  that chunk.  When provided, results are not accumulated and the function returns
  `nothing`.  Use this for streaming to disk when the full result is too large for RAM.
- All other keyword arguments are forwarded to [`calc_formation_temp`](@ref).

When `callback=nothing`, returns a single [`FormTempResult`](@ref) covering the
full wavelength range.
"""
function calc_formation_temp_chunked(star::StellarProps, linelist;
                                      chunk_width::T=50.0,
                                      wing_padding::T=30.0,
                                      overlap::T=5.0,
                                      Δλ::T=0.001,
                                      buffer::T=2.0,
                                      stitch_mode::Symbol=:midpoint,
                                      callback=nothing,
                                      showprogress::Bool=true,
                                      kwargs...) where T<:AF
    @assert chunk_width > overlap "chunk_width must exceed overlap"
    @assert stitch_mode in (:midpoint, :blend) "stitch_mode must be :midpoint or :blend"

    # extract wavelengths once (cm for binary search, Å for range computation)
    wl_cm = Float64[l.wl for l in linelist]
    wls_all = wl_cm .* CM_TO_ANGSTROM
    full_min = first(wls_all) - buffer
    full_max = last(wls_all) + buffer

    # round up full_max so all chunks have the same Nλ
    step = chunk_width - overlap
    n_chunks = ceil(Int, (full_max - full_min - overlap) / step)
    full_max = full_min + overlap + n_chunks * step

    # chunk start positions
    chunk_starts = [full_min + (i - 1) * step for i in 1:n_chunks]

    # accumulate per-chunk results when no callback
    use_callback = callback !== nothing
    chunk_results = use_callback ? nothing : FormTempResult[]

    prog = Progress(n_chunks; desc="Chunks: ", enabled=showprogress)
    for (ci, λ_start) in enumerate(chunk_starts)
        λ_stop = λ_start + chunk_width

        # select lines with padded wings
        ll_chunk = _select_lines_for_window(linelist, wl_cm, λ_start, λ_stop, wing_padding)

        # pass wing_padding as the continuum line_buffer so that compute_alpha!'s
        # interpolation grid covers all included line centers, without extending
        # the GPU wavelength grid (which would waste GPU memory)
        result = calc_formation_temp(star, ll_chunk; Δλ=Δλ,
                                     minλ=λ_start, maxλ=λ_stop,
                                     buffer=zero(T),
                                     line_buffer_Å=wing_padding + T(10),
                                     showprogress=false, # hardcoded for inner progress bar
                                     kwargs...)

        if use_callback
            callback(ci, result, ll_chunk)
        else
            push!(chunk_results, result)
        end
        next!(prog)
    end

    # callback mode: nothing to stitch
    use_callback && return nothing

    # single chunk: return directly
    n_chunks == 1 && return chunk_results[1]

    # stitch chunks together
    return _stitch_chunks(chunk_results, overlap, stitch_mode)
end


"""
    _stitch_chunks(results, overlap, mode)

Combine a vector of per-chunk `FormTempResult`s into a single result,
handling the overlap regions according to `mode`.
"""
function _stitch_chunks(results, overlap, mode::Symbol)
    if mode == :midpoint
        return _stitch_midpoint(results)
    else
        return _stitch_blend(results, overlap)
    end
end

function _stitch_midpoint(results)
    n = length(results)
    T = eltype(results[1].wavs)
    centers = [T(0.5) * (first(r.wavs) + last(r.wavs)) for r in results]

    all_wavs = T[]
    all_flux = T[]
    all_temps = T[]
    all_cfunc_cols = Vector{T}[]

    for i in 1:n
        r = results[i]
        left_bound  = i == 1 ? T(-Inf) : T(0.5) * (centers[i - 1] + centers[i])
        right_bound = i == n ? T(Inf)  : T(0.5) * (centers[i] + centers[i + 1])
        keep = (r.wavs .>= left_bound) .& (r.wavs .< right_bound)

        append!(all_wavs, r.wavs[keep])
        append!(all_flux, r.flux[keep])
        append!(all_temps, r.form_temps[keep])
        for j in findall(keep)
            push!(all_cfunc_cols, r.cont_func[:, j])
        end
    end

    cfunc_out = reduce(hcat, all_cfunc_cols)
    return FormTempResult(all_wavs, all_flux, all_temps, cfunc_out, results[1].atmosphere)
end

function _stitch_blend(results, overlap)
    n = length(results)
    T = eltype(results[1].wavs)

    # infer Δλ from the first chunk's wavelength grid
    # +1 for inclusive count: overlap of 2.0 Å at Δλ=0.01 spans 201 points
    Δλ = results[1].wavs[2] - results[1].wavs[1]
    N_ov = max(0, round(Int, T(overlap) / Δλ) + 1)

    # start with the first chunk
    all_wavs  = collect(results[1].wavs)
    all_flux  = collect(results[1].flux)
    all_temps = collect(results[1].form_temps)
    all_cfunc = collect(results[1].cont_func)  # (Natm1, Nλ)

    for i in 2:n
        r = results[i]
        Nλ_new = length(r.wavs)

        # the first N_ov points of this chunk overlap the last N_ov points of accumulated data
        N_ov_actual = min(N_ov, length(all_wavs), Nλ_new)

        # blend the overlap region with linear weights
        for k in 1:N_ov_actual
            w_new = T(k) / T(N_ov_actual + 1)
            w_old = one(T) - w_new
            acc_idx = length(all_wavs) - N_ov_actual + k
            all_flux[acc_idx]  = w_old * all_flux[acc_idx]  + w_new * r.flux[k]
            all_temps[acc_idx] = w_old * all_temps[acc_idx] + w_new * r.form_temps[k]
            all_cfunc[:, acc_idx] .= w_old .* all_cfunc[:, acc_idx] .+ w_new .* r.cont_func[:, k]
        end

        # append the non-overlapping tail
        if N_ov_actual < Nλ_new
            tail = (N_ov_actual + 1):Nλ_new
            append!(all_wavs, r.wavs[tail])
            append!(all_flux, r.flux[tail])
            append!(all_temps, r.form_temps[tail])
            all_cfunc = hcat(all_cfunc, r.cont_func[:, tail])
        end
    end

    return FormTempResult(all_wavs, all_flux, all_temps, all_cfunc, results[1].atmosphere)
end
