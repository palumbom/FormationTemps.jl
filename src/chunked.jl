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
                                 callback=nothing, kwargs...)

Compute formation temperatures over an arbitrarily large wavelength range by
processing fixed-width wavelength chunks.

Each chunk selects lines within `[λ_start - wing_padding, λ_stop + wing_padding]`,
ensuring that broad line wings contribute correctly at chunk boundaries.

# Arguments
- `chunk_width`: width of each wavelength chunk in Å.
- `wing_padding`: extra range (Å) beyond each chunk edge for linelist selection.
  30 Å is conservative enough for the strongest lines (H-alpha, Ca II).
- `overlap`: overlap width (Å) between adjacent chunks for smooth stitching.
- `callback`: optional `(chunk_idx, result::FormTempResult, ll_chunk) -> nothing`
  called after each chunk, where `ll_chunk` is the padded linelist view used for
  that chunk.  When provided, results are not accumulated and the function returns
  `nothing`.  Use this for streaming to disk when the full result is too large for RAM.
- All other keyword arguments are forwarded to [`calc_formation_temp`](@ref).

When `callback=nothing`, returns a `Vector{FormTempResult}` with one entry per
chunk.  Stitching is the caller's responsibility.
"""
function calc_formation_temp_chunked(star::StellarProps, linelist;
                                      chunk_width::T=50.0,
                                      wing_padding::T=30.0,
                                      overlap::T=5.0,
                                      Δλ::T=0.001,
                                      buffer::T=2.0,
                                      callback=nothing,
                                      showprogress::Bool=true,
                                      kwargs...) where T<:AF
    @assert chunk_width > overlap "chunk_width must exceed overlap"

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
                                     showprogress=false,
                                     kwargs...)

        if use_callback
            callback(ci, result, ll_chunk)
        else
            push!(chunk_results, result)
        end
        next!(prog)
    end

    use_callback && return nothing
    return chunk_results
end
