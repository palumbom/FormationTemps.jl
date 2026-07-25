function linear_interp(xs::AA{T,1}, ys::AA{T,1}; bc::T=T(NaN)) where T<:AF
    function f(x)
        if (((x < first(xs)) | (x > last(xs))) & !isnan(bc))
            return bc
        elseif x <= first(xs)
            return first(ys)
        elseif x >= last(xs)
            return last(ys)
        else
            i = searchsortedfirst(xs, x) - 1
            i0 = clamp(i, firstindex(ys), lastindex(ys))
            i1 = clamp(i+1, firstindex(ys), lastindex(ys))
            return (ys[i0] * (xs[i1] - x) + ys[i1] * (x - xs[i0])) / (xs[i1] - xs[i0])
        end
    end
    return f
end

function linear_interp_gpu(xs::AA{T,1}, ys::AA{T,1}) where T<:AF
    function f(x)
        if x <= CUDA.first(xs)
            return CUDA.first(ys)
        elseif x >= CUDA.last(xs)
            return CUDA.last(ys)
        else
            i = CUDA.searchsortedfirst(xs, x) - 1
            i0 = CUDA.clamp(i, CUDA.firstindex(ys), CUDA.lastindex(ys))
            i1 = CUDA.clamp(i+1, CUDA.firstindex(ys), CUDA.lastindex(ys))
            return (ys[i0] * (xs[i1] - x) + ys[i1] * (x - xs[i0])) / (xs[i1] - xs[i0])
        end
    end
    return f
end

# Default flag threshold for top-of-atmosphere contamination, as a fraction of the column
# peak. The statistic is strongly bimodal — near 0 for lines whose contribution decays inside
# the model, near 1 for cores still rising at the truncated ceiling — so the exact value
# matters little.
const BOUNDARY_R_THRESH = 0.33

"""
    ceiling_ratio(cfunc_dt) -> Vector
    ceiling_ratio(result::FormTempResult) -> Vector

Per-wavelength top-of-atmosphere contamination statistic: the topmost interval's flux
contribution as a fraction of that column's peak contribution.

Near 0 when a line's contribution function has decayed well inside the model atmosphere, and
near 1 when it is still rising at the truncated ceiling — in which case the formation
temperature is biased toward the top boundary and should be read as a lower limit. Threshold
it with [`boundary_mask`](@ref). Returned as a field of [`FormTempResult`](@ref).

An all-zero column (no contribution at all) yields 0, not `NaN`; those wavelengths are
reported separately by [`form_temps_from_cfunc`](@ref), which sets them to `NaN`.
"""
function ceiling_ratio(cfunc_dt::AA{T,2}) where T<:AF
    peak = vec(maximum(cfunc_dt, dims=1))
    return vec(cfunc_dt[1, :]) ./ max.(peak, eps(T))
end

ceiling_ratio(result::FormTempResult) = result.ceiling_ratio

"""
    boundary_mask(cfunc_dt; r_thresh=BOUNDARY_R_THRESH) -> BitVector
    boundary_mask(result::FormTempResult; r_thresh=result.r_thresh) -> BitVector

Flag wavelengths whose formation temperature is contaminated by the top of the model
atmosphere, i.e. where [`ceiling_ratio`](@ref) exceeds `r_thresh`.

For a `FormTempResult` the threshold defaults to the one the calculation used, so the mask
reproduces exactly the wavelengths it warned about. Pass `r_thresh` explicitly to re-threshold
after the fact.
"""
boundary_mask(cfunc_dt::AA{<:AF,2}; r_thresh::Real=BOUNDARY_R_THRESH) =
    ceiling_ratio(cfunc_dt) .> r_thresh

boundary_mask(result::FormTempResult; r_thresh::Real=result.r_thresh) =
    result.ceiling_ratio .> r_thresh

"""
    form_temps_from_cfunc(cfunc_dt, Ts; r_thresh=BOUNDARY_R_THRESH, warn_boundary=true) -> Vector

Formation temperature at 50% of the cumulative flux contribution.

Uses a node-anchored cumulative: `F = 0` at the top node `Ts[1]`, and the
contribution accumulated through interval `k` is assigned to the deep node
`Ts[k+1]`; the 50% crossing is interpolated against the node temperatures `Ts`.
This avoids the half-interval cool bias of pairing the cumulative (a cell-edge
quantity) with interval-center temperatures.

`cfunc_dt` is `(Natm-1, Nλ)` (per-interval flux contribution); `Ts` is `(Natm,)`
node temperatures. CPU; GPU callers pass host copies.

Two degenerate outcomes are warned about rather than returned silently:

- **Non-positive column total.** No median exists, so the formation temperature is `NaN`.
  Reachable when an upstream microturbulence kernel underflows and zeros a column.
- **Boundary contamination.** Where [`ceiling_ratio`](@ref) exceeds `r_thresh` the flux
  contribution is still peaking at the top of the model, so the formation temperature is
  biased toward `Ts[1]` and should be read as a lower limit. The warning counts exactly the
  wavelengths [`boundary_mask`](@ref) flags at the same `r_thresh`. `warn_boundary=false`
  silences it.
"""
function form_temps_from_cfunc(cfunc_dt::AA{T,2}, Ts::AA{T,1};
                               r_thresh::Real=BOUNDARY_R_THRESH,
                               warn_boundary::Bool=true) where T<:AF
    Natm = length(Ts)
    @assert size(cfunc_dt, 1) == Natm - 1 "cfunc_dt must have Natm-1 rows to match Ts"
    Nλ = size(cfunc_dt, 2)
    cum = cumsum(cfunc_dt, dims=1)

    # normalize by the column total (definition: 50% of total contribution; == maximum only
    # if monotonic). cum[end:end, :] is a copy, so the in-place divide reads a fixed
    # divisor. Guard non-positive totals: zero gives 0/0, and a (roundoff-only) negative
    # total inverts the CDF, which `linear_interp` would silently resolve to the *deepest*
    # temperature via its x >= last(xs) branch. Both are flagged as NaN below instead.
    totals = cum[end:end, :]
    empty_col = .!(totals .> zero(T))
    n_empty = count(empty_col)
    cum ./= ifelse.(empty_col, one(T), totals)

    form   = zeros(T, Nλ)
    Fnodes = zeros(T, Natm)
    @inbounds for i in 1:Nλ
        if empty_col[1, i]
            form[i] = T(NaN)
            continue
        end
        Fnodes[1] = zero(T)
        @views Fnodes[2:end] .= cum[:, i]     # cum[k] ↦ deep node Ts[k+1]
        form[i] = linear_interp(Fnodes, Ts)(T(0.5))
    end

    if n_empty > 0
        @warn "form_temps_from_cfunc: $n_empty of $Nλ wavelengths have no positive total " *
              "flux contribution; their formation temperatures are NaN. Usually an upstream " *
              "microturbulence kernel underflow (check velocity units, m/s)." maxlog=3
    end
    # same statistic and threshold as boundary_mask, so the warning cannot flag a different
    # set of wavelengths than the mask a caller applies downstream
    if warn_boundary
        n_flagged = count(boundary_mask(cfunc_dt; r_thresh=r_thresh))
        if n_flagged > 0
            @warn "form_temps_from_cfunc: $n_flagged of $Nλ wavelengths have flux " *
                  "contribution still peaking at the top of the model atmosphere " *
                  "(ceiling_ratio > $r_thresh), so their formation temperatures are biased " *
                  "toward Ts[1] = $(round(Ts[1], digits=1)) K and should be read as lower " *
                  "limits. Expected in strong saturated cores. Use boundary_mask to exclude " *
                  "them." maxlog=3
        end
    end
    return form
end
