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

# Default contamination threshold, as a fraction of the column peak. The statistic is
# strongly bimodal (near 0 or near 1), so the exact cut matters little.
const BOUNDARY_R_THRESH = 0.33

"""
    ceiling_ratio(cfunc_dt) -> Vector
    ceiling_ratio(result::FormTempResult) -> Vector

Per-wavelength top-of-atmosphere contamination statistic: the topmost interval's flux
contribution as a fraction of the column peak.

Near 0 when the contribution function decays inside the model atmosphere, near 1 when it is
still rising at the top layer, where the formation temperature is a lower limit set by where
the model was truncated. Threshold with [`boundary_mask`](@ref); also stored on
[`FormTempResult`](@ref).

An all-zero column yields 0, not `NaN`. [`form_temps_from_cfunc`](@ref) reports those
wavelengths instead.
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
matches the wavelengths it warned about. Pass `r_thresh` to re-threshold.
"""
boundary_mask(cfunc_dt::AA{<:AF,2}; r_thresh::Real=BOUNDARY_R_THRESH) =
    ceiling_ratio(cfunc_dt) .> r_thresh

boundary_mask(result::FormTempResult; r_thresh::Real=result.r_thresh) =
    result.ceiling_ratio .> r_thresh

"""
    form_temps_from_cfunc(cfunc_dt, Ts; r_thresh=BOUNDARY_R_THRESH, warn_boundary=true) -> Vector

Formation temperature at 50% of the cumulative flux contribution.

The cumulative is node-anchored: `F = 0` at the top node `Ts[1]`, and the contribution
accumulated through interval `k` is assigned to the deep node `Ts[k+1]`, so the 50% crossing
interpolates against node temperatures `Ts`. Interval-center temperatures (`elav(Ts)`) would
bias the result half an interval cool.

`cfunc_dt` is `(Natm-1, Nλ)` (per-interval flux contribution); `Ts` is `(Natm,)`
node temperatures. CPU; GPU callers pass host copies.

Two degenerate cases warn:

- Non-positive column total: no median exists, so the formation temperature is `NaN`.
  Reachable when an upstream microturbulence kernel underflows and zeros a column.
- Boundary contamination: where [`ceiling_ratio`](@ref) exceeds `r_thresh` the contribution is
  still peaking at the top layer, so the formation temperature is a lower limit near `Ts[1]`.
  Flags the same wavelengths as [`boundary_mask`](@ref) at that `r_thresh`;
  `warn_boundary=false` silences it.
"""
function form_temps_from_cfunc(cfunc_dt::AA{T,2}, Ts::AA{T,1};
                               r_thresh::Real=BOUNDARY_R_THRESH,
                               warn_boundary::Bool=true) where T<:AF
    Natm = length(Ts)
    @assert size(cfunc_dt, 1) == Natm - 1 "cfunc_dt must have Natm-1 rows to match Ts"
    Nλ = size(cfunc_dt, 2)
    cum = cumsum(cfunc_dt, dims=1)

    # normalize by the column total, not the maximum: the two differ unless the cumulative is
    # monotonic. cum[end:end, :] is a copy, so the in-place divide reads a fixed divisor.
    # Non-positive totals are flagged NaN below — zero gives 0/0, and a roundoff-negative
    # total inverts the CDF, which linear_interp resolves to the deepest temperature.
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
    # same statistic and threshold as boundary_mask, so warning and mask always agree
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
