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
    _cfunc_measure(T, τ_ref, Natm1) -> Vector{T}

Per-interval width of the reference optical-depth grid, in dex. The measure any statistic
that compares one interval of a contribution function against another must divide out.

An empty `τ_ref` (the Bézier τ path, which has no reference grid) falls back to a uniform
measure, reproducing a bare reduction over the per-interval integrals. That result is
grid-dependent, so it warns.
"""
function _cfunc_measure(::Type{T}, τ_ref::AA{<:AF,1}, Natm1::Int) where T<:AF
    if isempty(τ_ref)
        @warn "cont_func statistics: no reference optical depth grid supplied, falling " *
              "back to a uniform per-interval measure. On a non-uniform layer grid the " *
              "result is grid-dependent. Pass the atmosphere's τ_ref where available." maxlog=1
        return ones(T, Natm1)
    end
    @assert length(τ_ref) == Natm1 + 1 "τ_ref must have one more entry than cfunc_dt has rows"
    return T.(diff(log10.(τ_ref)))
end

"""
    cfunc_per_dex(cfunc_dt, τ_ref) -> Matrix
    cfunc_per_dex(result::FormTempResult) -> Matrix

Convert a contribution function from per-interval integrals to a density per dex of
reference optical depth, `dF / dlog₁₀τ_ref`.

`cont_func` holds integrals: `sum(cont_func, dims=1)` is the emergent flux, so the layer
measure is baked into each element. Sums over it are therefore independent of the layer
grid, but plotting it against depth, or comparing one interval against another, is not —
the native MARCS grid changes spacing by 2× at `log τ_ref = -3` and `+1`, which shows up as
a step of the same size. Divide by the interval width first.

Use this for plotting and for any across-layer comparison. Do **not** use it for weighted
sums over depth: those need the integrals, where the measure cancels (see
[`form_temps_from_cfunc`](@ref)).
"""
function cfunc_per_dex(cfunc_dt::AA{T,2}, τ_ref::AA{<:AF,1}) where T<:AF
    return cfunc_dt ./ _cfunc_measure(T, τ_ref, size(cfunc_dt, 1))
end

cfunc_per_dex(result::FormTempResult) =
    cfunc_per_dex(result.cont_func, result.atmosphere.τs)

"""
    ceiling_ratio(cfunc_dt, τ_ref) -> Vector
    ceiling_ratio(result::FormTempResult) -> Vector

Per-wavelength top-of-atmosphere contamination statistic: the topmost interval's flux
contribution density as a fraction of the column peak density.

Near 0 when the contribution function decays inside the model atmosphere, near 1 when it is
still rising at the top layer, where the formation temperature is a lower limit set by where
the model was truncated. Threshold with [`boundary_mask`](@ref); also stored on
[`FormTempResult`](@ref).

`τ_ref` is required because this compares two intervals: on the native MARCS grid the top
interval is twice as wide as one in the finely-sampled region, which inflates a bare ratio
of integrals by ~2× and over-flags. [`cfunc_per_dex`](@ref) divides that measure out.

An all-zero column yields 0, not `NaN`. [`form_temps_from_cfunc`](@ref) reports those
wavelengths instead.
"""
function ceiling_ratio(cfunc_dt::AA{T,2}, τ_ref::AA{<:AF,1}) where T<:AF
    Δ = _cfunc_measure(T, τ_ref, size(cfunc_dt, 1))
    out = Vector{T}(undef, size(cfunc_dt, 2))
    # the density is formed per element rather than as a matrix: this is called on the full
    # stitched contribution function, where materializing it would cost another Natm×Nλ array
    @inbounds for j in axes(cfunc_dt, 2)
        peak = zero(T)
        for k in eachindex(Δ)
            peak = max(peak, cfunc_dt[k, j] / Δ[k])
        end
        out[j] = (cfunc_dt[1, j] / Δ[1]) / max(peak, eps(T))
    end
    return out
end

ceiling_ratio(result::FormTempResult) = result.ceiling_ratio

"""
    boundary_mask(cfunc_dt, τ_ref; r_thresh=BOUNDARY_R_THRESH) -> BitVector
    boundary_mask(result::FormTempResult; r_thresh=result.r_thresh) -> BitVector

Flag wavelengths whose formation temperature is contaminated by the top of the model
atmosphere, i.e. where [`ceiling_ratio`](@ref) exceeds `r_thresh`.

For a `FormTempResult` the threshold defaults to the one the calculation used, so the mask
matches the wavelengths it warned about. Pass `r_thresh` to re-threshold.
"""
boundary_mask(cfunc_dt::AA{<:AF,2}, τ_ref::AA{<:AF,1};
              r_thresh::Real=BOUNDARY_R_THRESH) =
    ceiling_ratio(cfunc_dt, τ_ref) .> r_thresh

boundary_mask(result::FormTempResult; r_thresh::Real=result.r_thresh) =
    result.ceiling_ratio .> r_thresh

"""
    form_temps_from_cfunc(cfunc_dt, Ts; τ_ref=T[], r_thresh=BOUNDARY_R_THRESH,
                          warn_boundary=true) -> Vector

Formation temperature at 50% of the cumulative flux contribution.

The cumulative is node-anchored: `F = 0` at the top node `Ts[1]`, and the contribution
accumulated through interval `k` is assigned to the deep node `Ts[k+1]`, so the 50% crossing
interpolates against node temperatures `Ts`. Interval-center temperatures (`elav(Ts)`) would
bias the result half an interval cool.

`cfunc_dt` is `(Natm-1, Nλ)` (per-interval flux contribution); `Ts` is `(Natm,)`
node temperatures. CPU; GPU callers pass host copies.

`cfunc_dt` must be the per-interval **integrals**, not a density: the cumulative sum is what
makes this independent of the layer grid, because each element already carries its own
interval width. Passing [`cfunc_per_dex`](@ref) output here would drop that weighting.

Two degenerate cases warn:

- Non-positive column total: no median exists, so the formation temperature is `NaN`.
  Reachable when an upstream microturbulence kernel underflows and zeros a column.
- Boundary contamination: where [`ceiling_ratio`](@ref) exceeds `r_thresh` the contribution is
  still peaking at the top layer, so the formation temperature is a lower limit near `Ts[1]`.
  Flags the same wavelengths as [`boundary_mask`](@ref) at that `r_thresh`;
  `warn_boundary=false` silences it.

`τ_ref` is used only by that second warning, which compares intervals and so needs the layer
measure. Omitting it falls back to a uniform measure, matching what [`ceiling_ratio`](@ref)
does when no reference grid exists, so the warned set always equals the masked set.
"""
function form_temps_from_cfunc(cfunc_dt::AA{T,2}, Ts::AA{T,1};
                               τ_ref::AA{<:AF,1}=T[],
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
        n_flagged = count(boundary_mask(cfunc_dt, τ_ref; r_thresh=r_thresh))
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
