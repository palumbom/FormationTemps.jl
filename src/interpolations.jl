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

"""
    form_temps_from_cfunc(cfunc_dt, Ts; warn_boundary=true) -> Vector

Formation temperature at 50% of the cumulative flux contribution.

Uses a node-anchored cumulative: `F = 0` at the top node `Ts[1]`, and the
contribution accumulated through interval `k` is assigned to the deep node
`Ts[k+1]`; the 50% crossing is interpolated against the node temperatures `Ts`.
This avoids the half-interval cool bias of pairing the cumulative (a cell-edge
quantity) with interval-center temperatures.

`cfunc_dt` is `(Natm-1, Nλ)` (per-interval flux contribution); `Ts` is `(Natm,)`
node temperatures. CPU; GPU callers pass host copies.

Two degenerate outcomes are reported rather than returned silently:

- **Non-positive total contribution.** A column whose contribution does not sum to a
  positive number has no median to find; its formation temperature is `NaN`. This is
  reachable — the microturbulence underflow guard deliberately zeros a row when the Doppler
  shift moves the kernel out of the window (see the "Kernel normalization underflow guard"
  note), which can zero a whole column downstream.
- **Boundary-pinned columns.** If the crossing falls in the first interval, over half the
  flux contribution comes from the topmost layer pair and the returned value is set by
  where the model atmosphere was truncated, not by where the line actually forms. This is
  expected in deep line cores (the contribution function rises toward the top boundary as
  `E₂(τ)` truncates) and is now common by default, since hydrogen lines are included and
  Balmer cores form far above the MARCS photosphere. Treat those wavelengths as lower
  limits. `warn_boundary=false` silences the warning.
"""
function form_temps_from_cfunc(cfunc_dt::AA{T,2}, Ts::AA{T,1};
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

    form     = zeros(T, Nλ)
    Fnodes   = zeros(T, Natm)
    n_pinned = 0
    @inbounds for i in 1:Nλ
        if empty_col[1, i]
            form[i] = T(NaN)
            continue
        end
        Fnodes[1] = zero(T)
        @views Fnodes[2:end] .= cum[:, i]     # cum[k] ↦ deep node Ts[k+1]
        form[i] = linear_interp(Fnodes, Ts)(T(0.5))
        # crossing inside the first interval ⇒ value fixed by the top-of-atmosphere cutoff
        Fnodes[2] >= T(0.5) && (n_pinned += 1)
    end

    if n_empty > 0
        @warn "form_temps_from_cfunc: $n_empty of $Nλ wavelengths have no positive total " *
              "flux contribution; their formation temperatures are NaN. Usually an upstream " *
              "microturbulence kernel underflow (check velocity units, m/s)." maxlog=3
    end
    if warn_boundary && n_pinned > 0
        @warn "form_temps_from_cfunc: $n_pinned of $Nλ wavelengths reach 50% cumulative " *
              "contribution within the topmost layer interval, so their formation " *
              "temperatures are pinned to the top of the model atmosphere (Ts[1] = " *
              "$(round(Ts[1], digits=1)) K) and should be read as lower limits. Expected " *
              "in deep line cores (e.g. Balmer), which form above the MARCS grid." maxlog=3
    end
    return form
end
