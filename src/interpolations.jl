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
    form_temps_from_cfunc(cfunc_dt, Ts) -> Vector

Formation temperature at 50% of the cumulative flux contribution.

Uses a node-anchored cumulative: `F = 0` at the top node `Ts[1]`, and the
contribution accumulated through interval `k` is assigned to the deep node
`Ts[k+1]`; the 50% crossing is interpolated against the node temperatures `Ts`.
This avoids the half-interval cool bias of pairing the cumulative (a cell-edge
quantity) with interval-center temperatures.

`cfunc_dt` is `(Natm-1, Nλ)` (per-interval flux contribution); `Ts` is `(Natm,)`
node temperatures. CPU; GPU callers pass host copies.
"""
function form_temps_from_cfunc(cfunc_dt::AA{T,2}, Ts::AA{T,1}) where T<:AF
    Natm = length(Ts)
    @assert size(cfunc_dt, 1) == Natm - 1 "cfunc_dt must have Natm-1 rows to match Ts"
    Nλ = size(cfunc_dt, 2)
    cum = cumsum(cfunc_dt, dims=1)
    cum ./= cum[end:end, :]               # normalize by total (definition: 50% of total
                                          # contribution); == maximum only if monotonic.
                                          # cum[end:end, :] is a copy, so the in-place
                                          # divide reads a fixed divisor.
    form   = zeros(T, Nλ)
    Fnodes = zeros(T, Natm)
    @inbounds for i in 1:Nλ
        Fnodes[1] = zero(T)
        @views Fnodes[2:end] .= cum[:, i]     # cum[k] ↦ deep node Ts[k+1]
        form[i] = linear_interp(Fnodes, Ts)(T(0.5))
    end
    return form
end
