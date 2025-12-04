
function round_to_power(x::Real)
    iszero(x) && return 0
    p = floor(Int, log10(abs(x)))
    return round(x, digits = -p)
end

function searchsortednearest(a::AbstractVector{T}, x::T) where T
    idx = searchsortedfirst(a,x)
    if (idx==1); return idx; end
    if (idx>length(a)); return length(a); end
    if (a[idx]==x); return idx; end
    if (abs(a[idx]-x) < abs(a[idx-1]-x))
        return idx
    else
        return idx-1
    end
end

function searchsortednearest(x::T, a::AbstractVector{T}) where T
    return searchsortednearest(a, x)
end

elav(a::AbstractVector) = elav(a, dims=1)
function elav(a::AbstractArray{T,N}; dims::Integer) where {T,N}
    1 <= dims <= N || throw(ArgumentError("dimension $dims out of range (1:$N)"))

    r = axes(a)
    r0 = ntuple(i -> i == dims ? UnitRange(1, last(r[i]) - 1) : UnitRange(r[i]), N)
    r1 = ntuple(i -> i == dims ? UnitRange(2, last(r[i])) : UnitRange(r[i]), N)
    return (view(a, r1...) .+ view(a, r0...)) ./ 2.0
end

function moving_average(x, w)
    return imfilter(x, ones(w) ./ w, Pad(:replicate))
end

function round_and_format(num::Float64)
    rounded_num = Int(round(num))
    formatted_num = collect(string(rounded_num))
    num_length = length(formatted_num)

    if num_length <= 3
        return prod(formatted_num)
    end

    comma_idx = mod(num_length, 3)
    if comma_idx == 0
        comma_idx = 3
    end

    while comma_idx < num_length
        insert!(formatted_num, comma_idx+1, ',')
        comma_idx += 4
        num_length += 1
    end

    return replace(prod(formatted_num), "," => "{,}")
end

macro cusync(ex...)
    # destructure the `@sync` expression
    code = ex[end]
    kwargs = ex[1:end-1]

    # decode keyword arguments
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || error("Invalid keyword argument $kwarg")
        key, val = kwarg.args
        if key == :blocking
            Base.depwarn("the blocking keyword to @sync has been deprecated", :sync)
        else
            error("Unknown keyword argument $kwarg")
        end
    end

    quote
        local ret = $(esc(code))
        CUDA.synchronize()
        ret
    end
end

function searchsortednearest_gpu(a, x)
    idx = CUDA.searchsortedfirst(a, x)
    if (idx==1); return idx; end
    if (idx>CUDA.length(a)); return CUDA.length(a); end
    if (a[idx]==x); return idx; end
    if (CUDA.abs(a[idx]-x) < CUDA.abs(a[idx-1]-x))
        return idx
    else
        return idx-1
    end
end

function filter_array_gpu!(output, input, pred, n)
    # get indices from GPU blocks + threads
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    for i in idx:sdx:CUDA.length(input)
        if CUDA.isone(pred[i])
            n += 1
            output[n] = input[i]
        end
    end
    return nothing
end

# roll kernel by integer r so zero-lag aligns with padded center
function roll_1d!(dst, src, r, L)
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if j <= L
        jj = j - r
        if jj < 1
            jj += L
        elseif jj > L
            jj -= L
        end
        @inbounds dst[j] = src[jj]
    end
    return nothing
end