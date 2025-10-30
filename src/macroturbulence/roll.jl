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