"""
Equation 17.8 from Gray (2008), assuming A_R = A_T and ξ_R = ξ_T
"""
function gray_rt_macro_kernel(vs::AA{T,1}, ζ_rt::T) where T<:AF
    t1 = 2.0 .* exp.(-1.0 .* (vs ./ ζ_rt).^2.0) ./ (sqrt(π) .* ζ_rt)
    t2 = -2.0 .* abs.(vs) .* erfc.(abs.(vs) ./ ζ_rt) ./ ζ_rt.^2.0
    kernel = t1 .+ t2
    return kernel ./ sum(kernel)
end

function convolve_gray_rt_macro(xs::AA{T,1}, ys::AA{T,1}, ζ_rt::T) where T<:AF
    # short circuit
    if iszero(ζ_rt)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = gray_rt_macro_kernel(vs, ζ_rt)
    kshift = ifftshift(kernel)

    # return convolution via FFT (matches GPU convention)
    return real(ifft(fft(ys) .* fft(kshift)))
end

function convolve_gray_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_rt::T) where T<:AF
    # short circuit
    if iszero(ζ_rt)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = gray_rt_macro_kernel(vs, ζ_rt)
    kshift = ifftshift(kernel)
    ftk = fft(kshift)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end

function compute_padded_gray_rt_kernel_1D!(kernel_row, xs, λc, ζ_rt, Nλ, pad_left)
    # get thread index
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # evaluate the kernel
    if j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc
        av = CUDA.abs(xj)
        z = av / ζ_rt

        t1 = 2.0 * exp(-(xj/ζ_rt)^2) / (sqrt(π) * ζ_rt)
        t2 = -2.0 * av * erfc(z) / (ζ_rt^2)
        @inbounds kernel_row[j + pad_left] = t1 + t2
    end
    return nothing
end

function convolve_gray_rt_macro_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                                    ys::AA{T,2}, ζ_rt::T) where {T<:AF}
    # copy to device
    copyto!(cmem.xs_gpu, CuArray(xs))
    copyto!(cmem.ys_gpu, CuArray(ys))

    # short circuit
    if iszero(ζ_rt)
        return cmem.ys_gpu
    end

    # compute velocity offset from discrete center
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)
    CUDA.synchronize()

    # compute the padded kernel once
    kernel_row = reshape(@view(cmem.padded_kernel_gpu[1, :]), :)
    shifted_kernel_row = reshape(@view(cmem.shift_kernel_gpu[1, :]), :)

    fill!(kernel_row, zero(T))
    ts1 = (256,)
    bs1 = (cld(cmem.Nλ, ts1[1]),)
    @cuda threads=ts1 blocks=bs1 compute_padded_gray_rt_kernel_1D!(kernel_row,
                                                                   cmem.xs_gpu, λ0,
                                                                   ζ_rt, cmem.Nλ,
                                                                   cmem.pad_left)
    CUDA.synchronize()

    # normalize the kernel
    normval = CUDA.sum(kernel_row)
    kernel_row ./= normval

    # ensure zero-lag sits at padded center before FFT layout
    Ltot = length(kernel_row)
    center = Ltot ÷ 2
    r = center - (cmem.pad_left + i0)
    if r != 0
        @cuda threads=ts1 blocks=(cld(Ltot, ts1[1]),) roll_1d!(shifted_kernel_row, kernel_row, r, Ltot)
        CUDA.synchronize()
        tmp = kernel_row
        kernel_row = shifted_kernel_row
        shifted_kernel_row = tmp
    end

    # center -> FFT indexing
    CUDA.CUFFT.ifftshift!(shifted_kernel_row, kernel_row, 1)

    # make a contiguous 1-D device vector
    kr = copy(shifted_kernel_row)

    # forward fourier transforms (R2C on device)
    kernel_row_ft = CUDA.CUFFT.rfft(kr)
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem
    kft = reshape(kernel_row_ft, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft

    # inverse fourier transform
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region
    # out = @view cmem.conv_gpu[:, cmem.pad_left : cmem.pad_left + cmem.Nλ - 1]
    out = cmem.conv_gpu[:, cmem.pad_left : cmem.pad_left + cmem.Nλ - 1]
    CUDA.synchronize()
    return out
end
