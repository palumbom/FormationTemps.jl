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

    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rt_macro_kernel(vs, ζ_rt)

    # return convolution
    return imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
end 

function convolve_gray_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_rt::T) where T<:AF
    # short circuit
    if iszero(ζ_rt)
        return ys
    end

    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rt_macro_kernel(vs, ζ_rt)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= imfilter(ys[t, :], reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
    end
    return ys_out
end 

function compute_padded_gray_rt_kernel_2D!(kernel, xs, λc, ζ_rt, Nλ, pad_left)
    # get thread indices
    i = (blockIdx().y-1) * blockDim().y + threadIdx().y
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # loop over wavelength and atmosphere layer
    if i <= size(kernel,1) && j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc
        av = CUDA.abs(xj)
        z = av / ζ_rt

        t1 = 2.0 * exp(-(xj/ζ_rt)^2) / (sqrt(π) * ζ_rt)
        t2 = -2.0 * av * erfc(z) / (ζ_rt^2)
        val = t1 + t2
        @inbounds kernel[i, j + pad_left] = val
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

    # compute velocity offset 
    λ0 = mean(cmem.xs_gpu)

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, 
                                           cmem.pad_right)
    CUDA.synchronize()

    # compute the padded kernel
    fill!(cmem.padded_kernel_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs compute_padded_gray_rt_kernel_2D!(cmem.padded_kernel_gpu,
                                                                 cmem.xs_gpu, λ0, ζ_rt,
                                                                 cmem.Nλ, cmem.pad_left)
    CUDA.synchronize()

    # normalize the kernel
    cmem.norm_buffer .= CUDA.sum(cmem.padded_kernel_gpu, dims=2)
    cmem.padded_kernel_gpu ./= cmem.norm_buffer

    # shift the kernel so it is centered
    CUDA.CUFFT.ifftshift!(cmem.shift_kernel_gpu, cmem.padded_kernel_gpu, 2)

    # forward fourier transforms
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.shift_kernel_gpu)
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu

    # inverse fourier transform
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region
    # copyto!(cmem.ys_gpu, cmem.conv_gpu[:, cmem.pad_left+1 : cmem.pad_left + cmem.Nλ])
    # return nothing
    out = @view cmem.conv_gpu[:, cmem.pad_left+1 : cmem.pad_left + cmem.Nλ]
    CUDA.synchronize()
    return out
end