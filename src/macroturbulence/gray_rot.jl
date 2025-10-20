"""
Equation 18.14 from The Observation and Analysis of Stellar Photospheres
(Gray 2008)
"""
function gray_rot_kernel(vs::AA{T,1}, vsini::T, u1::T) where T<:AF
    # get LD terms
    ld1 = 2.0 * (one(T) - u1)
    ld2 = 0.5 * π * u1 
    ld3 = π * (one(T) - u1 / 3.0)

    # evaluate the kernel 
    xs = vs ./ vsini
    omx2 = abs.(one(T) .- xs .^ 2.0)
    kernel = (ld1 .* sqrt.(omx2) .+ ld2 .* omx2) ./ ld3
    kernel[abs.(xs) .> one(T)] .= zero(T)
    return kernel ./ sum(kernel)
end

function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,1}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rot_kernel(vs, vsini, u1)

    # return convolution
    return imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
end

function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,2}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rot_kernel(vs, vsini, u1)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= imfilter(ys[t, :], reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
    end
    return ys_out
end

function compute_padded_gray_kernel_2D!(kernel, xs, λc, vsini, u1, Nλ, pad_left)
    # get thread indices
    i = (blockIdx().y-1) * blockDim().y + threadIdx().y
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # get LD terms
    ld1 = 2.0 * (1.0 - u1)
    ld2 = 0.5 * π * u1 
    ld3 = π * (1.0 - u1 / 3.0)

    # loop over wavelength and atmosphere layer
    if i <= size(kernel,1) && j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc / vsini
        omx2 = CUDA.abs(1.0 - xj ^ 2.0)

        val = (ld1 * sqrt(omx2) + ld2 * omx2) / ld3
        val *= abs(xj) <= 1.0
        @inbounds kernel[i, j + pad_left] = val
    end
    return nothing
end

function convolve_gray_rotation_gpu(cmem::ConvolutionMemory, xs::AA{T,1}, 
                                    ys::AA{T,2}, vsini::T, u1::T) where {T<:AF}
    # copy to device
    copyto!(cmem.xs_gpu, CuArray(xs))
    copyto!(cmem.ys_gpu, CuArray(ys))

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
    @cuda threads=ts blocks=bs compute_padded_gray_kernel_2D!(cmem.padded_kernel_gpu,
                                                              cmem.xs_gpu, λ0, vsini,
                                                              u1, cmem.Nλ, cmem.pad_left)
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