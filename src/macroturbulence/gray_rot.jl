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

"""
    convolve_gray_rotation(xs, ys, vsini, u1)

Convolve a spectrum with the Gray (2008) rotation kernel using linear limb darkening.

TODO: finish docs.
"""
function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,1}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase: zero-lag at index 1)
    kernel = gray_rot_kernel(vs, vsini, u1)
    kshift = ifftshift(kernel)

    # return convolution via FFT (matches GPU convention)
    return real(ifft(fft(ys) .* fft(kshift)))
end

function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,2}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = gray_rot_kernel(vs, vsini, u1)
    kshift = ifftshift(kernel)
    ftk = fft(kshift)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end

function compute_padded_gray_kernel_1D!(kernel_row, xs, λc, vsini, u1, Nλ, pad_left)
    # get thread index
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # get LD terms
    ld1 = 2.0 * (1.0 - u1)
    ld2 = 0.5 * π * u1
    ld3 = π * (1.0 - u1 / 3.0)

    # evaluate the kernel
    if j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc / vsini
        omx2 = CUDA.abs(1.0 - xj ^ 2.0)

        val = (ld1 * sqrt(omx2) + ld2 * omx2) / ld3
        val *= abs(xj) <= 1.0
        @inbounds kernel_row[j + pad_left] = val
    end
    return nothing
end

function convolve_gray_rotation_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                                    ys::AA{T,2}, vsini::T, u1::T) where {T<:AF}
    # copy inputs to device
    copyto!(cmem.xs_gpu, CuArray(xs))
    copyto!(cmem.ys_gpu, CuArray(ys))

    # compute velocity offset from discrete center
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)
    CUDA.synchronize()

    # kernel rows as 1-D views to avoid dim ambiguity
    kernel_row = reshape(@view(cmem.padded_kernel_gpu[1, :]), :)
    shifted_kernel_row = reshape(@view(cmem.shift_kernel_gpu[1, :]), :)
    fill!(kernel_row, zero(T))
    
    ts1 = (256,)
    bs1 = (cld(cmem.Nλ, ts1[1]),)
    @cuda threads=ts1 blocks=bs1 compute_padded_gray_kernel_1D!(kernel_row,
                                                                cmem.xs_gpu, λ0,
                                                                vsini, u1,
                                                                cmem.Nλ, cmem.pad_left)
    CUDA.synchronize()

    # normalize the kernel
    normval = CUDA.sum(kernel_row)
    kernel_row ./= normval

    # ensure zero-lag sits at padded center before FFT layout
    Ltot = length(kernel_row)
    center = Ltot ÷ 2
    r = center - (cmem.pad_left + i0)  # integer roll needed if pads not perfectly symmetric
    if r != 0
        @cuda threads=ts1 blocks=(cld(Ltot, ts1[1]),) roll_1d!(shifted_kernel_row, kernel_row, r, Ltot)
        CUDA.synchronize()
        tmp = kernel_row
        kernel_row = shifted_kernel_row
        shifted_kernel_row = tmp
    end

    # center -> FFT indexing
    CUDA.CUFFT.ifftshift!(shifted_kernel_row, kernel_row, 1)

    # contiguous 1-D vector for rfft
    kr = copy(shifted_kernel_row)

    # forward fourier transforms (R2C on device; length = floor(Int, L/2)+1)
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
