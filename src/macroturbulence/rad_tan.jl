"""
Equation 17.6 from Gray (2008), assuming A_R = A_T and ξ_R = ξ_T
"""
function rt_macro_kernel(vs::AA{T,1}, ζ_rt::T, μ::T) where T<:AF
    # constants
    A_R = 0.5
    A_T = A_R
    sqrt_π = sqrt(π)

    # get trig
    ϵ = T(1e-6)
    cosθ = max(μ, ϵ)
    s2 = one(T) - μ * μ
    sinθ = sqrt(ifelse(s2 > zero(T), s2, ϵ*ϵ))

    # the terms
    t1 = @. A_R * exp(-(vs / (ζ_rt * cosθ))^2.0) / (sqrt_π * ζ_rt * cosθ)
    t2 = @. A_T * exp(-(vs / (ζ_rt * sinθ))^2.0) / (sqrt_π * ζ_rt * sinθ)
    kernel = t1 + t2
    return kernel ./ sum(kernel)
end

function convolve_rt_macro(xs::AA{T,1}, ys::AA{T,1}, ζ_rt::T, μ::T;
                           pad_left::Int=0, pad_right::Int=0) where T<:AF
    # short circuit
    if iszero(ζ_rt)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    Nλ = length(xs)
    Ltot = Nλ + pad_left + pad_right
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = rt_macro_kernel(vs, ζ_rt, μ)

    # pad kernel to Ltot, align zero-lag to padded center, then map to FFT indexing
    kpad = zeros(T, Ltot)
    @views kpad[pad_left+1:pad_left+Nλ] .= kernel
    center = Ltot ÷ 2 + 1
    r = center - (pad_left + i0)
    if r != 0
        kpad = circshift(kpad, r)
    end
    kshift = ifftshift(kpad)
    ftk = fft(kshift)

    # pad signal the same way, convolve in Fourier space, slice valid region
    ypad = zeros(T, Ltot)
    @views ypad[pad_left+1:pad_left+Nλ] .= ys
    conv = real(ifft(fft(ypad) .* ftk))
    return @view conv[pad_left+1:pad_left+Nλ]
end

function convolve_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_rt::T, μ::T;
                           pad_left::Int=0, pad_right::Int=0) where T<:AF
    # short circuit
    if iszero(ζ_rt)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    Nλ = length(xs)
    Ltot = Nλ + pad_left + pad_right
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = rt_macro_kernel(vs, ζ_rt, μ)
    kpad = zeros(T, Ltot)
    @views kpad[pad_left+1:pad_left+Nλ] .= kernel
    center = Ltot ÷ 2 + 1
    r = center - (pad_left + i0)
    if r != 0
        kpad = circshift(kpad, r)
    end
    ftk = fft(ifftshift(kpad))

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    ypad = zeros(T, Ltot)
    for t in axes(ys, 1)
        fill!(ypad, zero(T))
        @views ypad[pad_left+1:pad_left+Nλ] .= ys[t, :]
        conv = real(ifft(fft(ypad) .* ftk))
        @views ys_out[t, :] .= conv[pad_left+1:pad_left+Nλ]
    end
    return ys_out
end

function compute_padded_rt_kernel_1D!(kernel_row, xs, λc, ζ_rt, μ, Nλ, pad_left)
    # get thread index
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # evaluate the kernel
    if j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc

        # trig from μ directly; guard μ≈0 or μ≈1
        T = typeof(ζ_rt)
        ϵ = T(1e-6)
        cosθ = max(μ, ϵ)
        s2 = one(T) - μ * μ
        sinθ = sqrt(ifelse(s2 > zero(T), s2, ϵ*ϵ))

        invR = 0.5 / (sqrt(π) * ζ_rt * cosθ)
        invT = 0.5 / (sqrt(π) * ζ_rt * sinθ)

        t1 = exp(-(xj / (ζ_rt * cosθ))^2) * invR
        t2 = exp(-(xj / (ζ_rt * sinθ))^2) * invT
        @inbounds kernel_row[j + pad_left] = t1 + t2
    end
    return nothing
end

function convolve_rt_macro_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                               ys::AA{T,2}, ζ_rt::T, μ::T) where {T<:AF}
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
    @cuda threads=ts1 blocks=bs1 compute_padded_rt_kernel_1D!(kernel_row,
                                                              cmem.xs_gpu, λ0,
                                                              ζ_rt, μ, cmem.Nλ,
                                                              cmem.pad_left)
    CUDA.synchronize()

    # normalize the kernel
    normval = CUDA.sum(kernel_row)
    kernel_row ./= normval

    # ensure zero-lag sits at padded center before FFT layout
    Ltot = length(kernel_row)
    center = Ltot ÷ 2 + 1
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

    # if your cuFFT backward plan is unnormalized, apply 1/Ltot scaling once here
    cmem.conv_gpu .*= inv(T(Ltot))  # remove this line if your plan already scales

    # slice valid region
    out = cmem.conv_gpu[:, cmem.pad_left : cmem.pad_left + cmem.Nλ - 1]
    CUDA.synchronize()
    return out
end
