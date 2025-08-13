function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, μ_v::T, σ_v::T) where {T<:AF}
    # gaussian width depends on wavelength (constant in velocity)
    σ(x) = x * (σ_v / c_ms)
    g(x, n) = exp(-((x - n) / σ(x))^2.0)

    # offset the kernel by the velocity
    λ0 = mean(xs)
    λc = (μ_v / c_ms) * λ0 + λ0

    # sample the kernel
    kernel = g.(xs, λc)

    # normalize
    kernel ./= sum(kernel)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))

    # loop over slices of atmosphere
    for t in axes(ys, 1)
        ys_out[t, :] .= imfilter(ys[t, :], reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
    end
    return ys_out
end

function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, μ_v::AA{T,1}, σ_v::AA{T,1}) where {T<:AF}
    # allocate for kernel 
    kernel = zeros(length(xs))

    # allocate array for output spectrum
    ys_out = zeros(size(ys))

    # loop over slices of atmosphere
    for t in axes(ys, 1)
        # gaussian width depends on wavelength (constant in velocity)
        σ(x) = x * (σ_v[t] / c_ms)
        g(x, n) = exp(-((x - n) / σ(x))^2.0)

        # offset the kernel by the velocity
        λ0 = mean(xs)
        λc = (μ_v[t] / c_ms) * λ0 + λ0

        # sample the kernel
        kernel .= g.(xs, λc)

        # normalize
        kernel ./= sum(kernel)

        ys_out[t, :] .= imfilter(ys[t, :], reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
    end
    return ys_out
end

function compute_padded_kernel2D!(kernel, xs, λc, σ_fac, Nλ, pad_left)
    # get thread indices
    i = (blockIdx().y-1) * blockDim().y + threadIdx().y
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # loop over wavelength and atmosphere layer
    if i <= size(kernel,1) && j <= Nλ
        xj = xs[j]
        σi = xj * σ_fac[i]
        @inbounds kernel[i, j + pad_left] = exp(-((xj - λc[i]) / σi)^2.0)
        # TODO: handle denom going to zero^^^
    end
    return nothing
end

function pad_signal!(signal, ys, Nλ, pad_left, pad_right)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    Natm, L = size(signal)

    if row <= Natm && col <= L
        y = @view ys[row, :]

        if col <= pad_left
            @inbounds signal[row, col] = y[1]
        elseif col <= pad_left + Nλ
            @inbounds signal[row, col] = y[col - pad_left]
        elseif col <= L
            @inbounds signal[row, col] = y[end]
        end
    end
    return nothing
end

function convolve_wavelength_axis_gpu(xs::AA{T,1}, ys::AA{T,2}, μ_v::AA{T,1}, σ_v::AA{T,1}; Npad::Int=2400) where {T<:AF}
    # figure out indices + offsets for padding
    Nλ = length(xs)
    Natm = size(ys, 1)
    L = Nλ + Npad
    pad_left = Npad ÷ 2
    pad_right = L - Nλ - pad_left

    # compute velocity offset and width in wavelength units
    λ0 = mean(xs)
    λc = λ0 .+ (μ_v ./ c_ms) .* λ0
    σ_fac = σ_v ./ c_ms 

    # allocate on and send data to gpu
    ys_gpu = CuArray{T}(ys)
    signal_gpu = CUDA.zeros(T, Natm, L)
    kernel_gpu = CUDA.zeros(T, Natm, L)
    xs_gpu = CuArray{T}(xs)
    λc_gpu = CuArray{T}(λc)
    σ_fac_gpu = CuArray{T}(σ_fac)

    # pad the signal
    ts = (32,32)
    bs = (cld(Natm, ts[1]), cld(L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(signal_gpu, ys_gpu, Nλ, pad_left, pad_right)

    # compute the padded kernel
    ts = (32,32)
    bs = (cld(Nλ, ts[1]), cld(Natm, ts[2]))
    @cuda threads=ts blocks=bs compute_padded_kernel2D!(kernel_gpu, xs_gpu, λc_gpu, 
                                                        σ_fac_gpu, Nλ, pad_left)

    # synchronize before normalizing the kernel and shifting for fft
    kernel_gpu ./= sum(kernel_gpu, dims=2)
    kernel_gpu = ifftshift(kernel_gpu, 2)

    # convolution theorem
    kernel_ft = rfft(kernel_gpu, 2)
    signal_ft = rfft(signal_gpu, 2)
    conv_gpu = irfft(signal_ft .* kernel_ft, L, 2)
    # return Array(@view conv_gpu[:, pad_left + 1 : pad_left + Nλ])
    return @view conv_gpu[:, pad_left + 1 : pad_left + Nλ]
end

function convolve_wavelength_axis_gpu(cmem::ConvolutionMemory, xs::AA{T,1}, 
                                      ys::AA{T,2}, μ_v::CA{T,1}, σ_v::CA{T,1}) where {T<:AF}
    # copy to device
    copyto!(cmem.xs_gpu, xs)
    copyto!(cmem.ys_gpu, ys)

    # compute velocity offset and width in wavelength units
    λ0 = mean(cmem.xs_gpu)
    copyto!(cmem.λc_gpu, λ0 .* (1 .+ μ_v ./ c_ms))
    copyto!(cmem.σ_fac_gpu, σ_v ./ c_ms)

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, 
                                           cmem.pad_right)

    # compute the padded kernel
    fill!(cmem.padded_kernel_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs compute_padded_kernel2D!(cmem.padded_kernel_gpu,
                                                        cmem.xs_gpu, cmem.λc_gpu,
                                                        cmem.σ_fac_gpu,
                                                        cmem.Nλ, cmem.pad_left)

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
    return @view cmem.conv_gpu[:, cmem.pad_left+1 : cmem.pad_left + cmem.Nλ]
end

function convolve_wavelength_axis_gpu(cmem::ConvolutionMemory, xs::AA{T,1}, 
                                      ys::AA{T,2}, μ_v::AA{T,1}, σ_v::AA{T,1}) where {T<:AF}
    # copy to device
    copyto!(cmem.xs_gpu, xs)
    copyto!(cmem.ys_gpu, ys)

    # compute velocity offset and width in wavelength units
    let λ0 = mean(cmem.xs_gpu)
        cmem.λc_vec .= λ0 .* (1 .+ μ_v ./ c_ms)
        cmem.σ_fac_vec .= σ_v ./ c_ms
    end
    copyto!(cmem.λc_gpu, cmem.λc_vec)
    copyto!(cmem.σ_fac_gpu, cmem.σ_fac_vec)

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, 
                                           cmem.pad_right)

    # compute the padded kernel
    fill!(cmem.padded_kernel_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs compute_padded_kernel2D!(cmem.padded_kernel_gpu,
                                                        cmem.xs_gpu, cmem.λc_gpu,
                                                        cmem.σ_fac_gpu,
                                                        cmem.Nλ, cmem.pad_left)

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
    return @view cmem.conv_gpu[:, cmem.pad_left+1 : cmem.pad_left + cmem.Nλ]
end