function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, μ_v::T, σ_v::T) where {T<:AF}
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))

    # discrete center reference (GPU convention)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    λc = (μ_v / c_ms) * λ0 + λ0

    # kernel with wavelength-dependent width
    σ(x) = max(x * (σ_v / c_ms), σ_floor)
    g(x, n) = exp(-((x - n) / σ(x))^2.0)
    kernel = g.(xs, λc)
    kernel ./= sum(kernel)

    # snap center to grid index and align like GPU
    jc = clamp(searchsortedlast(xs, λc), 1, length(xs))
    r = i0 - jc
    kroll = circshift(kernel, r)
    kshift = ifftshift(kroll)   # zero-lag at index 1

    ftk = fft(kshift)

    ys_out = similar(ys)
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end

function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, μ_v::AA{T,1}, σ_v::AA{T,1}) where {T<:AF}
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))

    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]

    ys_out = similar(ys)

    for t in axes(ys, 1)
        λc = (μ_v[t] / c_ms) * λ0 + λ0

        σ(x) = max(x * (σ_v[t] / c_ms), σ_floor)
        g(x, n) = exp(-((x - n) / σ(x))^2.0)
        kernel = g.(xs, λc)
        kernel ./= sum(kernel)

        jc = clamp(searchsortedlast(xs, λc), 1, length(xs))
        r = i0 - jc
        kroll = circshift(kernel, r)
        kshift = ifftshift(kroll)

        ftk = fft(kshift)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end


function compute_padded_kernel2D!(kernel, xs, λc, σ_fac, Nλ, pad_left, σ_floor)
    # get thread indices
    i = (blockIdx().y-1) * blockDim().y + threadIdx().y
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # loop over wavelength and atmosphere layer
    if i <= size(kernel,1) && j <= Nλ
        xj = xs[j]
        σi = max(xj * σ_fac[i], σ_floor)
        @inbounds kernel[i, j + pad_left] = exp(-((xj - λc[i]) / σi)^2.0)
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

# roll each row by integer r[row] so zero-lag aligns with padded center
function roll_rows_2d!(dst, src, r, L)
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= size(src,1) && col <= L
        rr = r[row]
        jj = col - rr
        if jj < 1
            jj += L
        elseif jj > L
            jj -= L
        end
        @inbounds dst[row, col] = src[row, jj]
    end
    return nothing
end

function convolve_wavelength_axis_gpu(cmem::ConvolutionMemory, xs::AA{T,1}, 
                                      ys::AA{T,2}, μ_v::AA{T,1}, σ_v::AA{T,1}) where {T<:AF}
    # copy to device
    copyto!(cmem.xs_gpu, xs)
    copyto!(cmem.ys_gpu, ys)

    # compute velocity offset and width in wavelength units (discrete center)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    μ_host = Array(μ_v)
    λc_host = λ0 .* (1 .+ μ_host ./ c_ms)
    copyto!(cmem.λc_gpu, CuArray(λc_host))
    copyto!(cmem.σ_fac_gpu, σ_v ./ c_ms)

    # clamp broadening to prevent div by 0
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.1) * Δλ))

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, 
                                           cmem.pad_right)
    CUDA.synchronize()

    # compute the padded kernel
    fill!(cmem.padded_kernel_gpu, zero(T))
    ts = (32,32)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs compute_padded_kernel2D!(cmem.padded_kernel_gpu,
                                                        cmem.xs_gpu, cmem.λc_gpu,
                                                        cmem.σ_fac_gpu,
                                                        cmem.Nλ, cmem.pad_left,
                                                        σ_floor)
    CUDA.synchronize()

    # normalize the kernel
    cmem.norm_buffer .= CUDA.sum(cmem.padded_kernel_gpu, dims=2)
    cmem.padded_kernel_gpu ./= cmem.norm_buffer

    # compute per-row integer shifts so zero-lag is at padded center
    jc_host = Vector{Int}(undef, cmem.Natm)
    for t in 1:cmem.Natm
        jc = searchsortedlast(xs, λc_host[t])
        jc_host[t] = clamp(jc, 1, cmem.Nλ)
    end
    center = cmem.L ÷ 2
    r_host = center .- (cmem.pad_left .+ jc_host)
    r_gpu = CuArray(Int32.(r_host))

    # roll rows by r[row]
    tsr = (32,32)
    bsr = (cld(cmem.L, tsr[1]), cld(cmem.Natm, tsr[2]))
    @cuda threads=tsr blocks=bsr roll_rows_2d!(cmem.shift_kernel_gpu, cmem.padded_kernel_gpu, r_gpu, cmem.L)
    CUDA.synchronize()

    # center -> FFT indexing
    CUDA.CUFFT.ifftshift!(cmem.padded_kernel_gpu, cmem.shift_kernel_gpu, 2)

    # forward fourier transforms
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.padded_kernel_gpu)
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu

    # inverse fourier transform
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region
    out = @view cmem.conv_gpu[:, cmem.pad_left: cmem.pad_left + cmem.Nλ - 1]
    CUDA.synchronize()
    return out
end

function convolve_wavelength_axis_gpu(cmem::ConvolutionMemory, xs_d::CA{T,1}, 
                                      ys_d::CA{T,2}, μ_v_d::CA{T,1}, 
                                      σ_v_d::CA{T,1}) where {T<:AF}
    # copy device -> device into cmem buffers (no host round-trips)
    copyto!(cmem.xs_gpu, xs_d)
    copyto!(cmem.ys_gpu, ys_d)

    # host-side tiny summaries (bulk copies; no scalar GPU indexing)
    xs_h = Array(xs_d)                         # one bulk transfer
    μ_h = Array(μ_v_d)                         # one bulk transfer

    # discrete-center reference to avoid half-sample phase
    i0 = length(xs_h) ÷ 2 + 1
    λ0 = xs_h[i0]

    # per-layer center wavelengths and sigma factors (host → device once)
    λc_h = λ0 .* (1 .+ μ_h ./ c_ms)
    copyto!(cmem.λc_gpu, CuArray(λc_h))
    cmem.σ_fac_gpu .= σ_v_d ./ c_ms           # device-side broadcast

    # clamp broadening to prevent div by 0 (computed on host)
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))

    # pad the signal (replicate)
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)
    CUDA.synchronize()

    # compute the padded kernel on device
    fill!(cmem.padded_kernel_gpu, zero(T))
    ts2 = (32,32)
    bs2 = (cld(cmem.Nλ, ts2[1]), cld(cmem.Natm, ts2[2]))
    @cuda threads=ts2 blocks=bs2 compute_padded_kernel2D!(cmem.padded_kernel_gpu,
                                                          cmem.xs_gpu, cmem.λc_gpu,
                                                          cmem.σ_fac_gpu,
                                                          cmem.Nλ, cmem.pad_left,
                                                          σ_floor)
    CUDA.synchronize()

    # normalize kernel row-wise on device
    cmem.norm_buffer .= CUDA.sum(cmem.padded_kernel_gpu, dims=2)
    cmem.padded_kernel_gpu ./= cmem.norm_buffer

    # compute per-row integer rolls on host (bulk), avoiding GPU scalar access
    jc_h = searchsortedlast.(Ref(xs_h), λc_h)
    jc_h .= clamp.(jc_h, 1, cmem.Nλ)
    center = cmem.L ÷ 2
    r_h = center .- (cmem.pad_left .+ jc_h)
    r_d = CuArray(Int32.(r_h))

    # roll rows on device by r[row]
    tsr = (32,32)
    bsr = (cld(cmem.L, tsr[1]), cld(cmem.Natm, tsr[2]))
    @cuda threads=tsr blocks=bsr roll_rows_2d!(cmem.shift_kernel_gpu, cmem.padded_kernel_gpu, r_d, cmem.L)
    CUDA.synchronize()

    # center -> FFT indexing
    CUDA.CUFFT.ifftshift!(cmem.padded_kernel_gpu, cmem.shift_kernel_gpu, 2)

    # forward fourier transforms
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.padded_kernel_gpu)
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu

    # inverse fourier transform and slice valid region
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)
    out = @view cmem.conv_gpu[:, cmem.pad_left : cmem.pad_left + cmem.Nλ - 1]
    CUDA.synchronize()
    return out
end