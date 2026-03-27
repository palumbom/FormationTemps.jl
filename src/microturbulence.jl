"""
    convolve_wavelength_axis(xs, ys, μ_v, σ_v)

Convolve each row of `ys` with a Gaussian kernel that models microturbulent broadening
and a Doppler shift, using FFT convolution. The kernel width is wavelength-dependent
(constant in velocity units).

Arguments:
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}`: Input matrix with shape `(Natm, Nλ)`.
- `μ_v::Real` or `AbstractVector{<:Real}`: Line-of-sight velocity per row (m/s). A scalar
  applies the same shift to every row; a vector specifies per-row shifts.
- `σ_v::Real` or `AbstractVector{<:Real}`: Gaussian broadening width per row (m/s).

Returns:
- `ys_out::AbstractMatrix{<:Real}`: Broadened matrix with the same shape as `ys`.

Notes:
- Uses a real-space sampled Gaussian kernel. This differs from the analytical
  Fourier-domain Gaussian used by [`convolve_wavelength_axis_gpu`](@ref) when σ < ~3 pixels,
  producing systematic flux differences of ~4×10⁻⁴ at σ ≈ 1.8 px (ξ ≈ 850 m/s, Δλ = 0.01 Å).

See also: [`convolve_wavelength_axis_gpu`](@ref)
"""
function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, μ_v::T, σ_v::T) where {T<:AF}
    # clamp broadening to prevent div by 0
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))

    # gaussian width depends on wavelength (constant in velocity)
    σ(x) = max(x * (σ_v / c_ms), σ_floor)
    g(x, n) = exp(-((x - n) / σ(x))^2.0)

    # offset the kernel by the velocity (use discrete center to avoid half-sample offset)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    λc = (μ_v / c_ms) * λ0 + λ0

    # sample and normalize the kernel
    kernel = g.(xs, λc)
    kernel ./= sum(kernel)

    # FFT-style convolution that preserves the μ_v offset
    kshift = ifftshift(kernel)
    ftk = fft(kshift)

    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end

function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, μ_v::AA{T,1}, σ_v::AA{T,1}) where {T<:AF}
    # allocate for kernel 
    kernel = zeros(length(xs))

    # allocate array for output spectrum
    ys_out = zeros(size(ys))

    # clamp broadening to prevent div by 0
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))

    # discrete center reference
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]

    # loop over slices of atmosphere
    for t in axes(ys, 1)
        # gaussian width depends on wavelength (constant in velocity)
        σ(x) = max(x * (σ_v[t] / c_ms), σ_floor)
        g(x, n) = exp(-((x - n) / σ(x))^2.0)

        # offset the kernel by the velocity
        λc = (μ_v[t] / c_ms) * λ0 + λ0

        # sample and normalize the kernel
        kernel .= g.(xs, λc)
        kernel ./= sum(kernel)

        # FFT-style convolution that preserves the μ_v offset
        kshift = ifftshift(kernel)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* fft(kshift)))
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
        if col <= pad_left
            @inbounds signal[row, col] = ys[row, 1]
        elseif col <= pad_left + Nλ
            @inbounds signal[row, col] = ys[row, col - pad_left]
        elseif col <= L
            @inbounds signal[row, col] = ys[row, Nλ]
        end
    end
    return nothing
end

# extract valid (unpadded) region from conv_gpu into out_gpu
function extract_valid!(out, src, pad_left, Nλ)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= size(out, 1) && col <= Nλ
        @inbounds out[row, col] = src[row, col + pad_left - 1]
    end
    return nothing
end

# roll each row by integer r[row] so zero-lag aligns with padded center (without removing μ_v)
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

# Precompute row-wise Doppler shift/broadening in pixel units.
function precompute_doppler_params!(shift_pix, sigma_pix, μ_v, σ_v, scale, s_max)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(shift_pix)
        @inbounds shift_pix[i] = clamp(μ_v[i] * scale, -s_max, s_max)
        @inbounds sigma_pix[i] = σ_v[i] * scale
    end
    return nothing
end

# Build per-row Fourier-domain filter for a Doppler shift + Gaussian broadening.
# H[i, f] = exp(-2πi · f · shift_pix[i] / L) · exp(-(π · sigma_pix[i] · f / L)^2)
# f is a 1-indexed column mapped to the 0-indexed frequency bin f-1.
function build_doppler_filter!(filter, shift_pix, sigma_pix, invL, nfreq)
    i    = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    f_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(filter, 1) && f_idx <= nfreq
        # frequency bin in [0, nfreq-1]
        T = eltype(shift_pix)
        f0 = T(f_idx - 1)
        s = @inbounds shift_pix[i]
        σ = @inbounds sigma_pix[i]
        θ = -T(2π) * f0 * s * invL
        gauss = exp(-(T(π) * σ * f0 * invL)^2)
        sθ, cθ = sincos(θ)
        @inbounds filter[i, f_idx] = complex(gauss * cθ, gauss * sθ)
    end
    return nothing
end

"""
    convolve_wavelength_axis_gpu(cmem, xs, ys, μ_v, σ_v)

GPU implementation of [`convolve_wavelength_axis`](@ref). Applies a per-row Doppler
shift and Gaussian broadening in the Fourier domain using an analytical filter.

Arguments:
- `cmem::ConvolutionMemory`: Pre-allocated GPU working memory.
- `xs::AbstractVector{<:Real}` or `CuArray{<:Real,1}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}` or `CuArray{<:Real,2}`: Input matrix `(Natm, Nλ)`.
- `μ_v::CuArray{<:Real,1}`: Per-row Doppler velocity shift (m/s).
- `σ_v::CuArray{<:Real,1}`: Per-row Gaussian broadening width (m/s).

Returns:
- A view of `cmem.conv_gpu[:, pad_left+1 : pad_left+Nλ]` containing the broadened result.

Notes:
- Uses an analytical Fourier-domain Gaussian filter (H[f] = exp(−(πσf/L)²) × phase shift),
  which is more accurate than a sampled real-space kernel when σ < ~3 pixels.
- When `cmem.signal_cached` is `true`, the signal padding and forward FFT are skipped;
  set this flag when the absorption coefficients have not changed since the last call
  (e.g., across rotation tiles in the disk integration loop).

See also: [`convolve_wavelength_axis`](@ref)
"""
function convolve_wavelength_axis_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, μ_v::CA{T,1}, σ_v::CA{T,1}) where {T<:AF}
    # compute per-row shift (pixels) and Gaussian width (pixels)
    # s[i]     = μ_v[i] * λ0 / (c * Δλ)  — shift of row i in pixels
    # σ_pix[i] = σ_v[i] * λ0 / (c * Δλ)  — Gaussian broadening width in pixels
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    Δλ = median(diff(xs))
    cmem.doppler_scale = T(λ0 / (c_ms * Δλ))
    cmem.doppler_ready = true
    s_max = T(cmem.pad_left - 1)
    invL = inv(T(cmem.L))

    # precompute per-row Doppler parameters (reuse existing row buffers)
    ts_params = 256
    bs_params = cld(cmem.Natm, ts_params)
    @cuda threads=ts_params blocks=bs_params precompute_doppler_params!(
        cmem.λc_gpu, cmem.σ_fac_gpu, μ_v, σ_v, cmem.doppler_scale, s_max)

    # pad + FFT the signal (skip if signal_cached — αs unchanged since last call)
    if !cmem.signal_cached
        copyto!(cmem.ys_gpu, ys)
        ts = (32, 32)
        bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                               cmem.Nλ, cmem.pad_left, cmem.pad_right)
        mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)
    end

    # build per-row Fourier filter analytically (no spatial kernel, no normalization)
    nfreq = size(cmem.kernel_ft_gpu, 2)
    ts2 = (32, 32)
    bs2 = (cld(nfreq, ts2[1]), cld(cmem.Natm, ts2[2]))
    @cuda threads=ts2 blocks=bs2 build_doppler_filter!(cmem.kernel_ft_gpu,
                                                       cmem.λc_gpu, cmem.σ_fac_gpu,
                                                       invL, nfreq)

    # convolution theorem + inverse FFT
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region (signal occupies pad_left+1 : pad_left+Nλ in 1-indexed)
    return @view cmem.conv_gpu[:, cmem.pad_left+1:cmem.pad_left + cmem.Nλ]
end

# device-native overload: accepts CuArray inputs and avoids GPU scalar indexing
function convolve_wavelength_axis_gpu(cmem::ConvolutionMemory,
                                      xs_d::CuArray{T,1},
                                      ys_d::CuArray{T,2},
                                      μ_v_d::CuArray{T,1},
                                      σ_v_d::CuArray{T,1}) where {T<:AF}
    # initialize wavelength-to-pixel conversion once per memory object
    if !cmem.doppler_ready
        xs_h = Array(xs_d)
        i0 = length(xs_h) ÷ 2 + 1
        λ0 = xs_h[i0]
        Δλ = median(diff(xs_h))
        cmem.doppler_scale = T(λ0 / (c_ms * Δλ))
        cmem.doppler_ready = true
    end

    # maximum shift clamp in pixel units
    s_max = T(cmem.pad_left - 1)
    invL = inv(T(cmem.L))

    # precompute per-row Doppler parameters (reuse existing row buffers)
    ts_params = 256
    bs_params = cld(cmem.Natm, ts_params)
    @cuda threads=ts_params blocks=bs_params precompute_doppler_params!(
        cmem.λc_gpu, cmem.σ_fac_gpu, μ_v_d, σ_v_d, cmem.doppler_scale, s_max)

    # pad + FFT the signal (skip if signal_cached — αs unchanged since last call)
    if !cmem.signal_cached
        ts = (32, 32)
        bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, ys_d,
                                               cmem.Nλ, cmem.pad_left,
                                               cmem.pad_right)
        mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)
    end

    # build per-row Fourier filter analytically (no spatial kernel, no normalization)
    nfreq = size(cmem.kernel_ft_gpu, 2)
    ts2 = (32, 32)
    bs2 = (cld(nfreq, ts2[1]), cld(cmem.Natm, ts2[2]))
    @cuda threads=ts2 blocks=bs2 build_doppler_filter!(cmem.kernel_ft_gpu,
                                                       cmem.λc_gpu, cmem.σ_fac_gpu,
                                                       invL, nfreq)

    # convolution theorem + inverse FFT
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region (signal occupies pad_left+1 : pad_left+Nλ in 1-indexed)
    return @view cmem.conv_gpu[:, cmem.pad_left+1:cmem.pad_left + cmem.Nλ]
end
