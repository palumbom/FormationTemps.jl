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
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))

    σ(x) = max(x * (σ_v / c_ms), σ_floor)
    g(x, n) = exp(-((x - n) / σ(x))^2.0)

    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    λc = (μ_v / c_ms) * λ0 + λ0

    kernel = g.(xs, λc)
    kernel ./= sum(kernel)

    return _padded_convolve(collect(T, ys), kernel)
end

function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, μ_v::AA{T,1}, σ_v::AA{T,1}) where {T<:AF}
    Nλ = length(xs)
    Npad = 512
    L, _, pad_left, _ = _conv_mem_geometry(Nλ, Npad)

    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]

    ys_out = zeros(T, size(ys))
    kbuf = zeros(T, L)
    kvec = Vector{T}(undef, Nλ)
    sig = zeros(T, L)

    for t in axes(ys, 1)
        σ(x) = max(x * (σ_v[t] / c_ms), σ_floor)
        g(x, n) = exp(-((x - n) / σ(x))^2.0)
        λc = (μ_v[t] / c_ms) * λ0 + λ0

        @inbounds for j in 1:Nλ
            kvec[j] = g(xs[j], λc)
        end
        s = sum(kvec)
        kvec ./= s

        _kernel_to_dft_layout!(kbuf, kvec, i0)
        _pad_edges!(sig, view(ys, t, :), pad_left, Nλ)

        sig_ft = rfft(sig)
        ker_ft = rfft(kbuf)
        sig_ft .*= ker_ft
        conv = irfft(sig_ft, L)
        @inbounds for j in 1:Nλ
            ys_out[t, j] = conv[pad_left + j]
        end
    end
    return ys_out
end

"""
    _convolve_micro_inplace!(out, xs, ys, μ_v, σ_v, ws)

In-place microturbulent broadening using pre-allocated [`CPUTileWorkspace`](@ref)
buffers. When `μ_v` and `σ_v` are uniform across rows (the common case during
disk integration), the kernel FFT is computed once and reused for all atmosphere
layers. Otherwise falls back to per-row kernel computation. The vector interface
for `σ_v` is preserved so that per-layer microturbulence can be supported in the
future.
"""
function _convolve_micro_inplace!(out::AA{T,2}, xs::AA{T,1}, ys::AA{T,2},
                                  μ_v::AA{T,1}, σ_v::AA{T,1},
                                  ws::CPUTileWorkspace) where T<:AF
    Nλ = ws.Nλ
    Natm = size(ys, 1)
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]

    # check if kernel is uniform across rows (common case in disk integration)
    uniform = _allequal(μ_v) && _allequal(σ_v)

    # temporary Nλ-length kernel buffer (reused per iteration)
    kvec = Vector{T}(undef, Nλ)

    if uniform
        σ_val = σ_v[1]
        μ_val = μ_v[1]
        λc = (μ_val / c_ms) * λ0 + λ0
        @inbounds for j in 1:Nλ
            σx = max(xs[j] * (σ_val / c_ms), σ_floor)
            kvec[j] = exp(-((xs[j] - λc) / σx)^2.0)
        end
        s = sum(kvec)
        kvec ./= s

        _kernel_to_dft_layout!(ws.kernel_real, kvec, i0)
        mul!(ws.kernel_ft, ws.fft_plan, ws.kernel_real)

        _apply_fft_kernel!(out, ys, ws.kernel_ft, ws, Natm)
    else
        # per-row kernel (when σ_v or μ_v vary across atmosphere layers)
        for t in 1:Natm
            σ_val = σ_v[t]
            μ_val = μ_v[t]
            λc = (μ_val / c_ms) * λ0 + λ0
            @inbounds for j in 1:Nλ
                σx = max(xs[j] * (σ_val / c_ms), σ_floor)
                kvec[j] = exp(-((xs[j] - λc) / σx)^2.0)
            end
            s = sum(kvec)
            kvec ./= s

            _kernel_to_dft_layout!(ws.kernel_real, kvec, i0)
            mul!(ws.kernel_ft, ws.fft_plan, ws.kernel_real)

            # pad signal row, R2C FFT, convolve, extract
            _pad_edges!(ws.signal_padded, view(ys, t, :), ws.pad_left, Nλ)
            mul!(ws.signal_ft, ws.fft_plan, ws.signal_padded)
            ws.signal_ft .*= ws.kernel_ft
            mul!(ws.result_buf, ws.ifft_plan, ws.signal_ft)
            @inbounds for j in 1:Nλ
                out[t, j] = ws.result_buf[ws.pad_left + j]
            end
        end
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

# fused: computes per-row Doppler params inline and builds the filter in one kernel launch
function build_doppler_filter_direct!(filter, μ_v, σ_v, scale, s_max, invL, nfreq)
    i    = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    f_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(filter, 1) && f_idx <= nfreq
        T = eltype(μ_v)
        s = clamp(@inbounds(μ_v[i]) * scale, -s_max, s_max)
        σ = @inbounds(σ_v[i]) * scale
        f0 = T(f_idx - 1)
        θ = -T(2π) * f0 * s * invL
        gauss = exp(-(T(π) * σ * f0 * invL)^2)
        sθ, cθ = sincos(θ)
        @inbounds filter[i, f_idx] = complex(gauss * cθ, gauss * sθ)
    end
    return nothing
end

# batched Doppler filter: B*Natm rows, σ_v is shared (length Natm, index wraps)
function build_doppler_filter_batched!(filter, μ_v_batch, σ_v, scale, s_max, invL, nfreq, Natm, BNatm)
    i     = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    f_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= BNatm && f_idx <= nfreq
        T = eltype(μ_v_batch)
        i_local = ((i - 1) % Natm) + 1
        s = clamp(@inbounds(μ_v_batch[i]) * scale, -s_max, s_max)
        σ = @inbounds(σ_v[i_local]) * scale
        f0 = T(f_idx - 1)
        θ = -T(2π) * f0 * s * invL
        gauss = exp(-(T(π) * σ * f0 * invL)^2)
        sθ, cθ = sincos(θ)
        @inbounds filter[i, f_idx] = complex(gauss * cθ, gauss * sθ)
    end
    return nothing
end

# broadcast shared signal_ft (Natm rows) across B*Natm filter rows
function batched_spectral_multiply!(conv_ft, signal_ft, kernel_ft, Natm, BNatm)
    i     = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    f_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= BNatm && f_idx <= size(conv_ft, 2)
        i_local = ((i - 1) % Natm) + 1
        @inbounds conv_ft[i, f_idx] = signal_ft[i_local, f_idx] * kernel_ft[i, f_idx]
    end
    return nothing
end

"""
    convolve_wavelength_axis_batched!(bcmem, xs, ys, μ_v_batch, σ_v, Bcur)

Batched Doppler convolution for `Bcur` tiles simultaneously. The absorption signal
`ys` (Natm × Nλ) is shared across tiles; `μ_v_batch` (Bcur*Natm) provides per-tile
velocities. Returns a view of the valid region `(Bcur*Natm, Nλ)`.
"""
function convolve_wavelength_axis_batched!(bcmem::BatchedMicroConvMem{T},
                                           xs::AA{T,1}, ys::AA{T,2},
                                           μ_v_batch::CA{T,1}, σ_v::CA{T,1},
                                           Bcur::Int) where {T<:AF}
    Natm = bcmem.Natm
    BNatm = Bcur * Natm

    # initialize wavelength-to-pixel conversion once
    if !bcmem.doppler_ready
        i0 = length(xs) ÷ 2 + 1
        λ0 = xs[i0]
        Δλ = median(diff(xs))
        bcmem.doppler_scale = T(λ0 / (c_ms * Δλ))
        bcmem.doppler_ready = true
    end
    s_max = T(bcmem.pad_left - 1)
    invL = inv(T(bcmem.L))

    # pad + FFT the shared signal (skip if cached)
    if !bcmem.signal_cached
        copyto!(bcmem.ys_gpu, ys)
        ts = (32, 32)
        bs = (cld(Natm, ts[1]), cld(bcmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(bcmem.signal_gpu, bcmem.ys_gpu,
                                               bcmem.Nλ, bcmem.pad_left, bcmem.pad_right)
        mul!(bcmem.signal_ft_gpu, bcmem.plan_fwd, bcmem.signal_gpu)
    end

    # build batched Doppler filter (BNatm rows)
    nfreq = size(bcmem.kernel_ft_gpu, 2)
    BNatm32 = Int32(BNatm)
    ts2 = (32, 32)
    bs2 = (cld(nfreq, ts2[1]), cld(BNatm, ts2[2]))
    @cuda threads=ts2 blocks=bs2 build_doppler_filter_batched!(bcmem.kernel_ft_gpu,
                                                                μ_v_batch, σ_v,
                                                                bcmem.doppler_scale, s_max,
                                                                invL, nfreq, Int32(Natm),
                                                                BNatm32)

    # broadcast signal_ft × filter → conv_ft (custom kernel avoids replicating signal)
    bs3 = (cld(nfreq, ts2[1]), cld(BNatm, ts2[2]))
    @cuda threads=ts2 blocks=bs3 batched_spectral_multiply!(bcmem.conv_ft_gpu,
                                                             bcmem.signal_ft_gpu,
                                                             bcmem.kernel_ft_gpu,
                                                             Int32(Natm), BNatm32)

    # batched inverse FFT
    mul!(bcmem.conv_gpu, bcmem.plan_bwd, bcmem.conv_ft_gpu)

    # return view of valid region
    return @view bcmem.conv_gpu[1:BNatm, bcmem.pad_left+1:bcmem.pad_left+bcmem.Nλ]
end

"""
    convolve_wavelength_axis_gpu(cmem, xs, ys, μ_v, σ_v)

GPU implementation of [`convolve_wavelength_axis`](@ref). Applies a per-row Doppler
shift and Gaussian broadening in the Fourier domain using an analytical filter.

Arguments:
- `cmem::AbstractConvolutionMemory`: Pre-allocated GPU working memory.
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
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
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

    # pad + FFT the signal (skip if signal_cached — αs unchanged since last call)
    if !cmem.signal_cached
        copyto!(cmem.ys_gpu, ys)
        ts = (32, 32)
        bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                               cmem.Nλ, cmem.pad_left, cmem.pad_right)
        mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)
    end

    # fused: compute per-row Doppler params + build Fourier filter in one kernel
    nfreq = size(cmem.kernel_ft_gpu, 2)
    ts2 = (32, 32)
    bs2 = (cld(nfreq, ts2[1]), cld(cmem.Natm, ts2[2]))
    @cuda threads=ts2 blocks=bs2 build_doppler_filter_direct!(cmem.kernel_ft_gpu,
                                                               μ_v, σ_v,
                                                               cmem.doppler_scale, s_max,
                                                               invL, nfreq)

    # convolution theorem + inverse FFT
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region (signal occupies pad_left+1 : pad_left+Nλ in 1-indexed)
    return @view cmem.conv_gpu[:, cmem.pad_left+1:cmem.pad_left + cmem.Nλ]
end

# device-native overload: accepts CuArray inputs and avoids GPU scalar indexing
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory,
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

    # pad + FFT the signal (skip if signal_cached — αs unchanged since last call)
    if !cmem.signal_cached
        ts = (32, 32)
        bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, ys_d,
                                               cmem.Nλ, cmem.pad_left,
                                               cmem.pad_right)
        mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)
    end

    # fused: compute per-row Doppler params + build Fourier filter in one kernel
    nfreq = size(cmem.kernel_ft_gpu, 2)
    ts2 = (32, 32)
    bs2 = (cld(nfreq, ts2[1]), cld(cmem.Natm, ts2[2]))
    @cuda threads=ts2 blocks=bs2 build_doppler_filter_direct!(cmem.kernel_ft_gpu,
                                                               μ_v_d, σ_v_d,
                                                               cmem.doppler_scale, s_max,
                                                               invL, nfreq)

    # convolution theorem + inverse FFT
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region (signal occupies pad_left+1 : pad_left+Nλ in 1-indexed)
    return @view cmem.conv_gpu[:, cmem.pad_left+1:cmem.pad_left + cmem.Nλ]
end
