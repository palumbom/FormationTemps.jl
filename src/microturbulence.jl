"""
    convolve_wavelength_axis(xs, ys, μ_v, σ_v)

Convolve each row of `ys` with a Gaussian kernel that models microturbulent broadening
and a Doppler shift, using FFT convolution. The kernel width is wavelength-dependent
(constant in velocity units): σ(x) = x·σ_v/c.

Scalar `μ_v` and `σ_v` apply the same kernel to every row; vectors specify per-row
values. The GPU implementation (`convolve_wavelength_axis_gpu`) builds the same
kernel on device, so CPU and GPU agree to floating-point precision.
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

# ── CPU in-place (disk integration) ─────────────────────────────────────────

"""
    _convolve_micro_inplace!(out, xs, ys, μ_v, σ_v, ws)

In-place microturbulent broadening using pre-allocated [`CPUTileWorkspace`](@ref)
buffers.

Scalar `μ_v` and `σ_v` build one kernel and apply it to all atmosphere layers
(the common case in disk integration). Vector arguments build per-row kernels.
"""
function _convolve_micro_inplace!(out::AA{T,2}, xs::AA{T,1}, ys::AA{T,2},
                                  μ_v::T, σ_v::T,
                                  ws::CPUTileWorkspace) where T<:AF
    Nλ = ws.Nλ
    Natm = size(ys, 1)
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]
    λc = (μ_v / c_ms) * λ0 + λ0

    kvec = Vector{T}(undef, Nλ)
    @inbounds for j in 1:Nλ
        σx = max(xs[j] * (σ_v / c_ms), σ_floor)
        kvec[j] = exp(-((xs[j] - λc) / σx)^2.0)
    end
    kvec ./= sum(kvec)

    _kernel_to_dft_layout!(ws.kernel_real, kvec, i0)
    mul!(ws.kernel_ft, ws.fft_plan, ws.kernel_real)
    _apply_fft_kernel!(out, ys, ws.kernel_ft, ws, Natm)
    return nothing
end

function _convolve_micro_inplace!(out::AA{T,2}, xs::AA{T,1}, ys::AA{T,2},
                                  μ_v::AA{T,1}, σ_v::AA{T,1},
                                  ws::CPUTileWorkspace) where T<:AF
    Nλ = ws.Nλ
    Natm = size(ys, 1)
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]
    kvec = Vector{T}(undef, Nλ)

    for t in 1:Natm
        λc = (μ_v[t] / c_ms) * λ0 + λ0
        @inbounds for j in 1:Nλ
            σx = max(xs[j] * (σ_v[t] / c_ms), σ_floor)
            kvec[j] = exp(-((xs[j] - λc) / σx)^2.0)
        end
        kvec ./= sum(kvec)

        _kernel_to_dft_layout!(ws.kernel_real, kvec, i0)
        mul!(ws.kernel_ft, ws.fft_plan, ws.kernel_real)

        _pad_edges!(ws.signal_padded, view(ys, t, :), ws.pad_left, Nλ)
        mul!(ws.signal_ft, ws.fft_plan, ws.signal_padded)
        ws.signal_ft .*= ws.kernel_ft
        mul!(ws.result_buf, ws.ifft_plan, ws.signal_ft)
        @inbounds for j in 1:Nλ
            out[t, j] = ws.result_buf[ws.pad_left + j]
        end
    end
    return nothing
end

# ── GPU CUDA kernels ─────────────────────────────────────────────────────────

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

function extract_valid!(out, src, pad_left, Nλ)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= size(out, 1) && col <= Nλ
        @inbounds out[row, col] = src[row, col + pad_left - 1]
    end
    return nothing
end

# Build ONE kernel in DFT layout (scalar μ_v, σ_v).
function kernel_to_dft_layout_1d_gpu!(kbuf, xs, λ0, μ_v_val, σ_v_val, σ_floor, i0, Nλ, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > Nλ && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    σx = max(xj * (σ_v_val / T(c_ms)), σ_floor)
    # avoid catastrophic cancellation: (xj - λ0) and (μ_v/c)*λ0 are both small
    Δx = (xj - λ0) - (μ_v_val / T(c_ms)) * λ0
    val = exp(-(Δx / σx)^2)
    d = j - i0
    idx = d >= 0 ? d + 1 : L + d + 1
    @inbounds kbuf[idx] = val
    return nothing
end

# Build per-row kernels in DFT layout (vector μ_v, vector σ_v).
function kernel_to_dft_layout_2d_gpu!(kbuf, xs, μ_v, μ_v_off, σ_v, σ_floor, i0, Nλ, L)
    j   = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    Nrows = size(kbuf, 1)
    (row > Nrows || j > Nλ) && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    λ0 = @inbounds xs[i0]
    σx = max(xj * (@inbounds σ_v[row]) / T(c_ms), σ_floor)
    Δx = (xj - λ0) - (@inbounds μ_v[μ_v_off + row]) / T(c_ms) * λ0
    val = exp(-(Δx / σx)^2)
    d = j - i0
    idx = d >= 0 ? d + 1 : L + d + 1
    @inbounds kbuf[row, idx] = val
    return nothing
end

# Build per-row kernels in DFT layout (vector μ_v, scalar σ_v).
function kernel_to_dft_layout_2d_scalar_σ_gpu!(kbuf, xs, μ_v, μ_v_off, σ_v_val, σ_floor, i0, Nλ, L)
    j   = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    Nrows = size(kbuf, 1)
    (row > Nrows || j > Nλ) && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    λ0 = @inbounds xs[i0]
    σx = max(xj * (σ_v_val / T(c_ms)), σ_floor)
    Δx = (xj - λ0) - (@inbounds μ_v[μ_v_off + row]) / T(c_ms) * λ0
    val = exp(-(Δx / σx)^2)
    d = j - i0
    idx = d >= 0 ? d + 1 : L + d + 1
    @inbounds kbuf[row, idx] = val
    return nothing
end

# broadcast shared signal_ft across per-row kernel FTs
function batched_spectral_multiply!(conv_ft, signal_ft, kernel_ft, Natm, BNatm)
    f_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i     = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= BNatm && f_idx <= size(conv_ft, 2)
        i_local = ((i - 1) % Natm) + 1
        @inbounds conv_ft[i, f_idx] = signal_ft[i_local, f_idx] * kernel_ft[i, f_idx]
    end
    return nothing
end

# ── GPU host helpers ─────────────────────────────────────────────────────────

function _init_micro_params!(cmem, xs_h::AbstractVector{T}) where T<:AF
    copyto!(cmem.xs_gpu, xs_h)
    cmem.doppler_ready = true
    return nothing
end

# pad signal + forward FFT (host-array ys)
function _pad_and_fft_signal!(cmem, ys::AA{T,2}, xs_h::AbstractVector{T}) where T<:AF
    if !cmem.doppler_ready
        _init_micro_params!(cmem, xs_h)
    end
    if !cmem.signal_cached
        copyto!(cmem.ys_gpu, ys)
        ts = (32, 32)
        bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                               cmem.Nλ, cmem.pad_left, cmem.pad_right)
        mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)
    end
    return nothing
end

# pad signal + forward FFT (device-array ys)
function _pad_and_fft_signal!(cmem, ys_d::CuArray{T,2}) where T<:AF
    if !cmem.signal_cached
        ts = (32, 32)
        bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, ys_d,
                                               cmem.Nλ, cmem.pad_left, cmem.pad_right)
        mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)
    end
    return nothing
end

# build 1D kernel FT (scalar μ_v, σ_v → single kernel, broadcast)
function _build_kernel_ft_1d!(cmem, xs_h::AbstractVector{T}, μ_v_val::T, σ_v_val::T) where T<:AF
    Nλ = cmem.Nλ; L = cmem.L
    i0 = Nλ ÷ 2 + 1
    λ0 = xs_h[i0]
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    fill!(cmem.kr_1d, zero(T))
    ts = (256,); bs = (cld(Nλ, ts[1]),)
    @cuda threads=ts blocks=bs kernel_to_dft_layout_1d_gpu!(
        cmem.kr_1d, cmem.xs_gpu, T(λ0), μ_v_val, σ_v_val, σ_floor,
        Int32(i0), Int32(Nλ), Int32(L))
    normval = CUDA.sum(cmem.kr_1d)
    cmem.kr_1d ./= normval
    mul!(cmem.kernel_row_ft_1d, cmem.plan_fwd_1d, cmem.kr_1d)
    return nothing
end

# build per-row kernels into conv_gpu, normalize, batch-FFT into kernel_ft_gpu
function _build_per_row_kernels!(cmem, xs_h::AbstractVector{T},
                                  μ_v::CA{T,1}, σ_v::T) where T<:AF
    i0 = cmem.Nλ ÷ 2 + 1
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    Nrows = size(cmem.conv_gpu, 1)

    fill!(cmem.conv_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(Nrows, ts[2]))
    @cuda threads=ts blocks=bs kernel_to_dft_layout_2d_scalar_σ_gpu!(
        cmem.conv_gpu, cmem.xs_gpu, μ_v, Int32(0), σ_v, σ_floor,
        Int32(i0), Int32(cmem.Nλ), Int32(cmem.L))
    cmem.conv_gpu ./= sum(cmem.conv_gpu, dims=2)
    return nothing
end

function _build_per_row_kernels!(cmem, xs_h::AbstractVector{T},
                                  μ_v::CA{T,1}, σ_v::CA{T,1}) where T<:AF
    i0 = cmem.Nλ ÷ 2 + 1
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    Nrows = size(cmem.conv_gpu, 1)

    fill!(cmem.conv_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(Nrows, ts[2]))
    @cuda threads=ts blocks=bs kernel_to_dft_layout_2d_gpu!(
        cmem.conv_gpu, cmem.xs_gpu, μ_v, Int32(0), σ_v, σ_floor,
        Int32(i0), Int32(cmem.Nλ), Int32(cmem.L))
    cmem.conv_gpu ./= sum(cmem.conv_gpu, dims=2)
    return nothing
end

# IRFFT + return valid region view
function _irfft_and_extract(cmem)
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)
    return @view cmem.conv_gpu[:, cmem.pad_left+1:cmem.pad_left+cmem.Nλ]
end

# ── GPU public API: convolve_wavelength_axis_gpu ─────────────────────────────

"""
    convolve_wavelength_axis_gpu(cmem, xs, ys, μ_v, σ_v)

GPU implementation of [`convolve_wavelength_axis`](@ref). Builds the same real-space
kernel with wavelength-dependent σ(x) = x·σ_v/c as the CPU, matching exactly.

Scalar `μ_v` and `σ_v` build one kernel broadcast to all rows. Vector arguments
build per-row kernels via batch R2C FFT.

See also: [`convolve_wavelength_axis`](@ref)
"""
# scalar μ_v, scalar σ_v — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, μ_v::T, σ_v::T) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_kernel_ft_1d!(cmem, xs_h, μ_v, σ_v)
    kft = reshape(cmem.kernel_row_ft_1d, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft
    return _irfft_and_extract(cmem)
end

# scalar μ_v, scalar σ_v — device xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory,
                                      xs_d::CuArray{T,1}, ys_d::CuArray{T,2},
                                      μ_v::T, σ_v::T) where {T<:AF}
    if !cmem.doppler_ready
        _init_micro_params!(cmem, Array(xs_d))
    end
    _pad_and_fft_signal!(cmem, ys_d)
    xs_h = Array(cmem.xs_gpu)
    _build_kernel_ft_1d!(cmem, xs_h, μ_v, σ_v)
    kft = reshape(cmem.kernel_row_ft_1d, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft
    return _irfft_and_extract(cmem)
end

# vector μ_v, scalar σ_v — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, μ_v::CA{T,1}, σ_v::T) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_per_row_kernels!(cmem, xs_h, μ_v, σ_v)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# vector μ_v, scalar σ_v — device xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory,
                                      xs_d::CuArray{T,1}, ys_d::CuArray{T,2},
                                      μ_v::CA{T,1}, σ_v::T) where {T<:AF}
    if !cmem.doppler_ready
        _init_micro_params!(cmem, Array(xs_d))
    end
    _pad_and_fft_signal!(cmem, ys_d)
    xs_h = Array(cmem.xs_gpu)
    _build_per_row_kernels!(cmem, xs_h, μ_v, σ_v)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# vector μ_v, vector σ_v — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, μ_v::CA{T,1}, σ_v::CA{T,1}) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_per_row_kernels!(cmem, xs_h, μ_v, σ_v)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# vector μ_v, vector σ_v — device xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory,
                                      xs_d::CuArray{T,1}, ys_d::CuArray{T,2},
                                      μ_v::CA{T,1}, σ_v::CA{T,1}) where {T<:AF}
    if !cmem.doppler_ready
        _init_micro_params!(cmem, Array(xs_d))
    end
    _pad_and_fft_signal!(cmem, ys_d)
    xs_h = Array(cmem.xs_gpu)
    _build_per_row_kernels!(cmem, xs_h, μ_v, σ_v)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# ── GPU batched API ──────────────────────────────────────────────────────────

"""
    convolve_wavelength_axis_batched!(bcmem, xs, ys, μ_v_batch, σ_v, Bcur)

Batched Doppler convolution for `Bcur` tiles simultaneously. The absorption signal
`ys` (Natm × Nλ) is shared across tiles; `μ_v_batch` (Bcur*Natm) provides per-tile
velocities. Scalar `σ_v` avoids tiling allocations; vector `σ_v` is tiled across tiles.
Returns a view of the valid region `(Bcur*Natm, Nλ)`.
"""
function convolve_wavelength_axis_batched!(bcmem::BatchedMicroConvMem{T},
                                           xs::AA{T,1}, ys::AA{T,2},
                                           μ_v_batch::CA{T,1}, σ_v::T,
                                           Bcur::Int; tile_offset::Int=0) where {T<:AF}
    Natm = bcmem.Natm
    BNatm = Bcur * Natm
    xs_h = xs isa CuArray ? Array(xs) : collect(T, xs)

    if !bcmem.doppler_ready
        _init_micro_params!(bcmem, xs_h)
    end

    if !bcmem.signal_cached
        copyto!(bcmem.ys_gpu, ys)
        ts = (32, 32)
        bs = (cld(Natm, ts[1]), cld(bcmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(bcmem.signal_gpu, bcmem.ys_gpu,
                                               bcmem.Nλ, bcmem.pad_left, bcmem.pad_right)
        mul!(bcmem.signal_ft_gpu, bcmem.plan_fwd, bcmem.signal_gpu)
    end

    # per-row kernels with scalar σ_v
    i0 = bcmem.Nλ ÷ 2 + 1
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    ts2 = (32, 32)

    fill!(bcmem.conv_gpu, zero(T))
    bs_k = (cld(bcmem.Nλ, ts2[1]), cld(BNatm, ts2[2]))
    μ_v_off = Int32(tile_offset * Natm)
    @cuda threads=ts2 blocks=bs_k kernel_to_dft_layout_2d_scalar_σ_gpu!(
        bcmem.conv_gpu, bcmem.xs_gpu, μ_v_batch, μ_v_off, σ_v, σ_floor,
        Int32(i0), Int32(bcmem.Nλ), Int32(bcmem.L))
    bcmem.conv_gpu ./= sum(bcmem.conv_gpu, dims=2)
    mul!(bcmem.kernel_ft_gpu, bcmem.plan_fwd_kernel, bcmem.conv_gpu)

    nfreq = size(bcmem.kernel_ft_gpu, 2)
    BNatm32 = Int32(BNatm)
    bs3 = (cld(nfreq, ts2[1]), cld(BNatm, ts2[2]))
    @cuda threads=ts2 blocks=bs3 batched_spectral_multiply!(
        bcmem.conv_ft_gpu, bcmem.signal_ft_gpu, bcmem.kernel_ft_gpu,
        Int32(Natm), BNatm32)

    mul!(bcmem.conv_gpu, bcmem.plan_bwd, bcmem.conv_ft_gpu)
    return @view bcmem.conv_gpu[1:BNatm, bcmem.pad_left+1:bcmem.pad_left+bcmem.Nλ]
end

function convolve_wavelength_axis_batched!(bcmem::BatchedMicroConvMem{T},
                                           xs::AA{T,1}, ys::AA{T,2},
                                           μ_v_batch::CA{T,1}, σ_v::CA{T,1},
                                           Bcur::Int; tile_offset::Int=0) where {T<:AF}
    Natm = bcmem.Natm
    BNatm = Bcur * Natm
    xs_h = xs isa CuArray ? Array(xs) : collect(T, xs)

    if !bcmem.doppler_ready
        _init_micro_params!(bcmem, xs_h)
    end

    if !bcmem.signal_cached
        copyto!(bcmem.ys_gpu, ys)
        ts = (32, 32)
        bs = (cld(Natm, ts[1]), cld(bcmem.L, ts[2]))
        @cuda threads=ts blocks=bs pad_signal!(bcmem.signal_gpu, bcmem.ys_gpu,
                                               bcmem.Nλ, bcmem.pad_left, bcmem.pad_right)
        mul!(bcmem.signal_ft_gpu, bcmem.plan_fwd, bcmem.signal_gpu)
    end

    # per-row kernels with vector σ_v (tiled across tiles)
    i0 = bcmem.Nλ ÷ 2 + 1
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    ts2 = (32, 32)

    fill!(bcmem.conv_gpu, zero(T))
    bs_k = (cld(bcmem.Nλ, ts2[1]), cld(BNatm, ts2[2]))
    σ_v_tiled = repeat(σ_v, Bcur)
    μ_v_off = Int32(tile_offset * Natm)
    @cuda threads=ts2 blocks=bs_k kernel_to_dft_layout_2d_gpu!(
        bcmem.conv_gpu, bcmem.xs_gpu, μ_v_batch, μ_v_off, σ_v_tiled, σ_floor,
        Int32(i0), Int32(bcmem.Nλ), Int32(bcmem.L))
    bcmem.conv_gpu ./= sum(bcmem.conv_gpu, dims=2)
    mul!(bcmem.kernel_ft_gpu, bcmem.plan_fwd_kernel, bcmem.conv_gpu)

    nfreq = size(bcmem.kernel_ft_gpu, 2)
    BNatm32 = Int32(BNatm)
    bs3 = (cld(nfreq, ts2[1]), cld(BNatm, ts2[2]))
    @cuda threads=ts2 blocks=bs3 batched_spectral_multiply!(
        bcmem.conv_ft_gpu, bcmem.signal_ft_gpu, bcmem.kernel_ft_gpu,
        Int32(Natm), BNatm32)

    mul!(bcmem.conv_gpu, bcmem.plan_bwd, bcmem.conv_ft_gpu)
    return @view bcmem.conv_gpu[1:BNatm, bcmem.pad_left+1:bcmem.pad_left+bcmem.Nλ]
end
