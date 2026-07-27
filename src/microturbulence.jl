"""
    convolve_wavelength_axis(xs, ys, v_los, v_mic)

Convolve each row of `ys` with a Gaussian kernel that models microturbulent broadening
and a Doppler shift, using FFT convolution. The kernel width is wavelength-dependent
(constant in velocity units): σ(x) = x·v_mic/c.

Scalar `v_los` and `v_mic` apply the same kernel to every row; vectors specify per-row
values. The GPU implementation (`convolve_wavelength_axis_gpu`) builds the same
kernel on device, so CPU and GPU agree to floating-point precision.

Padding is sized from the `v_los`/`v_mic` passed, so a large Doppler shift cannot wrap the
padded linear convolution.
"""
function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, v_los::T, v_mic::T) where {T<:AF}
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))

    σ(x) = max(x * (v_mic / c_ms), σ_floor)
    g(x, n) = exp(-((x - n) / σ(x))^2.0)

    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    λc = (v_los / c_ms) * λ0 + λ0

    kernel = g.(xs, λc)
    s = sum(kernel)
    kernel ./= ifelse(iszero(s), one(T), s)
    iszero(s) && @warn "Doppler kernel underflowed (zero-sum); convolved αs set to zero. v_los probably exceeds wavelength window." maxlog=3

    # kernel reaches |v_los| (shift) + ~3·v_mic (width); pad for both
    Npad = conv_npad_for_velocity(λ0, Δλ, conv_kernel_vmax(v_los, zero(T), v_mic))
    return _padded_convolve(collect(T, ys), kernel; Npad=Npad)
end

function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, v_los::AA{T,1}, v_mic::AA{T,1}) where {T<:AF}
    Nλ = length(xs)
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]

    # pad for the widest per-row kernel: max|v_los| (shift) + ~3·max(v_mic) (width)
    Npad = conv_npad_for_velocity(λ0, Δλ,
                                  conv_kernel_vmax(maximum(abs, v_los), zero(T), v_mic))
    L, _, pad_left, _ = _conv_mem_geometry(Nλ, Npad)

    ys_out = zeros(T, size(ys))
    kbuf = zeros(T, L)
    kvec = Vector{T}(undef, Nλ)
    sig = zeros(T, L)
    n_underflow = 0

    for t in axes(ys, 1)
        σ(x) = max(x * (v_mic[t] / c_ms), σ_floor)
        g(x, n) = exp(-((x - n) / σ(x))^2.0)
        λc = (v_los[t] / c_ms) * λ0 + λ0

        @inbounds for j in 1:Nλ
            kvec[j] = g(xs[j], λc)
        end
        s = sum(kvec)
        kvec ./= ifelse(iszero(s), one(T), s)
        n_underflow += Int(iszero(s))

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
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow of $(size(ys, 1)) row(s); those rows set to zero. v_los probably exceeds wavelength window." maxlog=3
    return ys_out
end

# ── CPU in-place (disk integration) ─────────────────────────────────────────

"""
    _convolve_micro_inplace!(out, xs, ys, v_los, v_mic, ws)

In-place microturbulent broadening using pre-allocated [`CPUTileWorkspace`](@ref)
buffers.

Scalar `v_los` and `v_mic` build one kernel and apply it to all atmosphere layers
(the common case in disk integration). Vector arguments build per-row kernels.
"""
function _convolve_micro_inplace!(out::AA{T,2}, xs::AA{T,1}, ys::AA{T,2},
                                  v_los::T, v_mic::T,
                                  ws::CPUTileWorkspace) where T<:AF
    Nλ = ws.Nλ
    Natm = size(ys, 1)
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]
    λc = (v_los / c_ms) * λ0 + λ0

    kvec = Vector{T}(undef, Nλ)
    @inbounds for j in 1:Nλ
        σx = max(xs[j] * (v_mic / c_ms), σ_floor)
        kvec[j] = exp(-((xs[j] - λc) / σx)^2.0)
    end
    s = sum(kvec)
    kvec ./= ifelse(iszero(s), one(T), s)
    iszero(s) && @warn "Doppler kernel underflowed (zero-sum); convolved αs set to zero for all $Natm layer(s). v_los probably exceeds wavelength window." maxlog=3

    _kernel_to_dft_layout!(ws.kernel_real, kvec, i0)
    mul!(ws.kernel_ft, ws.fft_plan, ws.kernel_real)
    _apply_fft_kernel!(out, ys, ws.kernel_ft, ws, Natm)
    return nothing
end

function _convolve_micro_inplace!(out::AA{T,2}, xs::AA{T,1}, ys::AA{T,2},
                                  v_los::AA{T,1}, v_mic::AA{T,1},
                                  ws::CPUTileWorkspace) where T<:AF
    Nλ = ws.Nλ
    Natm = size(ys, 1)
    Δλ = median(diff(xs))
    σ_floor = T(max(eps(T) * mean(xs), T(0.25) * Δλ))
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]
    kvec = Vector{T}(undef, Nλ)
    n_underflow = 0

    for t in 1:Natm
        λc = (v_los[t] / c_ms) * λ0 + λ0
        @inbounds for j in 1:Nλ
            σx = max(xs[j] * (v_mic[t] / c_ms), σ_floor)
            kvec[j] = exp(-((xs[j] - λc) / σx)^2.0)
        end
        s = sum(kvec)
        kvec ./= ifelse(iszero(s), one(T), s)
        n_underflow += Int(iszero(s))

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
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow of $Natm row(s); those rows set to zero. v_los probably exceeds wavelength window." maxlog=3
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

# Build a single kernel in DFT layout (scalar v_los, v_mic).
function kernel_to_dft_layout_1d_gpu!(kbuf, xs, λ0, v_los_val, v_mic_val, σ_floor, i0, Nλ, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > Nλ && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    σx = max(xj * (v_mic_val / T(c_ms)), σ_floor)
    # avoid catastrophic cancellation: (xj - λ0) and (v_los/c)*λ0 are both small
    Δx = (xj - λ0) - (v_los_val / T(c_ms)) * λ0
    val = exp(-(Δx / σx)^2)
    d = j - i0
    idx = d >= 0 ? d + 1 : L + d + 1
    @inbounds kbuf[idx] = val
    return nothing
end

# Build per-row kernels in DFT layout (vector v_los, vector v_mic).
# Natm_v is the period for v_mic indexing: v_mic[(row-1) % Natm_v + 1].
# When length(v_mic) == Nrows the modulo is a no-op; when Nrows = B*Natm
# the Natm-length v_mic wraps across batched tiles without tiling allocation.
function kernel_to_dft_layout_2d_gpu!(kbuf, xs, v_los, v_los_off, v_mic, σ_floor, i0, Nλ, L, Natm_v, BNatm)
    j   = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # guard on BNatm (the active batch height), not size(kbuf,1): the launch ceil-rounds to
    # the block size, and over-spawned rows would read v_los past its end on the final batch.
    (row > BNatm || j > Nλ) && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    λ0 = @inbounds xs[i0]
    mic_idx = (row - Int32(1)) % Natm_v + Int32(1)
    σx = max(xj * (@inbounds v_mic[mic_idx]) / T(c_ms), σ_floor)
    Δx = (xj - λ0) - (@inbounds v_los[v_los_off + row]) / T(c_ms) * λ0
    val = exp(-(Δx / σx)^2)
    d = j - i0
    idx = d >= 0 ? d + 1 : L + d + 1
    @inbounds kbuf[row, idx] = val
    return nothing
end

# Build per-row kernels in DFT layout (vector v_los, scalar v_mic).
function kernel_to_dft_layout_2d_scalar_v_mic_gpu!(kbuf, xs, v_los, v_los_off, v_mic_val, σ_floor, i0, Nλ, L, BNatm)
    j   = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # guard on BNatm (the active batch height), not size(kbuf,1): the launch ceil-rounds to
    # the block size, and over-spawned rows would read v_los past its end on the final batch.
    (row > BNatm || j > Nλ) && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    λ0 = @inbounds xs[i0]
    σx = max(xj * (v_mic_val / T(c_ms)), σ_floor)
    Δx = (xj - λ0) - (@inbounds v_los[v_los_off + row]) / T(c_ms) * λ0
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

# NB: xs_gpu may be repurposed as scratch by macro kernels (e.g. hirano.jl stores
# frequency-domain σ values there). xs_cpu is the authoritative wavelength grid
# after initialization; never re-read xs_gpu to recover the grid.
function _init_micro_params!(cmem, xs_h::AbstractVector{T}) where T<:AF
    copyto!(cmem.xs_gpu, xs_h)
    cmem.xs_cpu = collect(T, xs_h)
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

# build 1D kernel FT (scalar v_los, v_mic → single kernel, broadcast)
function _build_kernel_ft_1d!(cmem, xs_h::AbstractVector{T}, v_los_val::T, v_mic_val::T) where T<:AF
    Nλ = cmem.Nλ; L = cmem.L
    i0 = Nλ ÷ 2 + 1
    λ0 = xs_h[i0]
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    fill!(cmem.kr_1d, zero(T))
    ts = (256,); bs = (cld(Nλ, ts[1]),)
    @cuda threads=ts blocks=bs kernel_to_dft_layout_1d_gpu!(
        cmem.kr_1d, cmem.xs_gpu, T(λ0), v_los_val, v_mic_val, σ_floor,
        Int32(i0), Int32(Nλ), Int32(L))
    normval = CUDA.sum(cmem.kr_1d)
    cmem.kr_1d ./= ifelse(iszero(normval), one(T), normval)
    iszero(normval) && @warn "Doppler kernel underflowed (zero-sum) in 1D scalar path; convolved αs set to zero for ALL atmosphere layers. v_los probably exceeds wavelength window." maxlog=3
    mul!(cmem.kernel_row_ft_1d, cmem.plan_fwd_1d, cmem.kr_1d)
    return nothing
end

# build per-row kernels into conv_gpu, normalize, batch-FFT into kernel_ft_gpu
function _build_per_row_kernels!(cmem, xs_h::AbstractVector{T},
                                  v_los::CA{T,1}, v_mic::T) where T<:AF
    i0 = cmem.Nλ ÷ 2 + 1
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    Nrows = size(cmem.conv_gpu, 1)

    fill!(cmem.conv_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(Nrows, ts[2]))
    @cuda threads=ts blocks=bs kernel_to_dft_layout_2d_scalar_v_mic_gpu!(
        cmem.conv_gpu, cmem.xs_gpu, v_los, Int32(0), v_mic, σ_floor,
        Int32(i0), Int32(cmem.Nλ), Int32(cmem.L), Int32(Nrows))
    row_sums = sum(cmem.conv_gpu, dims=2)
    cmem.conv_gpu ./= ifelse.(iszero.(row_sums), one(T), row_sums)
    n_underflow = Int(CUDA.sum(iszero.(row_sums)))
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow row(s); those rows set to zero. v_los probably exceeds wavelength window." maxlog=3
    return nothing
end

function _build_per_row_kernels!(cmem, xs_h::AbstractVector{T},
                                  v_los::CA{T,1}, v_mic::CA{T,1}) where T<:AF
    i0 = cmem.Nλ ÷ 2 + 1
    Δλ = median(diff(xs_h))
    σ_floor = T(max(eps(T) * mean(xs_h), T(0.25) * Δλ))
    Nrows = size(cmem.conv_gpu, 1)

    fill!(cmem.conv_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(Nrows, ts[2]))
    @cuda threads=ts blocks=bs kernel_to_dft_layout_2d_gpu!(
        cmem.conv_gpu, cmem.xs_gpu, v_los, Int32(0), v_mic, σ_floor,
        Int32(i0), Int32(cmem.Nλ), Int32(cmem.L), Int32(Nrows), Int32(Nrows))
    row_sums = sum(cmem.conv_gpu, dims=2)
    cmem.conv_gpu ./= ifelse.(iszero.(row_sums), one(T), row_sums)
    n_underflow = Int(CUDA.sum(iszero.(row_sums)))
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow row(s); those rows set to zero. v_los probably exceeds wavelength window." maxlog=3
    return nothing
end

# IRFFT + return valid region view
function _irfft_and_extract(cmem)
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)
    return @view cmem.conv_gpu[:, cmem.pad_left+1:cmem.pad_left+cmem.Nλ]
end

# ── GPU public API: convolve_wavelength_axis_gpu ─────────────────────────────

"""
    convolve_wavelength_axis_gpu(cmem, xs, ys, v_los, v_mic)

GPU implementation of [`convolve_wavelength_axis`](@ref). Builds the same real-space
kernel with wavelength-dependent σ(x) = x·v_mic/c as the CPU, matching exactly.

Scalar `v_los` and `v_mic` build one kernel broadcast to all rows. Vector arguments
build per-row kernels via batch R2C FFT.

See also: [`convolve_wavelength_axis`](@ref)
"""
# scalar v_los, scalar v_mic — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, v_los::T, v_mic::T) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_kernel_ft_1d!(cmem, xs_h, v_los, v_mic)
    kft = reshape(cmem.kernel_row_ft_1d, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft
    return _irfft_and_extract(cmem)
end

# scalar v_los, scalar v_mic — device xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory,
                                      xs_d::CuArray{T,1}, ys_d::CuArray{T,2},
                                      v_los::T, v_mic::T) where {T<:AF}
    if !cmem.doppler_ready
        _init_micro_params!(cmem, Array(xs_d))
    end
    _pad_and_fft_signal!(cmem, ys_d)
    _build_kernel_ft_1d!(cmem, cmem.xs_cpu, v_los, v_mic)
    kft = reshape(cmem.kernel_row_ft_1d, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft
    return _irfft_and_extract(cmem)
end

# vector v_los, scalar v_mic — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, v_los::CA{T,1}, v_mic::T) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_per_row_kernels!(cmem, xs_h, v_los, v_mic)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# vector v_los, scalar v_mic — device xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory,
                                      xs_d::CuArray{T,1}, ys_d::CuArray{T,2},
                                      v_los::CA{T,1}, v_mic::T) where {T<:AF}
    if !cmem.doppler_ready
        _init_micro_params!(cmem, Array(xs_d))
    end
    _pad_and_fft_signal!(cmem, ys_d)
    _build_per_row_kernels!(cmem, cmem.xs_cpu, v_los, v_mic)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# vector v_los, vector v_mic — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, v_los::CA{T,1}, v_mic::CA{T,1}) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_per_row_kernels!(cmem, xs_h, v_los, v_mic)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# vector v_los, vector v_mic — device xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory,
                                      xs_d::CuArray{T,1}, ys_d::CuArray{T,2},
                                      v_los::CA{T,1}, v_mic::CA{T,1}) where {T<:AF}
    if !cmem.doppler_ready
        _init_micro_params!(cmem, Array(xs_d))
    end
    _pad_and_fft_signal!(cmem, ys_d)
    _build_per_row_kernels!(cmem, cmem.xs_cpu, v_los, v_mic)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract(cmem)
end

# ── GPU batched API ──────────────────────────────────────────────────────────

"""
    convolve_wavelength_axis_batched!(bcmem, xs, ys, v_los_batch, v_mic, Bcur)

Batched Doppler convolution for `Bcur` tiles simultaneously. The absorption signal
`ys` (Natm × Nλ) is shared across tiles; `v_los_batch` (Bcur*Natm) provides per-tile
velocities. Scalar `v_mic` avoids tiling allocations; vector `v_mic` is tiled across tiles.
Returns a view of the valid region `(Bcur*Natm, Nλ)`.
"""
function convolve_wavelength_axis_batched!(bcmem::BatchedMicroConvMem{T},
                                           xs::AA{T,1}, ys::AA{T,2},
                                           v_los_batch::CA{T,1}, v_mic::T,
                                           Bcur::Int; tile_offset::Int=0) where {T<:AF}
    Natm = bcmem.Natm
    BNatm = Bcur * Natm

    if !bcmem.doppler_ready
        xs_h = xs isa CuArray ? Array(xs) : collect(T, xs)
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

    # per-row kernels with scalar v_mic
    i0 = bcmem.Nλ ÷ 2 + 1
    Δλ = median(diff(bcmem.xs_cpu))
    σ_floor = T(max(eps(T) * mean(bcmem.xs_cpu), T(0.25) * Δλ))
    ts2 = (32, 32)

    fill!(bcmem.conv_gpu, zero(T))
    bs_k = (cld(bcmem.Nλ, ts2[1]), cld(BNatm, ts2[2]))
    v_los_off = Int32(tile_offset * Natm)
    @cuda threads=ts2 blocks=bs_k kernel_to_dft_layout_2d_scalar_v_mic_gpu!(
        bcmem.conv_gpu, bcmem.xs_gpu, v_los_batch, v_los_off, v_mic, σ_floor,
        Int32(i0), Int32(bcmem.Nλ), Int32(bcmem.L), Int32(BNatm))
    row_sums = sum(bcmem.conv_gpu, dims=2)
    bcmem.conv_gpu ./= ifelse.(iszero.(row_sums), one(T), row_sums)
    # Restrict underflow count to the first BNatm rows; trailing rows of
    # bcmem.conv_gpu are pre-zeroed by `fill!` and are not active.
    n_underflow = Int(CUDA.sum(iszero.(@view row_sums[1:BNatm, :])))
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow of $BNatm active row(s); those rows set to zero. v_los probably exceeds wavelength window." maxlog=3
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
                                           v_los_batch::CA{T,1}, v_mic::CA{T,1},
                                           Bcur::Int; tile_offset::Int=0) where {T<:AF}
    Natm = bcmem.Natm
    BNatm = Bcur * Natm

    if !bcmem.doppler_ready
        xs_h = xs isa CuArray ? Array(xs) : collect(T, xs)
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

    # per-row kernels with vector v_mic (modular indexing wraps across tiles)
    i0 = bcmem.Nλ ÷ 2 + 1
    Δλ = median(diff(bcmem.xs_cpu))
    σ_floor = T(max(eps(T) * mean(bcmem.xs_cpu), T(0.25) * Δλ))
    ts2 = (32, 32)

    fill!(bcmem.conv_gpu, zero(T))
    bs_k = (cld(bcmem.Nλ, ts2[1]), cld(BNatm, ts2[2]))
    v_los_off = Int32(tile_offset * Natm)
    @cuda threads=ts2 blocks=bs_k kernel_to_dft_layout_2d_gpu!(
        bcmem.conv_gpu, bcmem.xs_gpu, v_los_batch, v_los_off, v_mic, σ_floor,
        Int32(i0), Int32(bcmem.Nλ), Int32(bcmem.L), Int32(Natm), Int32(BNatm))
    row_sums = sum(bcmem.conv_gpu, dims=2)
    bcmem.conv_gpu ./= ifelse.(iszero.(row_sums), one(T), row_sums)
    # Restrict underflow count to the first BNatm rows; trailing rows of
    # bcmem.conv_gpu are pre-zeroed by `fill!` and are not active.
    n_underflow = Int(CUDA.sum(iszero.(@view row_sums[1:BNatm, :])))
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow of $BNatm active row(s); those rows set to zero. v_los probably exceeds wavelength window." maxlog=3
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
