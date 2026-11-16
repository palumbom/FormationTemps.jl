# Lower bound on the microturbulent Gaussian width: the larger of the representable
# resolution at this wavelength and a quarter pixel. A function of the grid alone, so it is
# constant for the lifetime of a convolution-memory struct. The two-argument form is for
# callers that already hold Δλ for their padding calculation.
_sigma_floor(xs::AA{T,1}, Δλ::T) where {T<:AF} =
    T(max(eps(T) * mean(xs), T(0.25) * Δλ))

_sigma_floor(xs::AA{T,1}) where {T<:AF} = _sigma_floor(xs, median(diff(xs)))

# Normalize a broadening kernel (host or device vector) to unit sum in place. A kernel that
# underflowed to all zeros stays zero, so its convolution contributes nothing, rather than
# dividing 0/0 into NaN; the warning names the kernel. For a nonzero sum this is the plain
# division, bit for bit.
function _normalize_kernel!(kernel::AA, name::AbstractString)
    s = sum(kernel)
    kernel ./= ifelse(iszero(s), one(s), s)
    iszero(s) && @warn "$name kernel underflowed (zero-sum); its convolution is set to zero." maxlog=3
    return kernel
end

# σ_floor for a caller-supplied grid: the cached value when the caller passed the grid the
# struct was initialised with, recomputed otherwise. Identity, not equality — the host-`xs`
# convolution overloads pass a fresh `collect` and must keep recomputing, which is what makes
# the cache bit-identical for every caller. The assertion matters because σ_floor is zero
# until _init_micro_params! runs, and a zero floor divides by zero in the Gaussian rather
# than erroring in median(diff(...)) the way an empty xs_cpu used to.
function _sigma_floor_cached(cmem, xs_h::AA{T,1}) where {T<:AF}
    @assert cmem.doppler_ready "σ_floor is only valid after _init_micro_params!"
    return xs_h === cmem.xs_cpu ? T(cmem.σ_floor) : _sigma_floor(xs_h)
end

"""
    convolve_wavelength_axis(xs, ys, v_los, v_mic)

Convolve each row of `ys` with a Gaussian kernel that models microturbulent broadening
and a Doppler shift, using FFT convolution. The kernel width is wavelength-dependent
(constant in velocity units): σ(x) = x·v_mic/c.

Scalar `v_los` and `v_mic` apply the same kernel to every row; vectors specify per-row
values. The GPU implementation (`convolve_wavelength_axis_gpu`) builds the same
kernel on device, so CPU and GPU agree to floating-point precision while the kernel lies inside
the window's taps. Here the kernel is sampled on the window's taps only and renormalized, so a
kernel that reaches past them is truncated; the GPU renders such a row by direct convolution
over the kernel's whole support instead.

Padding is sized from the `v_los`/`v_mic` passed, so a large Doppler shift cannot wrap the
padded linear convolution.
"""
function convolve_wavelength_axis(xs::AA{T,1}, ys::AA{T,2}, v_los::T, v_mic::T) where {T<:AF}
    # one Δλ, two consumers: the σ_floor below and conv_npad_for_velocity further down.
    # Both need the median, not step(xs) — on a non-uniform grid those differ, and
    # substituting one moves the kernel width and the derived padding together.
    Δλ = median(diff(xs))
    σ_floor = _sigma_floor(xs, Δλ)

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
    # one Δλ, two consumers: the σ_floor below and conv_npad_for_velocity further down.
    # Both need the median, not step(xs) — on a non-uniform grid those differ, and
    # substituting one moves the kernel width and the derived padding together.
    Δλ = median(diff(xs))
    σ_floor = _sigma_floor(xs, Δλ)
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
    σ_floor = _sigma_floor(xs)
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
    σ_floor = _sigma_floor(xs)
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
function kernel_to_dft_layout_1d_gpu!(kbuf, xs, dx, λ0, v_los_val, v_mic_val, σ_floor, i0, Nλ, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > Nλ && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    σx = max(xj * (v_mic_val / T(c_ms)), σ_floor)
    # dx[j] is the offset of sample j from the window center and (v_los/c)*λ0 the shift; both
    # are small, where subtracting λc from xj would cancel
    Δx = (@inbounds dx[j]) - (v_los_val / T(c_ms)) * λ0
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
function kernel_to_dft_layout_2d_gpu!(kbuf, xs, dx, v_los, v_los_off, v_mic, σ_floor, i0, Nλ, L, Natm_v, BNatm)
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
    Δx = (@inbounds dx[j]) - (@inbounds v_los[v_los_off + row]) / T(c_ms) * λ0
    val = exp(-(Δx / σx)^2)
    d = j - i0
    idx = d >= 0 ? d + 1 : L + d + 1
    @inbounds kbuf[row, idx] = val
    return nothing
end

# Build per-row kernels in DFT layout (vector v_los, scalar v_mic).
function kernel_to_dft_layout_2d_scalar_v_mic_gpu!(kbuf, xs, dx, v_los, v_los_off, v_mic_val, σ_floor, i0, Nλ, L, BNatm)
    j   = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # guard on BNatm (the active batch height), not size(kbuf,1): the launch ceil-rounds to
    # the block size, and over-spawned rows would read v_los past its end on the final batch.
    (row > BNatm || j > Nλ) && return nothing
    T = eltype(kbuf)
    xj = @inbounds xs[j]
    λ0 = @inbounds xs[i0]
    σx = max(xj * (v_mic_val / T(c_ms)), σ_floor)
    Δx = (@inbounds dx[j]) - (@inbounds v_los[v_los_off + row]) / T(c_ms) * λ0
    val = exp(-(Δx / σx)^2)
    d = j - i0
    idx = d >= 0 ? d + 1 : L + d + 1
    @inbounds kbuf[row, idx] = val
    return nothing
end

# broadcast shared signal_ft across per-row kernel FTs, and zero the rows past BNatm. At an odd
# padded length cuFFT's C2R transforms rows in pairs (2k-1, 2k), so with an odd BNatm the last
# active row is transformed with row BNatm + 1, which would otherwise hold a stale spectrum
# from an earlier batch (or the macro buffers overlaid on this memory). Launch over every row.
function batched_spectral_multiply!(conv_ft, signal_ft, kernel_ft, Natm, BNatm)
    f_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i     = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    f_idx <= size(conv_ft, 2) || return nothing
    if i <= BNatm
        i_local = ((i - 1) % Natm) + 1
        @inbounds conv_ft[i, f_idx] = signal_ft[i_local, f_idx] * kernel_ft[i, f_idx]
    elseif i <= size(conv_ft, 1)
        @inbounds conv_ft[i, f_idx] = zero(eltype(conv_ft))
    end
    return nothing
end

# ── GPU host helpers ─────────────────────────────────────────────────────────

# NB: xs_gpu may be repurposed as scratch by macro kernels (e.g. hirano.jl stores
# frequency-domain σ values there). xs_cpu is the authoritative wavelength grid
# after initialization; never re-read xs_gpu to recover the grid.
#
# `xs_exact` is the same grid at the precision it was generated at (Float64 from Korg); the
# kernel offsets in dx_gpu are differenced from it (see _kernel_offsets). With the default
# `xs_exact = xs_h` they are those of the rounded grid.
function _init_micro_params!(cmem, xs_h::AbstractVector{T},
                             xs_exact::AbstractVector{<:AF}=xs_h) where T<:AF
    copyto!(cmem.dx_gpu, _kernel_offsets(xs_h, xs_exact))
    copyto!(cmem.xs_gpu, xs_h)
    cmem.xs_cpu = collect(T, xs_h)
    # from xs_cpu, not xs_h, so the cached value is by construction what
    # _sigma_floor(cmem.xs_cpu) returns at every later call site
    cmem.σ_floor = _sigma_floor(cmem.xs_cpu)
    cmem.doppler_ready = true
    return nothing
end

# pad signal + forward FFT (host-array ys). ys is copied to ys_gpu on every call, also when the
# FFT is cached: the direct-convolution fallback reads ys_gpu, and a ConvolutionMemory shared by
# several opacities keeps there whichever filled the cache last.
function _pad_and_fft_signal!(cmem, ys::AA{T,2}, xs_h::AbstractVector{T}) where T<:AF
    if !cmem.doppler_ready
        _init_micro_params!(cmem, xs_h)
    end
    copyto!(cmem.ys_gpu, ys)
    if !cmem.signal_cached
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
    σ_floor = _sigma_floor_cached(cmem, xs_h)
    fill!(cmem.kr_1d, zero(T))
    ts = (256,); bs = (cld(Nλ, ts[1]),)
    @cuda threads=ts blocks=bs kernel_to_dft_layout_1d_gpu!(
        cmem.kr_1d, cmem.xs_gpu, cmem.dx_gpu, T(λ0), v_los_val, v_mic_val, σ_floor,
        Int32(i0), Int32(Nλ), Int32(L))
    normval = CUDA.sum(cmem.kr_1d)
    cmem.kr_1d ./= ifelse(iszero(normval), one(T), normval)
    iszero(normval) && @warn "Doppler kernel underflowed (zero-sum) in 1D scalar path: its center lies far outside the wavelength window, and every layer is rendered by direct convolution with the window's edge opacity. Check that v_los is in m/s." maxlog=3
    mul!(cmem.kernel_row_ft_1d, cmem.plan_fwd_1d, cmem.kr_1d)
    return nothing
end

# Doppler-kernel underflow is a configuration error — v_los in cm/s instead of m/s, or a
# velocity past the padded window — not a transient: it fires on the first renders or never.
# Counting the zero-sum rows needs a device→host read that drains the pipeline, so the read is
# paced: every build for the first `_UNDERFLOW_EAGER_BUILDS` on a given memory object, then one
# build in every `_UNDERFLOW_CHECK_STRIDE`. The guard itself (`ifelse(iszero(s), one(T), s)`)
# is unconditional; only the diagnostic is sampled. The batched path is not paced — per-tile
# v_los varies, so a late batch can be the first to underflow.
const _UNDERFLOW_EAGER_BUILDS = 64
const _UNDERFLOW_CHECK_STRIDE = 4096

_should_check_underflow(n::Int) =
    n <= _UNDERFLOW_EAGER_BUILDS || iszero(n % _UNDERFLOW_CHECK_STRIDE)

# Paced underflow diagnostic; increments the build counter as a side effect.
# The readback must stay INSIDE the branch. It is a blocking device→host copy, it is the
# entire cost this pacing exists to remove, and it is what makes the render impossible to
# capture into a CUDA graph. "Flattening" this by computing n_underflow unconditionally and
# gating only the @warn restores ~23 μs per render, breaks capture, and still passes every
# pacing test — test/test_graph_capture.jl is the only gate that catches it.
# The capture query is second so it is reached only on a build the policy says is due.
# A caller whose v_los is data-dependent per call rather than fixed for the run (a cube column
# sweep, not an MCMC) can force a check on any call by zeroing n_kernel_builds first.
# Not thread-safe: n_kernel_builds is a plain read-modify-write, and a cmem is owned by one
# task in every current caller.
function _maybe_report_underflow!(cmem)
    cmem.n_kernel_builds += 1
    _should_check_underflow(cmem.n_kernel_builds) || return nothing
    CUDA.is_capturing() && return nothing
    n_underflow = _count_zero_rows(cmem.conv_gpu, size(cmem.conv_gpu, 1))
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow row(s): their kernel centers lie far outside the wavelength window, and those rows are rendered by direct convolution with the window's edge opacity. Check that v_los is in m/s." maxlog=3
    return nothing
end

# Rows among the first `nrows` of the per-row kernels in `A` that are all zeros. The kernels are
# non-negative, so a row is zero exactly when its maximum is, and a maximum is exact in any
# reduction order. Allocates and blocks on a device→host read: call it only where the underflow
# diagnostic is due.
_count_zero_rows(A::CuMatrix, nrows::Int) =
    Int(CUDA.sum(iszero.(view(maximum(A, dims=2), 1:nrows, :))))

# Lanes per row of `normalize_rows_kernel!`; a power of two. Part of the summation order, so
# changing it moves the last bit of every per-row kernel.
const _ROW_NORM_LANES = 64
# Rows per block of `normalize_rows_kernel!`. Not part of the summation order: it only sets how
# many adjacent rows a warp reads together and how many blocks the work spreads over.
const _ROW_NORM_ROWS = 8

# elements j, j + nl, j + 2nl, j + 3nl of a row, read together so that several loads are in flight
@inline _row_run4(A, row, j, nl) =
    @inbounds (A[row, j], A[row, j + nl], A[row, j + Int32(2) * nl], A[row, j + Int32(3) * nl])

# Scale each of the first `Nrows` rows of `A` to unit sum in place, by the reciprocal of its sum;
# a row that sums to zero is left as it is. A row whose sum is so small that its reciprocal
# overflows (every element subnormal) is first scaled, together with its sum, by the power of two
# 1/eps(T): that is exact and makes the sum normal, so the row also reaches unit sum. Every other
# row is multiplied by the reciprocal of its own sum with no scaling. Elements are non-negative,
# so none exceeds its row sum and every scaled element stays finite.
# A block holds blockDim().x consecutive rows and blockDim().y lanes per row. Lane l of a row
# adds the row's elements l, l + nl, l + 2nl, … in that order and the nl partial sums are
# combined by a fixed binary tree, so the order of the additions depends on the row length and
# the lane count alone — not on how many rows are normalized together, on the array's shape, or
# on a reduction library.
function normalize_rows_kernel!(A::CDM{T}, Nrows::Int32, L::Int32) where T<:AF
    nr = Int32(blockDim().x)
    nl = Int32(blockDim().y)
    r = Int32(threadIdx().x)
    lane = Int32(threadIdx().y)
    row = (Int32(blockIdx().x) - Int32(1)) * nr + r
    partials = CuDynamicSharedArray(T, Int(nr) * Int(nl))
    sidx = r + (lane - Int32(1)) * nr
    # a thread whose row is past Nrows does no work, but must still reach every barrier
    active = row <= Nrows
    span = Int32(3) * nl
    step = Int32(4) * nl

    acc = zero(T)
    if active
        j = lane
        while j + span <= L
            x1, x2, x3, x4 = _row_run4(A, row, j, nl)
            acc = (((acc + x1) + x2) + x3) + x4
            j += step
        end
        while j <= L
            acc += @inbounds A[row, j]
            j += nl
        end
    end
    @inbounds partials[sidx] = acc
    sync_threads()

    d = nl >> Int32(1)
    while d >= Int32(1)
        if lane <= d
            @inbounds partials[sidx] += partials[sidx + d * nr]
        end
        sync_threads()
        d >>= Int32(1)
    end

    if active
        s = @inbounds partials[r]
        s = ifelse(iszero(s), one(T), s)
        w = one(T) / s
        # pre must stay exactly one(T) unless 1/s overflows: (x * one(T)) * w is x * w bit for bit.
        # Multiply by pre before w: pre * w overflows for every row that takes the scaled branch.
        pre = one(T)
        if isinf(w)
            pre = inv(eps(T))
            w = one(T) / (s * pre)
        end
        j = lane
        while j + span <= L
            x1, x2, x3, x4 = _row_run4(A, row, j, nl)
            @inbounds begin
                iszero(x1) || (A[row, j] = (x1 * pre) * w)
                iszero(x2) || (A[row, j + nl] = (x2 * pre) * w)
                iszero(x3) || (A[row, j + Int32(2) * nl] = (x3 * pre) * w)
                iszero(x4) || (A[row, j + Int32(3) * nl] = (x4 * pre) * w)
            end
            j += step
        end
        while j <= L
            x = @inbounds A[row, j]
            iszero(x) || (@inbounds A[row, j] = (x * pre) * w)
            j += nl
        end
    end
    return nothing
end

# Normalize the first `Nrows` rows of the per-row kernels in `A`. Use this, not a library
# reduction: `sum(A, dims=2)` picks its summation grouping from the array's shape, so a row's
# normalization would depend on how many rows share the array, and it allocates.
function _normalize_rows!(A::CuMatrix{T}, Nrows::Int) where T<:AF
    nr = _ROW_NORM_ROWS
    nl = _ROW_NORM_LANES
    @cuda threads=(nr, nl) blocks=cld(Nrows, nr) shmem=nr * nl * sizeof(T) normalize_rows_kernel!(
        A, Int32(Nrows), Int32(size(A, 2)))
    return nothing
end

# ── GPU direct-convolution fallback ──────────────────────────────────────────
#
# The FFT convolution builds each Doppler kernel on the Nλ taps of the window, so it renders
# replicate padding only while the kernel lies inside that tap span. Past it the kernel is
# truncated and renormalized, which pins the line at the window edge, and once every tap
# underflows the row is zero. After the inverse FFT, one launch recomputes every row whose kernel
# reaches past the span by direct convolution: the same Gaussian over its whole support, each
# source index clamped to the window. Rows inside the domain are not written and keep the bits of
# the FFT convolution.

# Kernel widths that bound the domain: a row keeps its FFT convolution when its kernel center lies
# at least this many σx inside both ends of the tap span, where the kernel mass beyond the span,
# erfc(k)/2 per side, is below eps(Float32). It decides which rows the fallback recomputes, so a
# change moves the bits of every row that crosses the boundary.
const _DOPPLER_DOMAIN_SIGMAS = 3.7

# Threads per block of doppler_direct_kernel!; a power of two. Part of its arithmetic: the tail
# masses are reduced over this many lanes by a fixed binary tree.
const _DOPPLER_DIRECT_THREADS = 256

# Largest reach of the fallback kernel past either end of the window, in pixels. It binds only for
# a kernel millions of pixels wide.
const _DOPPLER_DIRECT_MAX_REACH = 1 << 24

# A per-row parameter: a device vector read at element off + row, or one value for every row.
@inline _row_value(v::CDV, off::Int32, row::Int32) = @inbounds v[off + row]
@inline _row_value(v::AF, ::Int32, ::Int32) = v

# A per-row parameter whose vector repeats with the given period, as v_mic across batched tiles.
@inline _row_value_mod(v::CDV, period::Int32, row::Int32) =
    @inbounds v[(row - Int32(1)) % period + Int32(1)]
@inline _row_value_mod(v::AF, ::Int32, ::Int32) = v

# Whether a row keeps its FFT convolution: its kernel center lies _DOPPLER_DOMAIN_SIGMAS kernel
# widths inside both ends of the tap span. The row's grid is elements goff+1 : goff+Nλ of xs and
# dx. The shift and the widths are those of the per-row kernel builders. False for a non-finite
# velocity, which the fallback then renders as NaN.
@inline function _doppler_row_in_domain(xs, dx, goff::Int32, i0::Int32, Nλ::Int32,
                                        v::T, vm::T, σ_floor::T) where T<:AF
    @inbounds begin
        s = v / T(c_ms) * xs[goff + i0]
        k = T(_DOPPLER_DOMAIN_SIGMAS)
        σ_lo = max(xs[goff + Int32(1)] * vm / T(c_ms), σ_floor)
        σ_hi = max(xs[goff + Nλ] * vm / T(c_ms), σ_floor)
        return (s - k * σ_lo >= dx[goff + Int32(1)]) & (s + k * σ_hi <= dx[goff + Nλ])
    end
end

# Unnormalized tap of a row's Doppler kernel at offset d from the window center i0, for shift s:
# the Gaussian of the per-row kernel builders. Inside the window the offset and the wavelength
# that sets σx are the grid's own (offsets from the Float64 grid); outside it they continue from
# the nearer end at the mean step Δλ.
@inline function _doppler_tap(xs, dx, goff::Int32, i0::Int32, Nλ::Int32, d::Int32,
                              s::T, vm::T, σ_floor::T, Δλ::T) where T<:AF
    j = i0 + d
    @inbounds if j < Int32(1)
        e = T(j - Int32(1)) * Δλ
        Δx = dx[goff + Int32(1)] + e
        x = xs[goff + Int32(1)] + e
    elseif j > Nλ
        e = T(j - Nλ) * Δλ
        Δx = dx[goff + Nλ] + e
        x = xs[goff + Nλ] + e
    else
        Δx = dx[goff + j]
        x = xs[goff + j]
    end
    σx = max(x * vm / T(c_ms), σ_floor)
    return exp(-((Δx - s) / σx)^2)
end

# Sum of x over the threads of the block by a fixed binary tree, so the result depends on the
# block size alone. Every thread of the block must call it; sh holds blockDim().x elements and is
# free again on return.
@inline function _block_tree_sum(sh, x::T) where T
    t = Int32(threadIdx().x)
    @inbounds sh[t] = x
    sync_threads()
    h = Int32(blockDim().x) >> Int32(1)
    while h >= Int32(1)
        if t <= h
            @inbounds sh[t] += sh[t + h]
        end
        sync_threads()
        h >>= Int32(1)
    end
    r = @inbounds sh[1]
    sync_threads()
    return r
end

# compensated (Kahan) addition of x to sum, with running compensation comp
@inline function _kahan_add(sum::T, comp::T, x::T) where T
    y = x - comp
    t = sum + y
    return t, (t - sum) - y
end

# One term k·x of a running sum in the fallback's tap loop: compensated, except at Float64, where
# a plain sum is accurate over the window's taps and the arithmetic is the cost. The Float64 form
# is an explicit fma, not muladd, so that every specialization of the kernel rounds alike.
@inline _fallback_mac(sum::T, comp::T, k::T, x::T) where T<:AF = _kahan_add(sum, comp, k * x)
@inline _fallback_mac(sum::Float64, comp::Float64, k::Float64, x::Float64) = (fma(k, x, sum), comp)

# One block per (row, chunk of _DOPPLER_DIRECT_THREADS output pixels); a block whose row is inside
# the domain returns at once. For a row outside it, writes out[row, out_off + i], i ∈ 1:Nλ:
#   Σ_d k(d) src[clamp(i - d, 1, Nλ)] / Σ_d k(d)
# over offsets d within sqrt(log(1/eps(T))) + 1 kernel widths of the kernel center. Offsets
# d ≥ Nλ read the first pixel for every output and d ≤ -Nλ the last, so they enter through their
# summed mass. The offsets in between enter the numerator tap by tap: the q-th tap of each tile
# into the ((q - 1) % 4 + 1)-th of four running sums (_fallback_mac), combined in a fixed order.
# Each of the three kernel masses (d ≥ Nλ, d ≤ -Nλ, in between) is summed per thread over that
# thread's taps and then reduced by the fixed tree. A kernel wholly past one end gives that end's
# opacity. The result is a function of the row's inputs and the block size alone, so a row has the
# same bits however many rows share the launch.
# The opacities of the row are row (row - 1) % src_period + 1 of src. Its grid is elements
# g*Nλ+1 : (g+1)*Nλ of xs and dx, and σ_floor element g+1 of a vector σ_floor, for
# g = (row - 1) ÷ grid_rows.
function doppler_direct_kernel!(out::CDM{T}, out_off::Int32, src::CDM{T}, src_period::Int32,
                                xs::CDV{T}, dx::CDV{T}, grid_rows::Int32, σ_floor,
                                v_los, v_los_off::Int32, v_mic, v_mic_period::Int32,
                                i0::Int32, Nλ::Int32) where T<:AF
    row = Int32(blockIdx().x)
    t = Int32(threadIdx().x)
    nt = Int32(blockDim().x)
    i = (Int32(blockIdx().y) - Int32(1)) * nt + t
    g = (row - Int32(1)) ÷ grid_rows
    goff = g * Nλ
    v = _row_value(v_los, v_los_off, row)
    vm = _row_value_mod(v_mic, v_mic_period, row)
    σf = _row_value(σ_floor, Int32(0), g + Int32(1))
    # the same for every thread of the block, so no thread is left waiting at a barrier
    _doppler_row_in_domain(xs, dx, goff, i0, Nλ, v, vm, σf) && return nothing

    r_src = (row - Int32(1)) % src_period + Int32(1)
    active = i <= Nλ
    λ0 = @inbounds xs[goff + i0]
    s = v / T(c_ms) * λ0
    Δλ = @inbounds (dx[goff + Nλ] - dx[goff + Int32(1)]) / T(Nλ - Int32(1))
    σc = max((λ0 + abs(s)) * vm / T(c_ms), σf)
    R = (sqrt(-log(eps(T))) + one(T)) * σc
    if !(isfinite(s) && isfinite(R))
        active && (@inbounds out[row, out_off + i] = T(NaN))
        return nothing
    end
    dlo_f = (s - R) / Δλ
    dhi_f = (s + R) / Δλ
    if dlo_f >= T(Nλ)
        active && (@inbounds out[row, out_off + i] = src[r_src, Int32(1)])
        return nothing
    elseif dhi_f <= -T(Nλ)
        active && (@inbounds out[row, out_off + i] = src[r_src, Nλ])
        return nothing
    end
    cap = T(Nλ) + T(_DOPPLER_DIRECT_MAX_REACH)
    dlo = unsafe_trunc(Int32, floor(max(dlo_f, -cap)))
    dhi = unsafe_trunc(Int32, ceil(min(dhi_f, cap)))

    sh = CuStaticSharedArray(T, _DOPPLER_DIRECT_THREADS)
    # the source pixels one tile of taps reads for this block's outputs. A row's pixels are
    # Natm apart in src, so reading them straight from global memory in the tap loop would
    # touch one cache line per thread per tap.
    seg = CuStaticSharedArray(T, 2 * _DOPPLER_DIRECT_THREADS)
    i_a = (Int32(blockIdx().y) - Int32(1)) * nt + Int32(1)

    # mass of the offsets that read the first pixel (d ≥ Nλ) or the last (d ≤ -Nλ) for every output
    m_hi = zero(T)
    d = max(dlo, Nλ) + t - Int32(1)
    while d <= dhi
        m_hi += _doppler_tap(xs, dx, goff, i0, Nλ, d, s, vm, σf, Δλ)
        d += nt
    end
    m_lo = zero(T)
    d = dlo + t - Int32(1)
    while d <= min(dhi, -Nλ)
        m_lo += _doppler_tap(xs, dx, goff, i0, Nλ, d, s, vm, σf, Δλ)
        d += nt
    end
    m_hi = _block_tree_sum(sh, m_hi)
    m_lo = _block_tree_sum(sh, m_lo)

    # the offsets in between, one tile of taps at a time. Four independent running sums, so the
    # loop is not one dependent chain.
    a = max(dlo, Int32(1) - Nλ)
    b = min(dhi, Nλ - Int32(1))
    num1 = zero(T)
    num2 = zero(T)
    num3 = zero(T)
    num4 = zero(T)
    cmp1 = zero(T)
    cmp2 = zero(T)
    cmp3 = zero(T)
    cmp4 = zero(T)
    m_mid = zero(T)
    base = a
    while base <= b
        dt = base + t - Int32(1)
        kt = dt <= b ? _doppler_tap(xs, dx, goff, i0, Nλ, dt, s, vm, σf, Δλ) : zero(T)
        m_mid += kt
        @inbounds sh[t] = kt
        # output i reads pixel i - d for tap d = base + q - 1; over the block's outputs and this
        # tile's taps that is seg[m] = src[clamp(j0 + m)], m ∈ 1:2nt-1, at m = p - q, p = t + nt
        j0 = i_a - base - nt
        @inbounds seg[t] = src[r_src, clamp(j0 + t, Int32(1), Nλ)]
        @inbounds seg[t + nt] = src[r_src, clamp(j0 + t + nt, Int32(1), Nλ)]
        sync_threads()
        if active
            n = min(nt, b - base + Int32(1))
            p = t + nt
            q = Int32(1)
            @inbounds while q + Int32(3) <= n
                num1, cmp1 = _fallback_mac(num1, cmp1, sh[q], seg[p - q])
                num2, cmp2 = _fallback_mac(num2, cmp2, sh[q + Int32(1)], seg[p - q - Int32(1)])
                num3, cmp3 = _fallback_mac(num3, cmp3, sh[q + Int32(2)], seg[p - q - Int32(2)])
                num4, cmp4 = _fallback_mac(num4, cmp4, sh[q + Int32(3)], seg[p - q - Int32(3)])
                q += Int32(4)
            end
            @inbounds if q <= n
                num1, cmp1 = _fallback_mac(num1, cmp1, sh[q], seg[p - q])
            end
            @inbounds if q + Int32(1) <= n
                num2, cmp2 = _fallback_mac(num2, cmp2, sh[q + Int32(1)], seg[p - q - Int32(1)])
            end
            @inbounds if q + Int32(2) <= n
                num3, cmp3 = _fallback_mac(num3, cmp3, sh[q + Int32(2)], seg[p - q - Int32(2)])
            end
        end
        sync_threads()
        base += nt
    end
    den = _block_tree_sum(sh, m_mid)
    if active
        num = (num1 + num2) + (num3 + num4)
        S = (m_lo + den) + m_hi
        acc = (num + @inbounds(src[r_src, Nλ]) * m_lo) + @inbounds(src[r_src, Int32(1)]) * m_hi
        @inbounds out[row, out_off + i] = acc / S
    end
    return nothing
end

# Recompute in place every row of `out` (columns out_off+1 : out_off+Nλ) among the first Nrows
# whose Doppler kernel reaches past the tap span; see doppler_direct_kernel! for the arguments.
# One launch, no allocation and no device→host read, so it can be captured into a CUDA graph.
function _doppler_direct_rows!(out::CA{T,2}, out_off::Int, src::CA{T,2}, src_period::Int,
                               xs::CA{T,1}, dx::CA{T,1}, grid_rows::Int, σ_floor,
                               v_los, v_los_off::Int, v_mic, v_mic_period::Int,
                               Nλ::Int, Nrows::Int) where T<:AF
    @assert Nλ >= 2 "the Doppler fallback needs at least two wavelengths"
    nt = _DOPPLER_DIRECT_THREADS
    @cuda threads=nt blocks=(Nrows, cld(Nλ, nt)) doppler_direct_kernel!(
        out, Int32(out_off), src, Int32(src_period), xs, dx, Int32(grid_rows), σ_floor,
        v_los, Int32(v_los_off), v_mic, Int32(v_mic_period), Int32(Nλ ÷ 2 + 1), Int32(Nλ))
    return nothing
end

# build per-row kernels into conv_gpu and normalize them
function _build_per_row_kernels!(cmem, xs_h::AbstractVector{T},
                                  v_los::CA{T,1}, v_mic::T) where T<:AF
    i0 = cmem.Nλ ÷ 2 + 1
    σ_floor = _sigma_floor_cached(cmem, xs_h)
    Nrows = size(cmem.conv_gpu, 1)

    fill!(cmem.conv_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(Nrows, ts[2]))
    @cuda threads=ts blocks=bs kernel_to_dft_layout_2d_scalar_v_mic_gpu!(
        cmem.conv_gpu, cmem.xs_gpu, cmem.dx_gpu, v_los, Int32(0), v_mic, σ_floor,
        Int32(i0), Int32(cmem.Nλ), Int32(cmem.L), Int32(Nrows))
    _normalize_rows!(cmem.conv_gpu, Nrows)
    _maybe_report_underflow!(cmem)
    return nothing
end

function _build_per_row_kernels!(cmem, xs_h::AbstractVector{T},
                                  v_los::CA{T,1}, v_mic::CA{T,1}) where T<:AF
    i0 = cmem.Nλ ÷ 2 + 1
    σ_floor = _sigma_floor_cached(cmem, xs_h)
    Nrows = size(cmem.conv_gpu, 1)

    fill!(cmem.conv_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(cmem.Nλ, ts[1]), cld(Nrows, ts[2]))
    @cuda threads=ts blocks=bs kernel_to_dft_layout_2d_gpu!(
        cmem.conv_gpu, cmem.xs_gpu, cmem.dx_gpu, v_los, Int32(0), v_mic, σ_floor,
        Int32(i0), Int32(cmem.Nλ), Int32(cmem.L), Int32(Nrows), Int32(Nrows))
    _normalize_rows!(cmem.conv_gpu, Nrows)
    _maybe_report_underflow!(cmem)
    return nothing
end

# Inverse FFT, then the rows whose kernel leaves the tap span by direct convolution of `src`, the
# opacities whose padded-signal FFT cmem holds; returns the valid region as a view. v_los and
# v_mic are a device vector per row or one value for every row, as the kernels were built.
function _irfft_and_extract!(cmem, src::CA{T,2}, xs_h::AbstractVector{T}, v_los, v_mic) where T<:AF
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)
    Nrows = size(cmem.conv_gpu, 1)
    _doppler_direct_rows!(cmem.conv_gpu, cmem.pad_left, src, Nrows, cmem.xs_gpu, cmem.dx_gpu,
                          Nrows, _sigma_floor_cached(cmem, xs_h), v_los, 0, v_mic, Nrows,
                          cmem.Nλ, Nrows)
    return @view cmem.conv_gpu[:, cmem.pad_left+1:cmem.pad_left+cmem.Nλ]
end

# ── GPU public API: convolve_wavelength_axis_gpu ─────────────────────────────

"""
    convolve_wavelength_axis_gpu(cmem, xs, ys, v_los, v_mic)

GPU implementation of [`convolve_wavelength_axis`](@ref). Builds the same real-space
kernel with wavelength-dependent σ(x) = x·v_mic/c as the CPU, matching exactly.

Scalar `v_los` and `v_mic` build one kernel broadcast to all rows. Vector arguments
build per-row kernels via batch R2C FFT.

A row whose kernel reaches past the tap span (the `Nλ` taps of the window), by more than
`_DOPPLER_DOMAIN_SIGMAS` kernel widths of its center, is recomputed after the FFT by direct
convolution with each source index clamped to the window (replicate padding); every other row is
the FFT convolution. The CPU [`convolve_wavelength_axis`](@ref) truncates such a kernel at the
window instead, so the two agree only inside that domain.

The fallback reads `ys` on every call, also when `cmem.signal_cached` skips the forward FFT, so
`ys` must always be the opacities whose padded-signal FFT `cmem` holds. A host `ys` is copied to
`cmem.ys_gpu` on every call; a device `ys` is read in place.

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
    return _irfft_and_extract!(cmem, cmem.ys_gpu, xs_h, v_los, v_mic)
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
    return _irfft_and_extract!(cmem, ys_d, cmem.xs_cpu, v_los, v_mic)
end

# vector v_los, scalar v_mic — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, v_los::CA{T,1}, v_mic::T) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_per_row_kernels!(cmem, xs_h, v_los, v_mic)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract!(cmem, cmem.ys_gpu, xs_h, v_los, v_mic)
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
    return _irfft_and_extract!(cmem, ys_d, cmem.xs_cpu, v_los, v_mic)
end

# vector v_los, vector v_mic — host xs
function convolve_wavelength_axis_gpu(cmem::AbstractConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, v_los::CA{T,1}, v_mic::CA{T,1}) where {T<:AF}
    xs_h = collect(T, xs)
    _pad_and_fft_signal!(cmem, ys, xs_h)
    _build_per_row_kernels!(cmem, xs_h, v_los, v_mic)
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.conv_gpu)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu
    return _irfft_and_extract!(cmem, cmem.ys_gpu, xs_h, v_los, v_mic)
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
    return _irfft_and_extract!(cmem, ys_d, cmem.xs_cpu, v_los, v_mic)
end

# ── GPU batched API ──────────────────────────────────────────────────────────

"""
    convolve_wavelength_axis_batched!(bcmem, xs, ys, v_los_batch, v_mic, Bcur)

Batched Doppler convolution for `Bcur` tiles simultaneously. The absorption signal
`ys` (Natm × Nλ) is shared across tiles; `v_los_batch` (Bcur*Natm) provides per-tile
velocities. Scalar `v_mic` avoids tiling allocations; vector `v_mic` is tiled across tiles.
Returns a view of the valid region `(Bcur*Natm, Nλ)`.

Rows whose kernel reaches past the tap span are recomputed by direct convolution, as in
[`convolve_wavelength_axis_gpu`](@ref), from `bcmem.ys_gpu`. That buffer is filled only when
`bcmem.signal_cached` is false, so with the cache set `ys` must not have changed since the call
that filled it, which is also what the cached FFT requires.
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
    σ_floor = bcmem.σ_floor
    ts2 = (32, 32)

    fill!(bcmem.conv_gpu, zero(T))
    bs_k = (cld(bcmem.Nλ, ts2[1]), cld(BNatm, ts2[2]))
    v_los_off = Int32(tile_offset * Natm)
    @cuda threads=ts2 blocks=bs_k kernel_to_dft_layout_2d_scalar_v_mic_gpu!(
        bcmem.conv_gpu, bcmem.xs_gpu, bcmem.dx_gpu, v_los_batch, v_los_off, v_mic, σ_floor,
        Int32(i0), Int32(bcmem.Nλ), Int32(bcmem.L), Int32(BNatm))
    _normalize_rows!(bcmem.conv_gpu, BNatm)
    # Restrict underflow count to the first BNatm rows; trailing rows of
    # bcmem.conv_gpu are pre-zeroed by `fill!` and are not active.
    n_underflow = _count_zero_rows(bcmem.conv_gpu, BNatm)
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow of $BNatm active row(s): their kernel centers lie far outside the wavelength window, and those rows are rendered by direct convolution with the window's edge opacity. Check that v_los is in m/s." maxlog=3
    mul!(bcmem.kernel_ft_gpu, bcmem.plan_fwd_kernel, bcmem.conv_gpu)

    nfreq = size(bcmem.kernel_ft_gpu, 2)
    BNatm32 = Int32(BNatm)
    bs3 = (cld(nfreq, ts2[1]), cld(size(bcmem.conv_ft_gpu, 1), ts2[2]))
    @cuda threads=ts2 blocks=bs3 batched_spectral_multiply!(
        bcmem.conv_ft_gpu, bcmem.signal_ft_gpu, bcmem.kernel_ft_gpu,
        Int32(Natm), BNatm32)

    mul!(bcmem.conv_gpu, bcmem.plan_bwd, bcmem.conv_ft_gpu)
    _doppler_direct_rows!(bcmem.conv_gpu, bcmem.pad_left, bcmem.ys_gpu, Natm, bcmem.xs_gpu,
                          bcmem.dx_gpu, BNatm, σ_floor, v_los_batch, tile_offset * Natm, v_mic,
                          Natm, bcmem.Nλ, BNatm)
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
    σ_floor = bcmem.σ_floor
    ts2 = (32, 32)

    fill!(bcmem.conv_gpu, zero(T))
    bs_k = (cld(bcmem.Nλ, ts2[1]), cld(BNatm, ts2[2]))
    v_los_off = Int32(tile_offset * Natm)
    @cuda threads=ts2 blocks=bs_k kernel_to_dft_layout_2d_gpu!(
        bcmem.conv_gpu, bcmem.xs_gpu, bcmem.dx_gpu, v_los_batch, v_los_off, v_mic, σ_floor,
        Int32(i0), Int32(bcmem.Nλ), Int32(bcmem.L), Int32(Natm), Int32(BNatm))
    _normalize_rows!(bcmem.conv_gpu, BNatm)
    # Restrict underflow count to the first BNatm rows; trailing rows of
    # bcmem.conv_gpu are pre-zeroed by `fill!` and are not active.
    n_underflow = _count_zero_rows(bcmem.conv_gpu, BNatm)
    n_underflow > 0 && @warn "Doppler kernel underflowed in $n_underflow of $BNatm active row(s): their kernel centers lie far outside the wavelength window, and those rows are rendered by direct convolution with the window's edge opacity. Check that v_los is in m/s." maxlog=3
    mul!(bcmem.kernel_ft_gpu, bcmem.plan_fwd_kernel, bcmem.conv_gpu)

    nfreq = size(bcmem.kernel_ft_gpu, 2)
    BNatm32 = Int32(BNatm)
    bs3 = (cld(nfreq, ts2[1]), cld(size(bcmem.conv_ft_gpu, 1), ts2[2]))
    @cuda threads=ts2 blocks=bs3 batched_spectral_multiply!(
        bcmem.conv_ft_gpu, bcmem.signal_ft_gpu, bcmem.kernel_ft_gpu,
        Int32(Natm), BNatm32)

    mul!(bcmem.conv_gpu, bcmem.plan_bwd, bcmem.conv_ft_gpu)
    _doppler_direct_rows!(bcmem.conv_gpu, bcmem.pad_left, bcmem.ys_gpu, Natm, bcmem.xs_gpu,
                          bcmem.dx_gpu, BNatm, σ_floor, v_los_batch, tile_offset * Natm, v_mic,
                          Natm, bcmem.Nλ, BNatm)
    return @view bcmem.conv_gpu[1:BNatm, bcmem.pad_left+1:bcmem.pad_left+bcmem.Nλ]
end
