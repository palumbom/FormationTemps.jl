"""
    rt_macro_kernel(vs, ζ_rt, μ)

Compute the anisotropic radial-tangential macroturbulence kernel from Gray (2008)
(Eq. 17.6), assuming equal amplitudes (A_R = A_T) and equal velocity scales (ζ_R = ζ_T).

Arguments:
- `vs::AbstractVector{<:Real}`: Velocity grid centered on the line core (m/s).
- `ζ_rt::Real`: Radial-tangential macroturbulence velocity scale (m/s).
- `μ::Real`: Cosine of the angle between the local normal and the line of sight.

Returns:
- `kernel::Vector{<:Real}`: Normalized macroturbulence kernel evaluated on `vs`.

See also: [`convolve_rt_macro`](@ref), [`gray_iso_rt_macro_kernel`](@ref)
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

"""
    convolve_rt_macro(xs, ys, ζ_rt, μ)

Convolve a spectrum with the anisotropic radial-tangential macroturbulence kernel from
Gray (2008) (Eq. 17.6), assuming equal radial and tangential velocity scales (`ζ_R = ζ_T`).

Arguments:
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractArray{<:Real}`: Spectrum on `xs` (vector or matrix with rows as spectra).
- `ζ_rt::Real`: Radial-tangential macroturbulence velocity scale (m/s).
- `μ::Real`: Cosine of the angle between the local normal and the line of sight.

Returns:
- `ys_out::AbstractArray{<:Real}`: Convolved spectrum (or `ys` if `ζ_rt == 0`).

See also: [`rt_macro_kernel`](@ref), [`convolve_iso_rt_macro`](@ref)
"""
function convolve_rt_macro(xs::AA{T,1}, ys::AA{T,1}, ζ_rt::T, μ::T) where T<:AF
    if iszero(ζ_rt)
        return ys
    end
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0
    kernel = rt_macro_kernel(vs, ζ_rt, μ)
    return _padded_convolve(collect(T, ys), kernel)
end

function convolve_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_rt::T, μ::T) where T<:AF
    if iszero(ζ_rt)
        return ys
    end
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0
    kernel = rt_macro_kernel(vs, ζ_rt, μ)
    return _padded_convolve(collect(T, ys), kernel)
end

"""
    _convolve_macro_inplace!(out, xs, ys, ζ_rt, μ, ws)

In-place radial-tangential macroturbulence convolution using pre-allocated
[`CPUTileWorkspace`](@ref) buffers. Uses padded linear convolution with
edge replication and R2C FFTs, matching the GPU path. When `ζ_rt == 0`,
copies `ys` into `out` directly.
"""
function _convolve_macro_inplace!(out::AA{T,2}, xs::AA{T,1}, ys::AA{T,2},
                                  ζ_rt::T, μ::T,
                                  ws::CPUTileWorkspace) where T<:AF
    Nrows = size(ys, 1)
    if iszero(ζ_rt)
        copyto!(out, ys)
        return nothing
    end

    Nλ = ws.Nλ
    i0 = Nλ ÷ 2 + 1
    λ0 = xs[i0]

    ϵ = T(1e-6)
    cosθ = max(μ, ϵ)
    s2 = one(T) - μ * μ
    sinθ = sqrt(ifelse(s2 > zero(T), s2, ϵ * ϵ))
    sqrt_π = sqrt(T(π))

    # evaluate kernel into a temporary Nλ-length buffer, then normalize
    kvec = Vector{T}(undef, Nλ)
    @inbounds for j in 1:Nλ
        v = c_ms * (xs[j] - λ0) / λ0
        t1 = T(0.5) * exp(-(v / (ζ_rt * cosθ))^2) / (sqrt_π * ζ_rt * cosθ)
        t2 = T(0.5) * exp(-(v / (ζ_rt * sinθ))^2) / (sqrt_π * ζ_rt * sinθ)
        kvec[j] = t1 + t2
    end
    s = sum(kvec)
    kvec ./= s

    # place kernel in DFT layout, then R2C FFT
    _kernel_to_dft_layout!(ws.kernel_real, kvec, i0)
    mul!(ws.kernel_ft, ws.fft_plan, ws.kernel_real)

    _apply_fft_kernel!(out, ys, ws.kernel_ft, ws, Nrows)
    return nothing
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

        invR = T(0.5) / (sqrt(T(π)) * ζ_rt * cosθ)
        invT = T(0.5) / (sqrt(T(π)) * ζ_rt * sinθ)

        t1 = exp(-(xj / (ζ_rt * cosθ))^2) * invR
        t2 = exp(-(xj / (ζ_rt * sinθ))^2) * invT
        @inbounds kernel_row[j + pad_left] = t1 + t2
    end
    return nothing
end

"""
    convolve_rt_macro_gpu(cmem, xs, ys, ζ_rt, μ)

GPU implementation of [`convolve_rt_macro`](@ref). Convolves each row of `ys`
with the anisotropic radial-tangential macroturbulence kernel using padded FFT
convolution on the device.

Arguments:
- `cmem::MacroConvolutionMemory`: Pre-allocated GPU working memory.
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}`: Input matrix with shape `(Natm, Nλ)`.
- `ζ_rt::Real`: Radial-tangential macroturbulence velocity scale (m/s).
- `μ::Real`: Cosine of the angle between the local normal and the line of sight.

Returns:
- `out::CuArray{<:Real,2}`: Convolved matrix on the GPU, same shape as `ys`.

Notes:
- Short-circuits (returns `CuArray(ys)`) when `ζ_rt` is zero.
- CPU and GPU both use padded linear convolution with edge replication.

See also: [`convolve_rt_macro`](@ref), [`rt_macro_kernel`](@ref)
"""
function convolve_rt_macro_gpu(cmem::MacroConvolutionMemory, xs::AA{T,1},
                               ys::AA{T,2}, ζ_rt::T, μ::T) where {T<:AF}
    # short circuit before any copy so the caller's ys is returned unmodified
    if iszero(ζ_rt)
        return CuArray(ys)
    end

    # copy to device
    xs_h = xs isa CA ? Array(xs) : collect(T, xs)
    copyto!(cmem.ys_gpu, ys)
    copyto!(cmem.xs_gpu, xs_h)

    # compute velocity offset from discrete center
    i0 = length(xs_h) ÷ 2 + 1
    λ0 = xs_h[i0]

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)

    # compute the padded kernel once (reuse pre-allocated row buffers)
    kernel_row = cmem.padded_kernel_gpu
    shifted_kernel_row = cmem.shift_kernel_gpu
    fill!(kernel_row, zero(T))

    ts1 = (256,)
    bs1 = (cld(cmem.Nλ, ts1[1]),)
    @cuda threads=ts1 blocks=bs1 compute_padded_rt_kernel_1D!(kernel_row,
                                                              cmem.xs_gpu, λ0,
                                                              ζ_rt, μ, cmem.Nλ,
                                                              cmem.pad_left)

    # normalize the kernel
    normval = CUDA.sum(kernel_row)
    kernel_row ./= normval

    # ensure zero-lag sits at padded center before FFT layout
    Ltot = length(kernel_row)
    center = Ltot ÷ 2
    r = center - (cmem.pad_left + i0)
    if r != 0
        @cuda threads=ts1 blocks=(cld(Ltot, ts1[1]),) roll_1d!(shifted_kernel_row, kernel_row, r, Ltot)
        tmp = kernel_row
        kernel_row = shifted_kernel_row
        shifted_kernel_row = tmp
    end

    # center -> FFT indexing (writes into shifted_kernel_row)
    CUDA.CUFFT.ifftshift!(shifted_kernel_row, kernel_row, 1)

    # copy into contiguous 1D buffer and FFT (no allocation)
    copyto!(cmem.kr_1d, shifted_kernel_row)
    mul!(cmem.kernel_row_ft_1d, cmem.plan_fwd_1d, cmem.kr_1d)

    # forward FFT of padded signal
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem (broadcast 1D kernel across all rows)
    kft = reshape(cmem.kernel_row_ft_1d, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft

    # inverse fourier transform
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # extract valid region into pre-allocated output buffer
    ts2 = (32, 32)
    bs2 = (cld(cmem.Natm, ts2[1]), cld(cmem.Nλ, ts2[2]))
    @cuda threads=ts2 blocks=bs2 extract_valid!(cmem.out_gpu, cmem.conv_gpu,
                                                 cmem.pad_left, cmem.Nλ)
    return cmem.out_gpu
end

"""
    precompute_rt_macro_kernel_ft(cmem, xs, ζ_rt, μ)

Precompute the Fourier transform of the RT macroturbulence kernel for a given `μ`.
Returns a `CuVector{Complex{T}}` that can be passed to [`convolve_rt_macro_gpu_cached`](@ref).
"""
function precompute_rt_macro_kernel_ft(cmem::MacroConvolutionMemory, xs::AA{T,1},
                                       ζ_rt::T, μ::T) where {T<:AF}
    # populate wavelength grid on device and get center wavelength without scalar indexing
    if xs isa CA
        copyto!(cmem.xs_gpu, xs)
        xs_h = Array(xs)
    else
        copyto!(cmem.xs_gpu, CuArray(xs))
        xs_h = xs
    end

    i0 = length(xs_h) ÷ 2 + 1
    λ0 = xs_h[i0]

    # compute padded kernel
    kernel_row = cmem.padded_kernel_gpu
    shifted_kernel_row = cmem.shift_kernel_gpu
    fill!(kernel_row, zero(T))

    ts1 = (256,)
    bs1 = (cld(cmem.Nλ, ts1[1]),)
    @cuda threads=ts1 blocks=bs1 compute_padded_rt_kernel_1D!(kernel_row,
                                                              cmem.xs_gpu, λ0,
                                                              ζ_rt, μ, cmem.Nλ,
                                                              cmem.pad_left)

    # normalize
    normval = CUDA.sum(kernel_row)
    kernel_row ./= normval

    # roll to align zero-lag
    Ltot = length(kernel_row)
    center = Ltot ÷ 2
    r = center - (cmem.pad_left + i0)
    if r != 0
        @cuda threads=ts1 blocks=(cld(Ltot, ts1[1]),) roll_1d!(shifted_kernel_row, kernel_row, r, Ltot)
        tmp = kernel_row
        kernel_row = shifted_kernel_row
        shifted_kernel_row = tmp
    end

    # center -> FFT indexing
    CUDA.CUFFT.ifftshift!(shifted_kernel_row, kernel_row, 1)

    # FFT into pre-allocated buffer and return a copy (caller stores it)
    copyto!(cmem.kr_1d, shifted_kernel_row)
    mul!(cmem.kernel_row_ft_1d, cmem.plan_fwd_1d, cmem.kr_1d)
    return copy(cmem.kernel_row_ft_1d)
end

"""
    convolve_rt_macro_gpu_cached(cmem, ys, kernel_ft)

Convolve `ys` with a precomputed RT macroturbulence kernel FFT. Skips kernel
computation entirely. Use with [`precompute_rt_macro_kernel_ft`](@ref).
"""
function convolve_rt_macro_gpu_cached(cmem::MacroConvolutionMemory,
                                      ys::AA{T,2},
                                      kernel_ft::CuVector{Complex{T}}) where {T<:AF}
    # copy signal to device buffer
    copyto!(cmem.ys_gpu, ys)

    # pad the signal
    ts = (32, 32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)

    # forward FFT of padded signal
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem (broadcast 1D kernel across all rows)
    kft = reshape(kernel_ft, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft

    # inverse fourier transform
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # extract valid region into pre-allocated output buffer (zero allocation)
    ts2 = (32, 32)
    bs2 = (cld(cmem.Natm, ts2[1]), cld(cmem.Nλ, ts2[2]))
    @cuda threads=ts2 blocks=bs2 extract_valid!(cmem.out_gpu, cmem.conv_gpu,
                                                 cmem.pad_left, cmem.Nλ)
    return cmem.out_gpu
end

# ── batched Fourier-domain macro accumulation ─────────────────────────────────

"""
    batched_macro_multiply_accumulate_kernel!(acc_ft, signal_ft, kernel_cache_flat,
                                              μ_idx, dA_tiles, tile_offset, Natm1, Bcur)

CUDA kernel: for each (layer k, frequency f), loop over B tiles in the batch, multiply
the tile's forward-FFT'd signal by its cached macro kernel FFT, weight by dA, and
accumulate into the Fourier-space accumulator.
"""
function batched_macro_multiply_accumulate_kernel!(acc_ft, signal_ft, kernel_cache_flat,
                                                    μ_idx, dA_tiles, tile_offset,
                                                    Natm1, Bcur)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    f = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    nfreq = size(acc_ft, 2)
    (k > Natm1 || f > nfreq) && return nothing

    @inbounds for bi in 1:Bcur
        row = (bi - 1) * Natm1 + k
        ki = μ_idx[tile_offset + bi]
        dA_i = dA_tiles[tile_offset + bi]
        acc_ft[k, f] += signal_ft[row, f] * kernel_cache_flat[ki, f] * dA_i
    end
    return nothing
end

"""
    batched_macro_multiply_accumulate!(acc_ft, signal_ft, kernel_cache_flat,
                                       μ_idx, dA_tiles, Natm1, Bcur; tile_offset=0)

Launch the batched macro multiply-accumulate kernel. `signal_ft` is `(Bcur*Natm1, nfreq)`
from a batched forward FFT of padded cfdt. `kernel_cache_flat` is `(N_unique_μ, nfreq)`.
`μ_idx` maps tile index → row in kernel_cache_flat. Result accumulates into `acc_ft`
`(Natm1, nfreq)`.
"""
function batched_macro_multiply_accumulate!(acc_ft::CA{Complex{T},2},
                                            signal_ft::CA{Complex{T},2},
                                            kernel_cache_flat::CA{Complex{T},2},
                                            μ_idx::CA{Int32,1},
                                            dA_tiles::CA{T,1},
                                            Natm1::Int, Bcur::Int;
                                            tile_offset::Int=0) where T<:AF
    nfreq = size(acc_ft, 2)
    ts = (16, 16)
    bs = (cld(Natm1, ts[1]), cld(nfreq, ts[2]))
    @cuda threads=ts blocks=bs batched_macro_multiply_accumulate_kernel!(
        acc_ft, signal_ft, kernel_cache_flat, μ_idx, dA_tiles,
        Int32(tile_offset), Int32(Natm1), Int32(Bcur))
    return nothing
end

# ── batched macro kernel precomputation ───────────────────────────────────────

"""
    compute_rt_macro_dft_layout_2d!(kbuf, xs, μ_vals, i0, ζ_rt, Nλ, L)

CUDA kernel: evaluate the RT macroturbulence kernel for multiple μ values in parallel,
writing directly in DFT layout (zero-lag at index 1). Each row of `kbuf` (N_unique, L)
corresponds to one unique μ value. Skips the roll + ifftshift needed by the serial path.
"""
function compute_rt_macro_dft_layout_2d!(kbuf, xs, μ_vals, i0, ζ_rt, Nλ, L)
    j   = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    Nrows = size(kbuf, 1)
    (row > Nrows || j > Nλ) && return nothing
    T = eltype(kbuf)

    xj = @inbounds xs[j]
    λ0 = @inbounds xs[i0]
    v = c_ms * (xj - λ0) / λ0
    μ = @inbounds μ_vals[row]

    ϵ = T(1e-6)
    cosθ = max(μ, ϵ)
    s2 = one(T) - μ * μ
    sinθ = sqrt(ifelse(s2 > zero(T), s2, ϵ * ϵ))

    invR = T(0.5) / (sqrt(T(π)) * ζ_rt * cosθ)
    invT_val = T(0.5) / (sqrt(T(π)) * ζ_rt * sinθ)
    t1 = exp(-(v / (ζ_rt * cosθ))^2) * invR
    t2 = exp(-(v / (ζ_rt * sinθ))^2) * invT_val

    d = j - i0
    idx = d > 0 ? d : L + d
    @inbounds kbuf[row, idx] = t1 + t2
    return nothing
end
