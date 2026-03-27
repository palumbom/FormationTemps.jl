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
    # short circuit
    if iszero(ζ_rt)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = rt_macro_kernel(vs, ζ_rt, μ)
    kshift = ifftshift(kernel)

    # return convolution via FFT (matches GPU convention)
    return real(ifft(fft(ys) .* fft(kshift)))
end

function convolve_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_rt::T, μ::T) where T<:AF
    # short circuit
    if iszero(ζ_rt)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = rt_macro_kernel(vs, ζ_rt, μ)
    kshift = ifftshift(kernel)
    ftk = fft(kshift)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
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

"""
    convolve_rt_macro_gpu(cmem, xs, ys, ζ_rt, μ)

GPU implementation of [`convolve_rt_macro`](@ref). Convolves each row of `ys`
with the anisotropic radial-tangential macroturbulence kernel using padded FFT
convolution on the device.

Arguments:
- `cmem::ConvolutionMemory`: Pre-allocated GPU working memory.
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}`: Input matrix with shape `(Natm, Nλ)`.
- `ζ_rt::Real`: Radial-tangential macroturbulence velocity scale (m/s).
- `μ::Real`: Cosine of the angle between the local normal and the line of sight.

Returns:
- `out::CuArray{<:Real,2}`: Convolved matrix on the GPU, same shape as `ys`.

Notes:
- Short-circuits (returns `CuArray(ys)`) when `ζ_rt` is zero.
- The GPU evaluates `erfc` in a CUDA kernel; results differ from the CPU `erfc`
  (Julia standard library) by ~1e-4 relative to peak flux.

See also: [`convolve_rt_macro`](@ref), [`rt_macro_kernel`](@ref)
"""
function convolve_rt_macro_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                               ys::AA{T,2}, ζ_rt::T, μ::T) where {T<:AF}
    # short circuit before any copy so the caller's ys is returned unmodified
    if iszero(ζ_rt)
        return CuArray(ys)
    end

    # copy signal to device — avoid CuArray() wrapper allocations
    if ys isa CA
        copyto!(cmem.ys_gpu, ys)
    else
        copyto!(cmem.ys_gpu, CuArray(ys))
    end

    # populate wavelength grid on device and get center wavelength without scalar indexing
    if xs isa CA
        copyto!(cmem.xs_gpu, xs)
        xs_h = Array(xs)
    else
        copyto!(cmem.xs_gpu, CuArray(xs))
        xs_h = xs
    end

    # compute velocity offset from discrete center
    i0 = length(xs_h) ÷ 2 + 1
    λ0 = xs_h[i0]

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)

    # compute the padded kernel once (reuse pre-allocated row buffers)
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
    center = Ltot ÷ 2
    r = center - (cmem.pad_left + i0)
    if r != 0
        @cuda threads=ts1 blocks=(cld(Ltot, ts1[1]),) roll_1d!(shifted_kernel_row, kernel_row, r, Ltot)
        CUDA.synchronize()
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

    # slice valid region
    out = cmem.conv_gpu[:, cmem.pad_left : cmem.pad_left + cmem.Nλ - 1]
    CUDA.synchronize()
    return out
end

"""
    precompute_rt_macro_kernel_ft(cmem, xs, ζ_rt, μ)

Precompute the Fourier transform of the RT macroturbulence kernel for a given `μ`.
Returns a `CuVector{Complex{T}}` that can be passed to [`convolve_rt_macro_gpu_cached`](@ref).
"""
function precompute_rt_macro_kernel_ft(cmem::ConvolutionMemory, xs::AA{T,1},
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

    # normalize
    normval = CUDA.sum(kernel_row)
    kernel_row ./= normval

    # roll to align zero-lag
    Ltot = length(kernel_row)
    center = Ltot ÷ 2
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
function convolve_rt_macro_gpu_cached(cmem::ConvolutionMemory,
                                      ys::CA{T,2},
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
