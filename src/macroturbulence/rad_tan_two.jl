"""
    rt_macro_kernel(vs, ζ_r, ζ_t, μ)

Compute the anisotropic radial-tangential macroturbulence kernel from Gray (2008)
(Eq. 17.6) with independent radial and tangential velocity scales, assuming equal
amplitudes (A_R = A_T).

Arguments:
- `vs::AbstractVector{<:Real}`: Velocity grid centered on the line core (m/s).
- `ζ_r::Real`: Radial macroturbulence velocity scale (m/s).
- `ζ_t::Real`: Tangential macroturbulence velocity scale (m/s).
- `μ::Real`: Cosine of the angle between the local normal and the line of sight.

Returns:
- `kernel::Vector{<:Real}`: Normalized macroturbulence kernel evaluated on `vs`.

See also: [`convolve_rt_macro`](@ref)
"""
function rt_macro_kernel(vs::AA{T,1}, ζ_r::T, ζ_t::T, μ::T) where T<:AF
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
    t1 = @. A_R * exp(-(vs / (ζ_r * cosθ))^2.0) / (sqrt_π * ζ_r * cosθ)
    t2 = @. A_T * exp(-(vs / (ζ_t * sinθ))^2.0) / (sqrt_π * ζ_t * sinθ)
    kernel = t1 + t2
    return kernel ./ sum(kernel)
end

"""
    convolve_rt_macro(xs, ys, ζ_r, ζ_t, μ)

Convolve a spectrum with the anisotropic radial-tangential macroturbulence kernel,
allowing independent radial and tangential velocity scales.

Arguments:
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractArray{<:Real}`: Spectrum on `xs` (vector or matrix with rows as spectra).
- `ζ_r::Real`: Radial macroturbulence velocity scale (m/s).
- `ζ_t::Real`: Tangential macroturbulence velocity scale (m/s).
- `μ::Real`: Cosine of the angle between the local normal and the line of sight.

Returns:
- `ys_out::AbstractArray{<:Real}`: Convolved spectrum (or `ys` if `ζ_r == 0` and `ζ_t == 0`).

See also: [`rt_macro_kernel`](@ref), [`convolve_rt_macro_gpu`](@ref)
"""

function convolve_rt_macro(xs::AA{T,1}, ys::AA{T,1}, ζ_r::T, ζ_t::T, μ::T) where T<:AF
    # short circuit
    if iszero(ζ_r) && iszero(ζ_t)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = rt_macro_kernel(vs, ζ_r, ζ_t, μ)
    kshift = ifftshift(kernel)

    # return convolution via FFT (matches GPU convention)
    return real(ifft(fft(ys) .* fft(kshift)))
end

function convolve_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_r::T, ζ_t::T, μ::T) where T<:AF
    # short circuit
    if iszero(ζ_r) && iszero(ζ_t)
        return ys
    end

    # offset the kernel by the velocity (discrete center)
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = rt_macro_kernel(vs, ζ_r, ζ_t, μ)
    kshift = ifftshift(kernel)
    ftk = fft(kshift)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end

function compute_padded_rt_kernel_1D!(kernel_row, xs, λc, ζ_r, ζ_t, μ, Nλ, pad_left)
    # get thread index
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # evaluate the kernel
    if j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc

        # trig from μ directly; guard μ≈0 or μ≈1
        T = typeof(ζ_r)
        ϵ = T(1e-6)
        cosθ = max(μ, ϵ)
        s2 = one(T) - μ * μ
        sinθ = sqrt(ifelse(s2 > zero(T), s2, ϵ*ϵ))

        invR = 0.5 / (sqrt(π) * ζ_r * cosθ)
        invT = 0.5 / (sqrt(π) * ζ_t * sinθ)

        t1 = exp(-(xj / (ζ_r * cosθ))^2) * invR
        t2 = exp(-(xj / (ζ_t * sinθ))^2) * invT
        @inbounds kernel_row[j + pad_left] = t1 + t2
    end
    return nothing
end

"""
    convolve_rt_macro_gpu(cmem, xs, ys, ζ_r, ζ_t, μ)

GPU implementation of `convolve_rt_macro(xs, ys, ζ_r, ζ_t, μ)`. Convolves each row
of `ys` with the anisotropic radial-tangential macroturbulence kernel using padded FFT
convolution on the device, allowing independent radial and tangential velocity scales.

Arguments:
- `cmem::MacroConvolutionMemory`: Pre-allocated GPU working memory.
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}`: Input matrix with shape `(Natm, Nλ)`.
- `ζ_r::Real`: Radial macroturbulence velocity scale (m/s).
- `ζ_t::Real`: Tangential macroturbulence velocity scale (m/s).
- `μ::Real`: Cosine of the angle between the local normal and the line of sight.

Returns:
- `out::CuArray{<:Real,2}`: Convolved matrix on the GPU, same shape as `ys`.

Notes:
- Short-circuits (returns `CuArray(ys)`) when both `ζ_r` and `ζ_t` are zero.
- CPU and GPU results differ at the spectrum edges due to circular vs padded FFT convention.

See also: [`convolve_rt_macro`](@ref), [`rt_macro_kernel`](@ref)
"""
function convolve_rt_macro_gpu(cmem::MacroConvolutionMemory, xs::AA{T,1},
                               ys::AA{T,2}, ζ_r::T, ζ_t::T, μ::T) where {T<:AF}
    # short circuit before any copy so the caller's ys is returned unmodified
    if iszero(ζ_r) && iszero(ζ_t)
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

    # compute the padded kernel once
    kernel_row = cmem.padded_kernel_gpu
    shifted_kernel_row = cmem.shift_kernel_gpu
    fill!(kernel_row, zero(T))

    ts1 = (256,)
    bs1 = (cld(cmem.Nλ, ts1[1]),)
    @cuda threads=ts1 blocks=bs1 compute_padded_rt_kernel_1D!(kernel_row,
                                                              cmem.xs_gpu, λ0,
                                                              ζ_r, ζ_t, μ, cmem.Nλ,
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

    # center -> FFT indexing, then R2C FFT of padded kernel (no allocation)
    CUDA.CUFFT.ifftshift!(shifted_kernel_row, kernel_row, 1)
    copyto!(cmem.kr_1d, shifted_kernel_row)
    mul!(cmem.kernel_row_ft_1d, cmem.plan_fwd_1d, cmem.kr_1d)

    # forward FFT of padded signal
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem
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
