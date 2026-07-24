"""
    gray_iso_rt_macro_kernel(vs, ζ_rt)

Compute the isotropic radial-tangential macroturbulence kernel from Gray (2008)
(Eq. 17.8), assuming equal amplitudes (A_R = A_T) and equal velocity scales (ζ_R = ζ_T).
The kernel is the disk-integrated (μ-averaged) form.

Arguments:
- `vs::AbstractVector{<:Real}`: Velocity grid centered on the line core (m/s).
- `ζ_rt::Real`: Isotropic macroturbulence velocity scale (m/s).

Returns:
- `kernel::Vector{<:Real}`: Normalized macroturbulence kernel evaluated on `vs`.

See also: [`convolve_iso_rt_macro`](@ref), [`rt_macro_kernel`](@ref)
"""
function gray_iso_rt_macro_kernel(vs::AA{T,1}, ζ_rt::T) where T<:AF
    t1 = 2.0 .* exp.(-1.0 .* (vs ./ ζ_rt).^2.0) ./ (sqrt(π) .* ζ_rt)
    t2 = -2.0 .* abs.(vs) .* erfc.(abs.(vs) ./ ζ_rt) ./ ζ_rt.^2.0
    kernel = t1 .+ t2
    # TODO(zero-sum-guard): unguarded normalization; can produce NaN if the kernel
    # underflows. Apply the ifelse(iszero(s), one(T), s) guard used in microturbulence.jl.
    return kernel ./ sum(kernel)
end

"""
    convolve_iso_rt_macro(xs, ys, ζ_rt)

Convolve a spectrum with the isotropic radial-tangential macroturbulence kernel from
Gray (2008) (Eq. 17.8). This is the disk-integrated (μ-averaged) form appropriate for
stellar flux spectra.

Arguments:
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractArray{<:Real}`: Spectrum on `xs` (vector or matrix with rows as spectra).
- `ζ_rt::Real`: Isotropic macroturbulence velocity scale (m/s).

Returns:
- `ys_out::AbstractArray{<:Real}`: Convolved spectrum (or `ys` if `ζ_rt == 0`).

See also: [`gray_iso_rt_macro_kernel`](@ref), [`convolve_iso_rt_macro_gpu`](@ref),
[`convolve_rt_macro`](@ref)
"""
function convolve_iso_rt_macro(xs::AA{T,1}, ys::AA{T,1}, ζ_rt::T) where T<:AF
    if iszero(ζ_rt)
        return ys
    end
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0
    kernel = gray_iso_rt_macro_kernel(vs, ζ_rt)
    return _padded_convolve(collect(T, ys), kernel)
end

function convolve_iso_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_rt::T) where T<:AF
    if iszero(ζ_rt)
        return ys
    end
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0
    kernel = gray_iso_rt_macro_kernel(vs, ζ_rt)
    return _padded_convolve(collect(T, ys), kernel)
end

function compute_padded_iso_rt_kernel_1D!(kernel_row, xs, λc, ζ_rt, Nλ, pad_left)
    # get thread index
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # evaluate the kernel
    if j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc
        av = CUDA.abs(xj)
        z = av / ζ_rt

        T = typeof(ζ_rt)
        t1 = T(2) * exp(-(xj/ζ_rt)^2) / (sqrt(T(π)) * ζ_rt)
        t2 = T(-2) * av * erfc(z) / (ζ_rt^2)
        @inbounds kernel_row[j + pad_left] = t1 + t2
    end
    return nothing
end

"""
    convolve_iso_rt_macro_gpu(cmem, xs, ys, ζ_rt)

GPU implementation of [`convolve_iso_rt_macro`](@ref). Convolves each row of `ys`
with the isotropic radial-tangential macroturbulence kernel using padded FFT
convolution on the device.

Arguments:
- `cmem::MacroConvolutionMemory`: Pre-allocated GPU working memory.
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}`: Input matrix with shape `(Natm, Nλ)`.
- `ζ_rt::Real`: Isotropic macroturbulence velocity scale (m/s).

Returns:
- `out::CuArray{<:Real,2}`: Convolved matrix on the GPU, same shape as `ys`.

Notes:
- Short-circuits (returns `CuArray(ys)`) when `ζ_rt` is zero.
- CPU and GPU both use padded linear convolution with edge replication.

See also: [`convolve_iso_rt_macro`](@ref), [`gray_iso_rt_macro_kernel`](@ref)
"""
function convolve_iso_rt_macro_gpu(cmem::MacroConvolutionMemory, xs::AA{T,1},
                                   ys::AA{T,2}, ζ_rt::T) where {T<:AF}
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

    # compute the padded kernel once
    kernel_row = cmem.padded_kernel_gpu
    shifted_kernel_row = cmem.shift_kernel_gpu
    fill!(kernel_row, zero(T))

    ts1 = (256,)
    bs1 = (cld(cmem.Nλ, ts1[1]),)
    @cuda threads=ts1 blocks=bs1 compute_padded_iso_rt_kernel_1D!(kernel_row,
                                                                  cmem.xs_gpu, λ0,
                                                                  ζ_rt, cmem.Nλ,
                                                                  cmem.pad_left)

    # normalize the kernel
    # TODO(zero-sum-guard): unguarded normalization; can produce NaN if the kernel
    # underflows. Apply the ifelse(iszero(s), one(T), s) guard used in microturbulence.jl.
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
