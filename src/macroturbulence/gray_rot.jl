"""
    gray_rot_kernel(vs, vsini, u1)

Compute the Gray (2008) rotation broadening kernel (Eq. 18.14) with linear
limb darkening.

Arguments:
- `vs::AbstractVector{<:Real}`: Velocity grid centered on the line core (m/s).
- `vsini::Real`: Projected rotational velocity (m/s).
- `u1::Real`: Linear limb-darkening coefficient.

Returns:
- `kernel::Vector{<:Real}`: Normalized rotation kernel evaluated on `vs`.

See also: [`convolve_gray_rotation`](@ref)
"""
function gray_rot_kernel(vs::AA{T,1}, vsini::T, u1::T) where T<:AF
    # get LD terms
    ld1 = 2.0 * (one(T) - u1)
    ld2 = 0.5 * π * u1
    ld3 = π * (one(T) - u1 / 3.0)

    # evaluate the kernel
    xs = vs ./ vsini
    omx2 = abs.(one(T) .- xs .^ 2.0)
    kernel = (ld1 .* sqrt.(omx2) .+ ld2 .* omx2) ./ ld3
    kernel[abs.(xs) .> one(T)] .= zero(T)
    return kernel ./ sum(kernel)
end

"""
    convolve_gray_rotation(xs, ys, vsini, u1)

Convolve a spectrum with the Gray (2008) rotation kernel using linear limb darkening.

Arguments:
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractArray{<:Real}`: Spectrum on `xs` (vector or matrix with rows as spectra).
- `vsini::Real`: Projected rotational velocity (m/s).
- `u1::Real`: Linear limb-darkening coefficient.

Returns:
- `ys_out::AbstractArray{<:Real}`: Convolved spectrum with the same shape as `ys`.

See also: [`gray_rot_kernel`](@ref), [`convolve_gray_rotation_gpu`](@ref)
"""
function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,1}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase: zero-lag at index 1)
    kernel = gray_rot_kernel(vs, vsini, u1)
    kshift = ifftshift(kernel)

    # return convolution via FFT (matches GPU convention)
    return real(ifft(fft(ys) .* fft(kshift)))
end

function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,2}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel (GPU-style phase)
    kernel = gray_rot_kernel(vs, vsini, u1)
    kshift = ifftshift(kernel)
    ftk = fft(kshift)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end

function compute_padded_gray_kernel_1D!(kernel_row, xs, λc, vsini, u1, Nλ, pad_left)
    # get thread index
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # get LD terms
    ld1 = 2.0 * (1.0 - u1)
    ld2 = 0.5 * π * u1
    ld3 = π * (1.0 - u1 / 3.0)

    # evaluate the kernel
    if j <= Nλ
        xj = c_ms * (xs[j] - λc) / λc / vsini
        omx2 = CUDA.abs(1.0 - xj ^ 2.0)

        val = (ld1 * sqrt(omx2) + ld2 * omx2) / ld3
        val *= abs(xj) <= 1.0
        @inbounds kernel_row[j + pad_left] = val
    end
    return nothing
end

"""
    convolve_gray_rotation_gpu(cmem, xs, ys, vsini, u1)

GPU implementation of [`convolve_gray_rotation`](@ref). Convolves each row of `ys`
with the Gray (2008) rotation kernel using padded FFT convolution on the device.

Arguments:
- `cmem::ConvolutionMemory`: Pre-allocated GPU working memory.
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}`: Input matrix with shape `(Natm, Nλ)`.
- `vsini::Real`: Projected rotational velocity (m/s).
- `u1::Real`: Linear limb-darkening coefficient.

Returns:
- `out::CuArray{<:Real,2}`: Convolved matrix on the GPU, same shape as `ys`.

Notes:
- CPU and GPU results differ at the first and last `~vsini/c × λ₀/Δλ` pixels because
  the CPU uses an unpadded circular FFT while the GPU uses a padded linear convolution.

See also: [`convolve_gray_rotation`](@ref), [`gray_rot_kernel`](@ref)
"""
function convolve_gray_rotation_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                                    ys::AA{T,2}, vsini::T, u1::T) where {T<:AF}
    # copy to device — avoid CuArray() wrapper allocations
    if ys isa CA
        copyto!(cmem.ys_gpu, ys)
    else
        copyto!(cmem.ys_gpu, CuArray(ys))
    end
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

    # kernel rows as 1-D views to avoid dim ambiguity
    kernel_row = reshape(@view(cmem.padded_kernel_gpu[1, :]), :)
    shifted_kernel_row = reshape(@view(cmem.shift_kernel_gpu[1, :]), :)
    fill!(kernel_row, zero(T))

    ts1 = (256,)
    bs1 = (cld(cmem.Nλ, ts1[1]),)
    @cuda threads=ts1 blocks=bs1 compute_padded_gray_kernel_1D!(kernel_row,
                                                                cmem.xs_gpu, λ0,
                                                                vsini, u1,
                                                                cmem.Nλ, cmem.pad_left)
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
