const intres_glob = 1024

"""
    hirano_rotmacro_ft_kernel(σs, vsini, ζ_rt; u1=0.43, u2=0.31, intres=intres_glob)

Compute the Fourier transform of the Hirano et al. (2011) rotation+macroturbulence
kernel (Eq. B12) by quadrature integration over the visible stellar disk.

Arguments:
- `σs::AbstractVector{<:Real}`: Frequency grid (s/m; inverse of velocity units).
- `vsini::Real`: Projected rotational velocity (m/s).
- `ζ_rt::Real`: Radial-tangential macroturbulence velocity scale (m/s).
- `u1::Real=0.43`: Linear limb-darkening coefficient.
- `u2::Real=0.31`: Quadratic limb-darkening coefficient.
- `intres::Int=intres_glob`: Number of quadrature points for the disk integral.

Returns:
- `Kσ::Vector{<:Real}`: Fourier transform of the combined kernel, same length as `σs`.

See also: [`convolve_hirano_rotmacro`](@ref)
"""
function hirano_rotmacro_ft_kernel(σs::AA{T,1}, vsini::T, ζ_rt::T; u1::T=0.43, u2::T=0.31, intres::Int=intres_glob) where T<:AF
    # quadrature grid in t∈[0,1]
    t = reshape(collect(range(zero(T), one(T), length=intres)), :, 1)
    dt = t[2] - t[1]

    # limb-darkening factor (normalized)
    μ = sqrt.(max.(zero(T), one(T) .- t.^2))  # guard tiny negatives at t≈1
    t1 = (one(T) .- u1 .* (one(T) .- μ) .- u2 .* (one(T) .- μ).^2) ./ (one(T) .- u1/3 - u2/6)

    # macroturbulence × rotation factor
    a = (π^2) * (ζ_rt^2) .* (σs.^2)

    # assemble the integrand
    expterm = exp.(-a .* (one(T) .- t.^2)') .+ exp.(-a .* (t.^2)')
    j0 = besselj0.(2π .* σs .* vsini .* t')
    m = t1' .* expterm .* j0 .* t'

    # integrate
    s = sum(m; dims=2) .- 0.5 .* (m[:, 1] .+ m[:, end])
    return vec(s) .* dt
end

"""
    convolve_hirano_rotmacro(xs, ys, vsini, ζ_rt, u1, u2; intres=intres_glob)

Convolve a spectrum with the Hirano et al. (2011) combined rotation+macroturbulence
kernel. The kernel is computed analytically in the Fourier domain and applied via FFT.

Arguments:
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractArray{<:Real}`: Spectrum on `xs` (vector or matrix with rows as spectra).
- `vsini::Real`: Projected rotational velocity (m/s).
- `ζ_rt::Real`: Radial-tangential macroturbulence velocity scale (m/s).
- `u1::Real`: Linear limb-darkening coefficient.
- `u2::Real`: Quadratic limb-darkening coefficient.
- `intres::Int=intres_glob`: Number of quadrature points for the disk integral.

Returns:
- `ys_out::AbstractArray{<:Real}`: Convolved spectrum with the same shape as `ys`.

See also: [`hirano_rotmacro_ft_kernel`](@ref), [`convolve_hirano_rotmacro_gpu`](@ref)
"""
function convolve_hirano_rotmacro(xs::AA{T,1}, ys::AA{T,1}, vsini::T,
                                  ζ_rt::T, u1::T, u2::T;
                                  intres::Int=intres_glob) where T<:AF
    # velocity grid
    N = length(xs)
    i0 = N ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0
    Δv = (vs[end] - vs[1]) / (N - 1)
    dv = diff(vs)

    # frequency grid and kernel FT
    σ = FFTW.fftfreq(N) ./ Δv
    Kσ = hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)

    # inverse FT → circular kernel with zero-lag at index 1 (GPU-style phase)
    K_dft = Kσ ./ Δv
    k_circ = real(ifft(K_dft))
    k_circ ./= sum(k_circ)

    # convolution via FFT (matches GPU convention)
    return real(ifft(fft(ys) .* fft(k_circ)))
end

function convolve_hirano_rotmacro(xs::AA{T,1}, ys::AA{T,2}, vsini::T,
                                  ζ_rt::T, u1::T, u2::T;
                                  intres::Int=intres_glob) where T<:AF
    # velocity grid
    N = length(xs)
    i0 = N ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0
    Δv = (vs[end] - vs[1]) / (N - 1)
    dv = diff(vs)

    # frequency grid and kernel FT
    σ = FFTW.fftfreq(N) ./ Δv
    Kσ = hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)

    # inverse FT → circular kernel with zero-lag at index 1 (GPU-style phase)
    K_dft = Kσ ./ Δv
    k_circ = real(ifft(K_dft))
    k_circ ./= sum(k_circ)
    ftk = fft(k_circ)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= real(ifft(fft(ys[t, :]) .* ftk))
    end
    return ys_out
end

function hirano_rotmacro_kernel_from_xs(xs::AA{T,1}, vsini::T, ζ_rt::T; u1::T=0.43, u2::T=0.31, intres::Int=intres_glob) where T<:AF
    N = length(xs)
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0
    Δv = (vs[end] - vs[1]) / (N - 1)

    σ = FFTW.fftfreq(N) ./ Δv
    Kσ = hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)

    K_dft = Kσ ./ Δv
    k_circ = real(ifft(K_dft))      # circular kernel, zero-lag at index 1
    kernel = FFTW.fftshift(k_circ)  # center at v=0
    kernel ./= sum(kernel)
    return kernel
end

"""
    convolve_hirano_rotmacro_gpu(cmem, xs, ys, vsini, ζ_rt, u1, u2; intres=intres_glob)

GPU implementation of [`convolve_hirano_rotmacro`](@ref). Convolves each row of `ys`
with the Hirano et al. (2011) combined rotation+macroturbulence kernel using padded
FFT convolution on the device.

Arguments:
- `cmem::ConvolutionMemory`: Pre-allocated GPU working memory.
- `xs::AbstractVector{<:Real}`: Wavelength grid (Å).
- `ys::AbstractMatrix{<:Real}`: Input matrix with shape `(Natm, Nλ)`.
- `vsini::Real`: Projected rotational velocity (m/s).
- `ζ_rt::Real`: Radial-tangential macroturbulence velocity scale (m/s).
- `u1::Real`: Linear limb-darkening coefficient.
- `u2::Real`: Quadratic limb-darkening coefficient.
- `intres::Int=intres_glob`: Quadrature resolution for the kernel integral.

Returns:
- `out::CuArray{<:Real,2}`: Convolved matrix on the GPU, same shape as `ys`.

Notes:
- Short-circuits (returns `CuArray(ys)`) when both `vsini` and `ζ_rt` are zero.
- The kernel is computed on the CPU and transferred to the GPU; the FFT convolution
  runs on the device.
- CPU and GPU results differ at the first and last `~vsini/c × λ₀/Δλ` pixels because
  the CPU uses an unpadded circular FFT while the GPU uses a padded linear convolution.

See also: [`convolve_hirano_rotmacro`](@ref), [`hirano_rotmacro_ft_kernel`](@ref)
"""
function convolve_hirano_rotmacro_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, vsini::T, ζ_rt::T,
                                      u1::T, u2::T; intres::Int=intres_glob) where {T<:AF}
    # short circuit before any copy so the caller's ys is returned unmodified
    if iszero(vsini) && iszero(ζ_rt)
        return CuArray(ys)
    end

    # copy to device — avoid CuArray() wrapper allocations
    if ys isa CA
        copyto!(cmem.ys_gpu, ys)
    else
        copyto!(cmem.ys_gpu, CuArray(ys))
    end

    # get CPU-side wavelength info (avoid scalar indexing of CuArray)
    xs_h = xs isa CA ? Array(xs) : xs

    # circular kernel centered at v=0 (FFT-shifted) with sum=1
    kernel_N = hirano_rotmacro_kernel_from_xs(xs_h, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)

    # pad the signal
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)

    # place kernel into padded work buffer on device
    kernel_row = reshape(@view(cmem.padded_kernel_gpu[1, :]), :)
    shifted_kernel_row = reshape(@view(cmem.shift_kernel_gpu[1, :]), :)

    fill!(kernel_row, zero(T))
    kdev = CuArray(kernel_N)
    @views kernel_row[cmem.pad_left+1 : cmem.pad_left+cmem.Nλ] .= kdev

    # ensure zero-lag sits at padded center before FFT layout
    i0 = length(xs_h) ÷ 2 + 1
    center = length(kernel_row) ÷ 2
    r = center - (cmem.pad_left + i0)
    if r != 0
        ts1 = (256,)
        @cuda threads=ts1 blocks=(cld(length(kernel_row), ts1[1]),) roll_1d!(shifted_kernel_row, kernel_row, r, length(kernel_row))
        CUDA.synchronize()
        tmp = kernel_row
        kernel_row = shifted_kernel_row
        shifted_kernel_row = tmp
    end

    # center -> FFT indexing, then R2C FFT of padded kernel (no allocation)
    CUDA.CUFFT.ifftshift!(shifted_kernel_row, kernel_row, 1)
    copyto!(cmem.kr_1d, shifted_kernel_row)
    mul!(cmem.kernel_row_ft_1d, cmem.plan_fwd_1d, cmem.kr_1d)

    # convolution theorem
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)
    kft = reshape(cmem.kernel_row_ft_1d, 1, :)
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* kft
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # extract valid region into pre-allocated output buffer
    ts2 = (32, 32)
    bs2 = (cld(cmem.Natm, ts2[1]), cld(cmem.Nλ, ts2[2]))
    @cuda threads=ts2 blocks=bs2 extract_valid!(cmem.out_gpu, cmem.conv_gpu,
                                                 cmem.pad_left, cmem.Nλ)
    return cmem.out_gpu
end
