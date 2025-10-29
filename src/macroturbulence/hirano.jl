"""
Equation B12 from Hirano et al. (2011). NOTE: This is returns the Fourier
Transform of the rotmacro convolution kernel, not the kernel itself!! 
"""
function hirano_rotmacro_ft_kernel(σs::AA{T,1}, vsini::T, ζ_rt::T; u1::T=0.43, u2::T=0.31, intres::Int=100) where T<:AF
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

function convolve_hirano_rotmacro(xs::AA{T,1}, ys::AA{T,1}, vsini::T, 
                                  ζ_rt::T, u1::T, u2::T; 
                                  intres::Int=100) where T<:AF
    # velocity grid
    N = length(xs)
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0
    Δv = (vs[end] - vs[1]) / (N - 1)
    dv = diff(vs)

    # frequency grid and kernel FT
    σ = FFTW.fftfreq(N) ./ Δv
    Kσ = hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)

    # inverse FT
    K_dft = Kσ ./ Δv
    k_circ = real(ifft(K_dft))              # circular kernel, zero-lag at index 1
    kernel  = FFTW.fftshift(k_circ)          # center the kernel around v=0
    kernel ./= sum(kernel)

    return imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
end

function convolve_hirano_rotmacro(xs::AA{T,1}, ys::AA{T,2}, vsini::T, 
                                  ζ_rt::T, u1::T, u2::T; 
                                  intres::Int=100) where T<:AF
    # velocity grid
    N = length(xs)
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0
    Δv = (vs[end] - vs[1]) / (N - 1)
    dv = diff(vs)

    # frequency grid and kernel FT
    σ = FFTW.fftfreq(N) ./ Δv
    Kσ = hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)

    # inverse FT
    K_dft = Kσ ./ Δv
    k_circ = real(ifft(K_dft))              # circular kernel, zero-lag at index 1
    kernel  = FFTW.fftshift(k_circ)          # center the kernel around v=0
    kernel ./= sum(kernel)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= imfilter(ys[t, :], reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
    end
    return ys_out
end

# === GPU helper: copy a 1-D kernel into a padded 2-D kernel buffer ===
function write_padded_kernel_rows!(k2d, k1d, Nλ, pad_left)
    i = (blockIdx().y-1) * blockDim().y + threadIdx().y
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= size(k2d,1) && j <= Nλ
        @inbounds k2d[i, j + pad_left] = k1d[j]
    end
    return nothing
end

# === CPU helper: build the (centered, normalized) 1-D Hirano kernel from xs ===
function hirano_rotmacro_kernel_from_xs(xs::AA{T,1}, vsini::T, ζ_rt::T; u1::T=0.43, u2::T=0.31, intres::Int=100) where T<:AF
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

# === GPU convolution (2D) using cuFFT plans in `cmem` (same pattern as gray_rt) ===
function convolve_hirano_rotmacro_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, vsini::T, ζ_rt::T,
                                      u1::T, u2::T; intres::Int=100) where {T<:AF}
    # copy inputs to device
    copyto!(cmem.xs_gpu, CuArray(xs))
    copyto!(cmem.ys_gpu, CuArray(ys))

    # short circuit (no broadening)
    if iszero(vsini) && iszero(ζ_rt)
        return cmem.ys_gpu
    end

    # build centered, normalized kernel on CPU (cheaper & reliable than doing the Bessel/integral on GPU)
    kernel_host = hirano_rotmacro_kernel_from_xs(xs, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)
    kernel_gpu = CuArray(kernel_host)

    # pad the signal [Natm × (Nλ + pad_left + pad_right)]
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu, cmem.Nλ, cmem.pad_left, cmem.pad_right)
    CUDA.synchronize()

    # write padded kernel rows (same kernel for every atmospheric layer row)
    fill!(cmem.padded_kernel_gpu, zero(T))
    ts = (32,32)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs write_padded_kernel_rows!(cmem.padded_kernel_gpu, kernel_gpu, cmem.Nλ, cmem.pad_left)
    CUDA.synchronize()

    # (re)normalize on device for numerical safety
    cmem.norm_buffer .= CUDA.sum(cmem.padded_kernel_gpu, dims=2)
    cmem.padded_kernel_gpu ./= cmem.norm_buffer

    # center for FFT-based convolution
    CUDA.CUFFT.ifftshift!(cmem.shift_kernel_gpu, cmem.padded_kernel_gpu, 2)

    # forward FFTs
    mul!(cmem.kernel_ft_gpu, cmem.plan_fwd, cmem.shift_kernel_gpu)
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem
    cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* cmem.kernel_ft_gpu

    # inverse FFT
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)

    # slice valid region [Natm × Nλ]; return a view on device
    out = @view cmem.conv_gpu[:, cmem.pad_left : cmem.pad_left + cmem.Nλ - 1]
    CUDA.synchronize()
    return out
end