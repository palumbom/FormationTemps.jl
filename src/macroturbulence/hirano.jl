global const intres_glob = 500

"""
Equation B12 from Hirano et al. (2011). NOTE: This is returns the Fourier
Transform of the rotmacro convolution kernel, not the kernel itself!! 
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

function convolve_hirano_rotmacro_gpu(cmem::ConvolutionMemory, xs::AA{T,1},
                                      ys::AA{T,2}, vsini::T, ζ_rt::T,
                                      u1::T, u2::T; intres::Int=intres_glob) where {T<:AF}
    # copy to device
    copyto!(cmem.xs_gpu, CuArray(xs))
    copyto!(cmem.ys_gpu, CuArray(ys))

    # short circuit
    if iszero(vsini) && iszero(ζ_rt)
        return cmem.ys_gpu
    end

    # velocity grid stats from discrete center
    N = length(xs)
    i0 = N ÷ 2 + 1
    λ0 = xs[i0]
    vs = c_ms .* (xs .- λ0) ./ λ0
    Δv = (vs[end] - vs[1]) / (N - 1)

    # pad the signal (replicate) to length L
    ts = (32,32)
    bs = (cld(cmem.Natm, ts[1]), cld(cmem.L, ts[2]))
    @cuda threads=ts blocks=bs pad_signal!(cmem.signal_gpu, cmem.ys_gpu,
                                           cmem.Nλ, cmem.pad_left, cmem.pad_right)
    CUDA.synchronize()

    # R2C frequency grid for padded length L
    L = cmem.L
    nf = fld(L, 2) + 1
    invLΔv = one(T) / (T(L) * Δv)
    σ = Array{T}(undef, nf)
    @inbounds for k in 0:nf-1
        σ[k+1] = T(k) * invLΔv
    end

    # H(σ) on padded grid; normalize DC gain to 1 (matches CPU's sum(normalized kernel)=1)
    Kσ = hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)
    Kd_host = Kσ ./ Kσ[1]  # Δv cancels in K_dft/K_dft[1]

    # phase to place zero-lag at padded center (remove integer roll)
    center = L ÷ 2
    r = center - (cmem.pad_left + i0)  # integer offset
    twopi = T(2π)
    θstep = -twopi * T(r) / T(L)
    ks = T.(0:nf-1)
    phase = Complex{T}.(cos.(θstep .* ks), sin.(θstep .* ks))

    # frequency response with phase correction on GPU
    Kd = CuArray(Kd_host .* phase)

    # forward fourier transform of padded signal (R2C)
    mul!(cmem.signal_ft_gpu, cmem.plan_fwd, cmem.signal_gpu)

    # convolution theorem (broadcast along rows)
    @views cmem.conv_ft_gpu .= cmem.signal_ft_gpu .* reshape(Kd, 1, :)

    # inverse fourier transform and slice valid region
    mul!(cmem.conv_gpu, cmem.plan_bwd, cmem.conv_ft_gpu)
    out = @view cmem.conv_gpu[:, cmem.pad_left : cmem.pad_left + cmem.Nλ - 1]
    CUDA.synchronize()
    return out
end
