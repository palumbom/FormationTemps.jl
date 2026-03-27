"""
    ConvolutionMemory{T<:AbstractFloat}

Pre-allocated GPU buffers and CUFFT plans for batched FFT-based spectral convolution.

Caches a padded signal FFT across convolution calls so that repeated convolutions of the
same underlying absorption coefficients (e.g., across disk integration tiles that vary only
in μ and velocity) can reuse the signal transform. Set `signal_cached = true` after the
first call to skip re-padding and re-transforming on subsequent calls.

Fields:
- `Nλ`: Number of wavelength points in the unpadded signal.
- `Natm`: Number of atmosphere layers (rows in the signal matrix).
- `Npad`: Effective number of padding samples added (split symmetrically on each side).
- `L`: Total padded FFT length (FFT-friendly: factors of 2, 3, 5, 7).
- `pad_left`, `pad_right`: Number of padding samples on the left and right of the signal.
- `doppler_scale`: Cached wavelength-to-pixel conversion factor `λ₀ / (c Δλ)`.
- `doppler_ready`: Whether `doppler_scale` has been computed for the current wavelength grid.
- `signal_cached`: Whether the padded signal FFT in `signal_ft_gpu` is current.

See also: [`ConvolutionMemory(Nλ, Natm, Npad)`](@ref)
"""
mutable struct ConvolutionMemory{T<:AF}
    Nλ::Int
    Natm::Int
    Npad::Int
    L::Int
    pad_left::Int
    pad_right::Int
    doppler_scale::T
    doppler_ready::Bool
    signal_cached::Bool

    # coordinates and spectra
    xs_gpu::CA{T,1}
    ys_gpu::CA{T,2}
    λc_gpu::CA{T,1}
    σ_fac_gpu::CA{T,1}
    λc_vec::AA{T,1}
    σ_fac_vec::AA{T,1}

    # cpu buffer 
    σ_v_cpu::AA{T,1}
    μ_v_cpu::AA{T,1}

    # working buffers
    signal_gpu::CA{T,2}
    kernel_gpu::CA{T,2}
    padded_kernel_gpu::CA{T,2}
    shift_kernel_gpu::CA{T,2}
    norm_buffer::CA{T,1}

    # Fourier-domain buffers
    kernel_ft_gpu::CuMatrix{Complex{T}}
    signal_ft_gpu::CuMatrix{Complex{T}}
    conv_ft_gpu::CuMatrix{Complex{T}}
    conv_gpu::CuMatrix{T}

    # FFT plans (2D batched, along dim=2)
    plan_fwd::CUDA.CUFFT.CuFFTPlan
    plan_bwd::AbstractFFTs.ScaledPlan

    # 1D FFT infrastructure for macroturbulence kernel (shared across rows)
    kr_1d::CA{T,1}                       # real kernel buffer, length L
    kernel_row_ft_1d::CuVector{Complex{T}} # FFT of 1D kernel, length nfreq
    plan_fwd_1d::CUDA.CUFFT.CuFFTPlan    # 1D R2C plan

    # pre-allocated output buffer for convolution result (Natm × Nλ, unpadded)
    out_gpu::CA{T,2}
end

function is_fft_friendly_len(L::Int)
    n = L
    for p in (2, 3, 5, 7)
        while n % p == 0
            n ÷= p
        end
    end
    return n == 1
end

function next_fft_friendly_len(L::Int)
    L_candidate = L
    while !is_fft_friendly_len(L_candidate)
        L_candidate += 1
    end
    return L_candidate
end

"""
    ConvolutionMemory(Nλ, Natm, Npad; T=Float64)

Allocate GPU buffers and CUFFT plans for batched convolution of `Natm × Nλ` matrices.

`Npad` is the minimum number of padding samples added to the signal on each side; the
actual total padding is rounded up so that the padded length `L` is FFT-friendly
(factors of 2, 3, 5, 7 only). The effective `Npad` stored in the struct is `L - Nλ`.
"""
function ConvolutionMemory(Nλ::Int, Natm::Int, Npad::Int; T=Float64)
    Nλ > 0 || error("Nλ must be positive")
    Natm > 0 || error("Natm must be positive")
    Npad >= 0 || error("Npad must be non-negative")

    # choose an FFT-friendly padded length
    requested_L = Nλ + Npad
    requested_L > Nλ || error("requested padded length must exceed Nλ")
    L = next_fft_friendly_len(requested_L)
    Npad_eff = L - Nλ
    Npad_eff >= 2 || error("effective Npad must be >= 2, got $Npad_eff")

    # split padding
    pad_left = Npad_eff ÷ 2
    pad_right = Npad_eff - pad_left
    (pad_left + Nλ + pad_right == L) || error("padding split inconsistent with L")

    # this is used for shift clamping in the Doppler filter kernel
    (pad_left - 1) >= 0 || error("pad_left must be >= 1")
    doppler_scale = zero(T)
    doppler_ready = false
    signal_cached = false

    # allocate inputs
    xs_gpu = CUDA.zeros(T, Nλ)
    ys_gpu = CUDA.zeros(T, Natm, Nλ)
    λc_gpu = CUDA.zeros(T, Natm)
    σ_fac_gpu = CUDA.zeros(T, Natm)
    λc_vec = zeros(Natm)
    σ_fac_vec = zeros(Natm)

    # cpu buffer 
    σ_v_cpu = zeros(Natm)
    μ_v_cpu = zeros(Natm)

    # allocate for padded kernels
    signal_gpu = CUDA.zeros(T, Natm, L)
    kernel_gpu = CUDA.zeros(T, Natm, L)
    padded_kernel_gpu = CUDA.zeros(T, Natm, L)
    shift_kernel_gpu = CUDA.zeros(T, Natm, L)
    norm_buffer = CUDA.zeros(T, Natm)

    # Fourier buffers
    nfreq = fld(L, 2) + 1
    kernel_ft_gpu = CuMatrix{Complex{T}}(undef, Natm, nfreq)
    signal_ft_gpu = similar(kernel_ft_gpu)
    conv_ft_gpu = similar(kernel_ft_gpu)
    conv_gpu = CuMatrix{T}(undef, Natm, L)

    # plan FFTs along dim=2 (wavelength axis)
    plan_fwd = CUDA.CUFFT.plan_rfft(signal_gpu, 2)
    plan_bwd = CUDA.CUFFT.plan_irfft(conv_ft_gpu, L, 2)

    # 1D FFT infrastructure for macroturbulence kernel
    kr_1d = CUDA.zeros(T, L)
    kernel_row_ft_1d = CuVector{Complex{T}}(undef, nfreq)
    plan_fwd_1d = CUDA.CUFFT.plan_rfft(kr_1d)

    # pre-allocated output buffer (unpadded dimensions)
    out_gpu = CUDA.zeros(T, Natm, Nλ)

    # construct and return
    return ConvolutionMemory(Nλ, Natm, Npad_eff, L, pad_left, pad_right,
                             doppler_scale, doppler_ready, signal_cached,
                             xs_gpu, ys_gpu, λc_gpu, σ_fac_gpu, λc_vec,
                             σ_fac_vec, σ_v_cpu, μ_v_cpu, signal_gpu,
                             kernel_gpu, padded_kernel_gpu, shift_kernel_gpu,
                             norm_buffer, kernel_ft_gpu, signal_ft_gpu,
                             conv_ft_gpu, conv_gpu, plan_fwd, plan_bwd,
                             kr_1d, kernel_row_ft_1d, plan_fwd_1d, out_gpu)
end
