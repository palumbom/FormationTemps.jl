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

    # FFT plans
    plan_fwd::CUDA.CUFFT.CuFFTPlan
    plan_bwd::AbstractFFTs.ScaledPlan
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
    
    # construct and return
    return ConvolutionMemory(Nλ, Natm, Npad_eff, L, pad_left, pad_right,
                             doppler_scale, doppler_ready, signal_cached,
                             xs_gpu, ys_gpu, λc_gpu, σ_fac_gpu, λc_vec,
                             σ_fac_vec, σ_v_cpu, μ_v_cpu, signal_gpu, 
                             kernel_gpu, padded_kernel_gpu, shift_kernel_gpu,
                             norm_buffer, kernel_ft_gpu, signal_ft_gpu, 
                             conv_ft_gpu, conv_gpu, plan_fwd, plan_bwd)
end
