mutable struct ConvolutionMemory{T<:AF}
    Nλ::Int
    Natm::Int
    Npad::Int
    L::Int
    pad_left::Int
    pad_right::Int

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

function ConvolutionMemory(Nλ::Int, Natm::Int, Npad::Int; T=Float64)
    # get dims
    L = Nλ + Npad
    pad_left = Npad ÷ 2
    pad_right = L - Nλ - pad_left

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
    return ConvolutionMemory(Nλ, Natm, Npad, L, pad_left, pad_right,
                             xs_gpu, ys_gpu, λc_gpu, σ_fac_gpu, λc_vec,
                             σ_fac_vec, σ_v_cpu, μ_v_cpu, signal_gpu, 
                             kernel_gpu, padded_kernel_gpu, shift_kernel_gpu,
                             norm_buffer, kernel_ft_gpu, signal_ft_gpu, 
                             conv_ft_gpu, conv_gpu, plan_fwd, plan_bwd)
end
