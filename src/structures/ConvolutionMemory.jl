"""
    AbstractConvolutionMemory{T<:AbstractFloat}

Abstract supertype for GPU convolution memory structs.  Concrete subtypes:

- [`ConvolutionMemory`](@ref): micro-turbulence path (Doppler filter + batched FFT).
- [`MacroConvolutionMemory`](@ref): macro-turbulence / rotation path (adds 1D kernel
  FFT plans, padded kernel buffers, and an output extraction buffer).
"""
abstract type AbstractConvolutionMemory{T<:AF} end

"""
    ConvolutionMemory{T<:AbstractFloat}

Pre-allocated GPU buffers and CUFFT plans for batched FFT-based Doppler convolution
(micro-turbulence path).

See also: [`MacroConvolutionMemory`](@ref), [`ConvolutionMemory(Nλ, Natm, Npad)`](@ref)
"""
mutable struct ConvolutionMemory{T<:AF} <: AbstractConvolutionMemory{T}
    Nλ::Int
    Natm::Int
    Npad::Int
    L::Int
    pad_left::Int
    pad_right::Int
    doppler_scale::T
    doppler_ready::Bool
    signal_cached::Bool

    # input signal buffer
    ys_gpu::CA{T,2}

    # padded signal
    signal_gpu::CA{T,2}

    # Fourier-domain buffers
    kernel_ft_gpu::CuMatrix{Complex{T}}
    signal_ft_gpu::CuMatrix{Complex{T}}
    conv_ft_gpu::CuMatrix{Complex{T}}
    conv_gpu::CuMatrix{T}

    # 2D batched FFT plans (along dim=2, wavelength axis)
    plan_fwd::CUDA.CUFFT.CuFFTPlan
    plan_bwd::AbstractFFTs.ScaledPlan

    # 1D R2C FFT for real-space micro kernel (Tier 1: uniform v_mic)
    xs_gpu::CA{T,1}                        # wavelength grid (Nλ)
    kr_1d::CA{T,1}                         # real kernel buffer (L)
    kernel_row_ft_1d::CuVector{Complex{T}} # FFT of 1D base kernel (nfreq)
    plan_fwd_1d::CUDA.CUFFT.CuFFTPlan     # 1D R2C plan on kr_1d
    kernel_cached::Bool                     # base kernel FT is valid
end

"""
    MacroConvolutionMemory{T<:AbstractFloat}

Pre-allocated GPU buffers and CUFFT plans for macro-turbulence / rotation convolution.
Extends the base convolution buffers with 1D kernel FFT infrastructure, padded kernel
work vectors, and an output extraction buffer.

See also: [`ConvolutionMemory`](@ref), [`MacroConvolutionMemory(Nλ, Natm, Npad)`](@ref)
"""
mutable struct MacroConvolutionMemory{T<:AF} <: AbstractConvolutionMemory{T}
    # ── shared fields (same layout as ConvolutionMemory) ──
    Nλ::Int
    Natm::Int
    Npad::Int
    L::Int
    pad_left::Int
    pad_right::Int
    doppler_scale::T
    doppler_ready::Bool
    signal_cached::Bool

    ys_gpu::CA{T,2}
    signal_gpu::CA{T,2}

    kernel_ft_gpu::CuMatrix{Complex{T}}
    signal_ft_gpu::CuMatrix{Complex{T}}
    conv_ft_gpu::CuMatrix{Complex{T}}
    conv_gpu::CuMatrix{T}

    plan_fwd::CUDA.CUFFT.CuFFTPlan
    plan_bwd::AbstractFFTs.ScaledPlan

    # ── macro-specific fields ──
    xs_gpu::CA{T,1}                        # wavelength grid for kernel evaluation (Nλ)
    padded_kernel_gpu::CA{T,1}             # 1D padded kernel work buffer (L)
    shift_kernel_gpu::CA{T,1}              # 1D shifted kernel work buffer (L)
    out_gpu::CA{T,2}                       # output extraction buffer (Natm, Nλ)

    # 1D R2C FFT for macro kernel
    kr_1d::CA{T,1}                         # real kernel buffer (L)
    kernel_row_ft_1d::CuVector{Complex{T}} # FFT of 1D kernel (nfreq)
    plan_fwd_1d::CUDA.CUFFT.CuFFTPlan
    kernel_cached::Bool                     # micro base kernel FT is valid

    # 1D C2C infrastructure for Hirano kernel
    kc_1d::CuVector{Complex{T}}            # complex buffer (Nλ)
    plan_bwd_1d::AbstractFFTs.ScaledPlan
end

# ── helpers ────────────────────────────────────────────────────────────────────

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

# shared padding/FFT geometry computation
function _conv_mem_geometry(Nλ::Int, Npad::Int)
    Nλ > 0 || error("Nλ must be positive")
    Npad >= 0 || error("Npad must be non-negative")
    requested_L = Nλ + Npad
    requested_L > Nλ || error("requested padded length must exceed Nλ")
    L = next_fft_friendly_len(requested_L)
    Npad_eff = L - Nλ
    Npad_eff >= 2 || error("effective Npad must be >= 2, got $Npad_eff")
    pad_left = Npad_eff ÷ 2
    pad_right = Npad_eff - pad_left
    (pad_left + Nλ + pad_right == L) || error("padding split inconsistent with L")
    pad_left >= 1 || error("pad_left must be >= 1, got $pad_left")
    return L, Npad_eff, pad_left, pad_right
end

# ── CPU padded linear convolution (R2C FFT) ──────────────────────────────────

"""
    _pad_edges!(dst, src, pad_left, Nλ)

Fill length-L vector `dst` with edge-replicated padding of `src`:
left pad = src[1], right pad = src[Nλ].
"""
function _pad_edges!(dst::Vector{T}, src, pad_left::Int, Nλ::Int) where T
    @inbounds for j in 1:pad_left
        dst[j] = src[1]
    end
    @inbounds for j in 1:Nλ
        dst[pad_left + j] = src[j]
    end
    L = length(dst)
    @inbounds for j in (pad_left + Nλ + 1):L
        dst[j] = src[Nλ]
    end
    return nothing
end

"""
    _kernel_to_dft_layout!(kbuf, kernel, i0)

Place a length-Nλ `kernel` (with zero-lag at index `i0`) into a length-L DFT-ordered
buffer `kbuf` (zero-lag at index 1, positive lags next, negative lags at end).
"""
function _kernel_to_dft_layout!(kbuf::Vector{T}, kernel::Vector{T}, i0::Int) where T
    L = length(kbuf)
    fill!(kbuf, zero(T))
    @inbounds for j in eachindex(kernel)
        d = j - i0
        idx = d >= 0 ? d + 1 : L + d + 1
        kbuf[idx] = kernel[j]
    end
    return nothing
end

"""
    _padded_convolve(ys::Vector, kernel::Vector; Npad=512)
    _padded_convolve(ys::Matrix, kernel::Vector; Npad=512)

CPU padded linear convolution with edge replication using R2C FFT.
Matches the GPU convolution strategy (padded signal, zero-padded kernel,
extract valid region). Returns an array the same size as `ys`.
"""
function _padded_convolve(ys::Vector{T}, kernel::Vector{T}; Npad::Int=512) where T<:AF
    Nλ = length(ys)
    L, _, pad_left, _ = _conv_mem_geometry(Nλ, Npad)
    i0 = Nλ ÷ 2 + 1

    # pad signal with edge replication
    sig = zeros(T, L)
    _pad_edges!(sig, ys, pad_left, Nλ)

    # place kernel in DFT layout (zero-lag at index 1)
    kbuf = zeros(T, L)
    _kernel_to_dft_layout!(kbuf, kernel, i0)

    # R2C convolution
    sig_ft = rfft(sig)
    ker_ft = rfft(kbuf)
    sig_ft .*= ker_ft
    conv = irfft(sig_ft, L)

    # extract valid region
    return conv[pad_left+1 : pad_left+Nλ]
end

function _padded_convolve(ys::Matrix{T}, kernel::Vector{T}; Npad::Int=512) where T<:AF
    Nλ = size(ys, 2)
    Nrows = size(ys, 1)
    L, _, pad_left, _ = _conv_mem_geometry(Nλ, Npad)
    i0 = Nλ ÷ 2 + 1

    # compute kernel FT once
    kbuf = zeros(T, L)
    _kernel_to_dft_layout!(kbuf, kernel, i0)
    ker_ft = rfft(kbuf)

    # convolve each row
    sig = zeros(T, L)
    out = zeros(T, Nrows, Nλ)
    for t in 1:Nrows
        _pad_edges!(sig, view(ys, t, :), pad_left, Nλ)
        sig_ft = rfft(sig)
        sig_ft .*= ker_ft
        conv = irfft(sig_ft, L)
        @inbounds for j in 1:Nλ
            out[t, j] = conv[pad_left + j]
        end
    end
    return out
end

# ── constructors ───────────────────────────────────────────────────────────────

"""
    ConvolutionMemory(Nλ, Natm, Npad; T=Float64)

Allocate GPU buffers and CUFFT plans for micro-turbulence convolution of
`Natm × Nλ` matrices.
"""
function ConvolutionMemory(Nλ::Int, Natm::Int, Npad::Int; T=Float64)
    Natm > 0 || error("Natm must be positive")
    L, Npad_eff, pad_left, pad_right = _conv_mem_geometry(Nλ, Npad)
    nfreq = fld(L, 2) + 1

    ys_gpu     = CUDA.zeros(T, Natm, Nλ)
    signal_gpu = CUDA.zeros(T, Natm, L)

    kernel_ft_gpu = CuMatrix{Complex{T}}(undef, Natm, nfreq)
    signal_ft_gpu = similar(kernel_ft_gpu)
    conv_ft_gpu   = similar(kernel_ft_gpu)
    conv_gpu      = CuMatrix{T}(undef, Natm, L)

    plan_fwd = CUDA.CUFFT.plan_rfft(signal_gpu, 2)
    plan_bwd = CUDA.CUFFT.plan_irfft(conv_ft_gpu, L, 2)

    # 1D kernel infrastructure
    xs_gpu           = CUDA.zeros(T, Nλ)
    kr_1d            = CUDA.zeros(T, L)
    kernel_row_ft_1d = CuVector{Complex{T}}(undef, nfreq)
    plan_fwd_1d      = CUDA.CUFFT.plan_rfft(kr_1d)

    return ConvolutionMemory{T}(Nλ, Natm, Npad_eff, L, pad_left, pad_right,
                                zero(T), false, false,
                                ys_gpu, signal_gpu,
                                kernel_ft_gpu, signal_ft_gpu, conv_ft_gpu, conv_gpu,
                                plan_fwd, plan_bwd,
                                xs_gpu, kr_1d, kernel_row_ft_1d, plan_fwd_1d, false)
end

"""
    MacroConvolutionMemory(Nλ, Natm, Npad; T=Float64)

Allocate GPU buffers and CUFFT plans for macro-turbulence / rotation convolution
of `Natm × Nλ` matrices. Includes 1D kernel FFT infrastructure.
"""
function MacroConvolutionMemory(Nλ::Int, Natm::Int, Npad::Int; T=Float64)
    Natm > 0 || error("Natm must be positive")
    L, Npad_eff, pad_left, pad_right = _conv_mem_geometry(Nλ, Npad)
    nfreq = fld(L, 2) + 1

    # shared buffers
    ys_gpu     = CUDA.zeros(T, Natm, Nλ)
    signal_gpu = CUDA.zeros(T, Natm, L)

    kernel_ft_gpu = CuMatrix{Complex{T}}(undef, Natm, nfreq)
    signal_ft_gpu = similar(kernel_ft_gpu)
    conv_ft_gpu   = similar(kernel_ft_gpu)
    conv_gpu      = CuMatrix{T}(undef, Natm, L)

    plan_fwd = CUDA.CUFFT.plan_rfft(signal_gpu, 2)
    plan_bwd = CUDA.CUFFT.plan_irfft(conv_ft_gpu, L, 2)

    # macro-specific buffers
    xs_gpu             = CUDA.zeros(T, Nλ)
    padded_kernel_gpu  = CUDA.zeros(T, L)
    shift_kernel_gpu   = CUDA.zeros(T, L)
    out_gpu            = CUDA.zeros(T, Natm, Nλ)

    kr_1d              = CUDA.zeros(T, L)
    kernel_row_ft_1d   = CuVector{Complex{T}}(undef, nfreq)
    plan_fwd_1d        = CUDA.CUFFT.plan_rfft(kr_1d)

    kc_1d              = CuVector{Complex{T}}(undef, Nλ)
    plan_bwd_1d        = CUDA.CUFFT.plan_ifft!(kc_1d)

    return MacroConvolutionMemory{T}(Nλ, Natm, Npad_eff, L, pad_left, pad_right,
                                     zero(T), false, false,
                                     ys_gpu, signal_gpu,
                                     kernel_ft_gpu, signal_ft_gpu, conv_ft_gpu, conv_gpu,
                                     plan_fwd, plan_bwd,
                                     xs_gpu, padded_kernel_gpu, shift_kernel_gpu, out_gpu,
                                     kr_1d, kernel_row_ft_1d, plan_fwd_1d, false,
                                     kc_1d, plan_bwd_1d)
end
