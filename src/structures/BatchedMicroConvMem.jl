"""
    BatchedMicroConvMem{T<:AbstractFloat}

Pre-allocated GPU buffers and cuFFT plans for batched Doppler convolution of B tiles
simultaneously. The signal FFT is shared across tiles (absorption is tile-independent);
only the Doppler filter and convolution product scale with B.

Layout: first dimension is `B*Natm` for per-tile buffers, `Natm` for shared buffers.
Tile `b` occupies rows `(b-1)*Natm+1 : b*Natm`.

See also: [`ConvolutionMemory`](@ref), [`BatchedMicroConvMem(Nλ, Natm, B, Npad)`](@ref)
"""
mutable struct BatchedMicroConvMem{T<:AF} <: AbstractConvolutionMemory{T}
    # ── geometry ──
    Nλ::Int
    Natm::Int                  # per-tile atmosphere layers
    B::Int                     # max batch size
    Npad::Int
    L::Int
    pad_left::Int
    pad_right::Int
    doppler_scale::T
    doppler_ready::Bool
    signal_cached::Bool

    # ── shared signal (Natm rows — same absorption for all tiles) ──
    ys_gpu::CA{T,2}                        # (Natm, Nλ) input buffer
    signal_gpu::CA{T,2}                    # (Natm, L) padded signal
    signal_ft_gpu::CuMatrix{Complex{T}}    # (Natm, nfreq) cached forward FFT
    plan_fwd::CUDA.CUFFT.CuFFTPlan         # R2C on (Natm, L)

    # ── per-tile batched buffers (B*Natm rows) ──
    kernel_ft_gpu::CuMatrix{Complex{T}}    # (B*Natm, nfreq) Doppler filter
    conv_ft_gpu::CuMatrix{Complex{T}}      # (B*Natm, nfreq) convolution product
    conv_gpu::CuMatrix{T}                  # (B*Natm, L) inverse FFT result
    plan_bwd::AbstractFFTs.ScaledPlan      # C2R on (B*Natm, L)

    # ── 1D R2C for real-space micro kernel (Tier 1: uniform v_mic) ──
    xs_gpu::CA{T,1}                        # wavelength grid (Nλ)
    xs_cpu::Vector{T}                      # CPU cache of wavelength grid, set once in _init_micro_params!
    kr_1d::CA{T,1}                         # real kernel buffer (L)
    kernel_row_ft_1d::CuVector{Complex{T}} # FFT of 1D base kernel (nfreq)
    plan_fwd_1d::CUDA.CUFFT.CuFFTPlan     # 1D R2C plan on kr_1d
    kernel_cached::Bool                     # base kernel FT is valid

    # ── batched R2C for per-row kernels (Tier 2: varying v_mic) ──
    plan_fwd_kernel::CUDA.CUFFT.CuFFTPlan  # R2C on (B*Natm, L)
end

"""
    BatchedMicroConvMem(Nλ, Natm, B, Npad; T=Float64)

Allocate GPU buffers for batched Doppler convolution of `B` tiles, each with
`Natm` atmosphere rows and `Nλ` wavelength points.
"""
function BatchedMicroConvMem(Nλ::Int, Natm::Int, B::Int, Npad::Int; T=Float64)
    Natm > 0 || error("Natm must be positive")
    B > 0 || error("B must be positive")
    L, Npad_eff, pad_left, pad_right = _conv_mem_geometry(Nλ, Npad)
    nfreq = fld(L, 2) + 1
    BNatm = B * Natm

    # shared signal buffers (Natm rows)
    ys_gpu        = CUDA.zeros(T, Natm, Nλ)
    signal_gpu    = CUDA.zeros(T, Natm, L)
    signal_ft_gpu = CuMatrix{Complex{T}}(undef, Natm, nfreq)
    plan_fwd      = CUDA.CUFFT.plan_rfft(signal_gpu, 2)

    # per-tile batched buffers (B*Natm rows)
    kernel_ft_gpu = CuMatrix{Complex{T}}(undef, BNatm, nfreq)
    conv_ft_gpu   = CuMatrix{Complex{T}}(undef, BNatm, nfreq)
    conv_gpu      = CuMatrix{T}(undef, BNatm, L)
    plan_bwd      = CUDA.CUFFT.plan_irfft(conv_ft_gpu, L, 2)

    # 1D kernel infrastructure (Tier 1)
    xs_gpu           = CUDA.zeros(T, Nλ)
    kr_1d            = CUDA.zeros(T, L)
    kernel_row_ft_1d = CuVector{Complex{T}}(undef, nfreq)
    plan_fwd_1d      = CUDA.CUFFT.plan_rfft(kr_1d)

    # batched kernel FFT (Tier 2)
    plan_fwd_kernel = CUDA.CUFFT.plan_rfft(conv_gpu, 2)

    return BatchedMicroConvMem{T}(Nλ, Natm, B, Npad_eff, L, pad_left, pad_right,
                                   zero(T), false, false,
                                   ys_gpu, signal_gpu, signal_ft_gpu, plan_fwd,
                                   kernel_ft_gpu, conv_ft_gpu, conv_gpu, plan_bwd,
                                   xs_gpu, Vector{T}(undef, 0),
                                   kr_1d, kernel_row_ft_1d, plan_fwd_1d, false,
                                   plan_fwd_kernel)
end
