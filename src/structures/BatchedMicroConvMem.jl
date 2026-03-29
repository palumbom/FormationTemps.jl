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

    return BatchedMicroConvMem{T}(Nλ, Natm, B, Npad_eff, L, pad_left, pad_right,
                                   zero(T), false, false,
                                   ys_gpu, signal_gpu, signal_ft_gpu, plan_fwd,
                                   kernel_ft_gpu, conv_ft_gpu, conv_gpu, plan_bwd)
end
