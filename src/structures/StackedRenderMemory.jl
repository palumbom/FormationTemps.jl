"""
    StackedRenderMemory{T<:AbstractFloat}

Pre-allocated GPU buffers and cuFFT plans for rendering the emergent intensity of `B`
atmospheres ("components") on one wavelength grid in a single set of kernel launches.

Layout: the first dimension of every stacked buffer is `B*Natm`; component `b` occupies rows
`(b-1)*Natm+1 : b*Natm` ([`component_rows`](@ref)).

Fields the caller touches:
- `v_los`, `v_mic`: per-layer line-of-sight velocity and microturbulent speed (m/s), length
  `B*Natm`; filled by the caller before each [`render_stacked!`](@ref).
- `intensity`: emergent intensity, shape `(Nλ, B)`; filled by [`render_stacked!`](@ref).
- `rows_independent`: whether a stacked component is bit-identical to its serial render.
  cuFFT transforms the rows of an odd-length real plan in consecutive pairs, so this holds
  only for an even `Natm` or an even padded length. [`render_stacked!`](@ref) asserts it.

Every array field is allocated once by the constructor and never reassigned, so views of
`v_los`, `v_mic` and `intensity` stay valid for the life of the struct.

See also: [`StackedRenderMemory(mem, cmem, atms, μ)`](@ref), [`set_signal_ft!`](@ref)
"""
mutable struct StackedRenderMemory{T<:AF, V<:CA{T,1}, M<:CA{T,2}, C<:CA{Complex{T},2},
                                   PF<:AbstractFFTs.Plan{T},
                                   PB<:AbstractFFTs.Plan{Complex{T}}}
    # ── geometry ──
    const B::Int                    # number of components
    const Natm::Int                 # layers per component
    const Nλ::Int
    const μ::T                      # the Bézier geometry below is built for this μ
    const rows_independent::Bool    # iseven(Natm) || iseven(L)
    const use_anchored::Bool        # τ integrator of the GPUMemory this was built from

    # ── shared with the serial render ──
    const cmem::ConvolutionMemory{T}    # source of σ_floor, xs_gpu, L and pad_left at render time
    const λs::V                         # wavelength grid (Nλ); the array of the GPUMemory

    # ── per-layer inputs (B*Natm) ──
    const v_los::V
    const v_mic::V
    const Ts::V                     # temperatures of the stacked atmospheres
    const tau_ds::V                 # Bézier path lengths; the last entry of a component is unused
    const tau_alphaC::V             # Bézier control-point weights

    # ── Doppler convolution (B*Natm rows) ──
    const signal_ft::C              # (B*Natm, nfreq) padded-signal FFT of every component
    const signal_set::BitVector     # components whose signal FFT has been set
    const kernel_ft::C              # (B*Natm, nfreq) FFT of the per-row Doppler kernels
    const conv_ft::C                # (B*Natm, nfreq) convolution product
    # (B*Natm, L) per-row kernels, then the inverse FFT result. _maybe_report_underflow!
    # reads this field and n_kernel_builds by name.
    const conv_gpu::M
    const plan_fwd::PF              # R2C on (B*Natm, L)
    const plan_bwd::PB              # C2R on (B*Natm, L)

    # ── radiative transfer ──
    const τs::M                     # (B*Natm, Nλ)
    const cfunc::M                  # (B*Natm, Nλ); the last row of a component is unused
    const intensity::M              # (Nλ, B)

    n_kernel_builds::Int            # stacked kernel builds so far; paces the underflow readback
end

"""
    StackedRenderMemory(mem, cmem, atms, μ)

Allocate the buffers for a stacked render of the atmospheres `atms` at disk position `μ`.
`cmem` is the `ConvolutionMemory` the serial render of these components uses and `mem` the
`GPUMemory` holding their wavelength grid; every atmosphere must have `cmem.Natm` layers.

The temperatures and the Bézier geometry of every component are read from `atms` here, once;
later changes to `zs_gpu` or `Ts_gpu` are not seen. Nothing is rendered, and `cmem` need not
have been used yet: its `σ_floor` is read at render time.
"""
function StackedRenderMemory(mem::GPUMemory{T}, cmem::ConvolutionMemory{T},
                             atms::AbstractVector{<:AtmosphereGPU{T}}, μ::T) where T<:AF
    B = length(atms)
    Natm = cmem.Natm
    Nλ = cmem.Nλ
    L = cmem.L
    @assert B >= 1 "a stacked render needs at least one atmosphere"
    @assert Natm >= 3 "the Bézier τ integration needs at least three layers"
    layers_match = all(atm -> length(atm.zs_gpu) == Natm && length(atm.Ts_gpu) == Natm, atms)
    @assert layers_match "every atmosphere must have the $Natm layers of cmem"
    @assert length(mem.λs) == Nλ "mem and cmem must share one wavelength grid"
    nfreq = fld(L, 2) + 1
    BNatm = B * Natm

    v_los = CUDA.zeros(T, BNatm)
    v_mic = CUDA.zeros(T, BNatm)

    # temperatures and Bézier geometry depend on the atmospheres and μ alone
    Ts = CUDA.zeros(T, BNatm)
    tau_ds = CUDA.zeros(T, BNatm)
    tau_alphaC = CUDA.zeros(T, BNatm)
    t_geom = 128
    b_geom = cld(Natm, t_geom)
    for (b, atm) in enumerate(atms)
        rows = ((b - 1) * Natm + 1):(b * Natm)
        copyto!(Ts, first(rows), atm.Ts_gpu, 1, Natm)
        @cuda threads=t_geom blocks=b_geom precompute_bezier_geometry!(
            μ, atm.zs_gpu, view(tau_ds, rows), view(tau_alphaC, rows))
    end

    signal_ft = CuMatrix{Complex{T}}(undef, BNatm, nfreq)
    kernel_ft = CuMatrix{Complex{T}}(undef, BNatm, nfreq)
    conv_ft = CuMatrix{Complex{T}}(undef, BNatm, nfreq)
    conv_gpu = CuMatrix{T}(undef, BNatm, L)
    plan_fwd = CUDA.CUFFT.plan_rfft(conv_gpu, 2)
    plan_bwd = CUDA.CUFFT.plan_irfft(conv_ft, L, 2)

    τs = CUDA.zeros(T, BNatm, Nλ)
    cfunc = CUDA.zeros(T, BNatm, Nλ)
    intensity = CUDA.zeros(T, Nλ, B)

    return StackedRenderMemory{T, typeof(v_los), typeof(τs), typeof(signal_ft),
                               typeof(plan_fwd), typeof(plan_bwd)}(
        B, Natm, Nλ, μ, iseven(Natm) || iseven(L), mem.use_anchored,
        cmem, mem.λs,
        v_los, v_mic, Ts, tau_ds, tau_alphaC,
        signal_ft, falses(B), kernel_ft, conv_ft, conv_gpu, plan_fwd, plan_bwd,
        τs, cfunc, intensity, 0)
end
