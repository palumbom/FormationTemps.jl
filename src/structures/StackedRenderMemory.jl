"""
    StackedRenderMemory{T<:AbstractFloat}

Pre-allocated GPU buffers and cuFFT plans for rendering the emergent intensity of `B`
atmospheres ("components") in a single set of kernel launches. Every component has its own
wavelength grid; the grids share one length `Nλ` and one padded length `L`.

Layout: the first dimension of every stacked buffer is `B*Natm`; component `b` occupies rows
`(b-1)*Natm+1 : b*Natm` ([`component_rows`](@ref)). The stacked wavelength grids are vectors of
length `B*Nλ`; the grid of component `b` occupies elements `(b-1)*Nλ+1 : b*Nλ`.

Fields the caller touches:
- `v_los`, `v_mic`: per-layer line-of-sight velocity and microturbulent speed (m/s), length
  `B*Natm`; filled by the caller before each [`render_stacked!`](@ref).
- `intensity`: emergent intensity, shape `(Nλ, B)`; filled by [`render_stacked!`](@ref). It is
  the caller's matrix when one was passed to the constructor.
- `rows_independent`: whether a stacked component is bit-identical to its serial render.
  cuFFT transforms the rows of an odd-length real plan in consecutive pairs, so this holds
  only for an even `Natm` or an even padded length. [`render_stacked!`](@ref) asserts it.

Every array field is set once by the constructor and never reassigned, so views of `v_los`,
`v_mic` and `intensity` stay valid for the life of the struct.

See also: [`StackedRenderMemory(mems, cmems, atms, μ)`](@ref), [`set_signal_ft!`](@ref)
"""
mutable struct StackedRenderMemory{T<:AF, V<:CA{T,1}, M<:CA{T,2}, MI<:CA{T,2},
                                   C<:CA{Complex{T},2}, PF<:AbstractFFTs.Plan{T},
                                   PB<:AbstractFFTs.Plan{Complex{T}}}
    # ── geometry ──
    const B::Int                    # number of components
    const Natm::Int                 # layers per component
    const Nλ::Int                   # wavelengths per component
    const L::Int                    # padded length of the Doppler convolution
    const pad_left::Int             # columns of conv_gpu before the Nλ valid ones
    const μ::T                      # the Bézier geometry below is built for this μ
    const rows_independent::Bool    # iseven(Natm) || iseven(L)
    const use_anchored::Bool        # whether any GPUMemory this was built from integrates τ anchored

    # ── per-component wavelength grids ──
    # cmems[b] is the ConvolutionMemory of the serial render of component b. It is read once,
    # on the first render, to fill xs and σ_floor.
    const cmems::Vector{ConvolutionMemory{T}}
    const λs::V                     # (B*Nλ) grids of the contribution function
    const xs::V                     # (B*Nλ) grids of the Doppler kernels; valid once grids_latched
    const σ_floor::V                # (B) Gaussian width floor of each grid; valid once grids_latched
    grids_latched::Bool             # xs and σ_floor hold the values of cmems

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
    const intensity::MI             # (Nλ, B); allocated here or supplied by the caller

    n_kernel_builds::Int            # stacked kernel builds so far; paces the underflow readback
end

# the matrix a stacked render writes its intensities to: a new one, or the caller's
_stacked_intensity(::Nothing, ::Type{T}, Nλ::Int, B::Int) where T<:AF = CUDA.zeros(T, Nλ, B)

function _stacked_intensity(intensity::CA{T,2}, ::Type{T}, Nλ::Int, B::Int) where T<:AF
    @assert size(intensity) == (Nλ, B) "intensity must be (Nλ, B) = ($Nλ, $B)"
    return intensity
end

"""
    StackedRenderMemory(mems, cmems, atms, μ; intensity=nothing)
    StackedRenderMemory(mem, cmem, atms, μ; intensity=nothing)

Allocate the buffers for a stacked render of the atmospheres `atms` at disk position `μ`.
Component `b` is rendered on the wavelength grid of `mems[b]` and `cmems[b]`: the `GPUMemory`
and the `ConvolutionMemory` its serial render uses. The three vectors have one entry per
component, and several components may share a memory. Every `cmem` must have the same `Nλ`,
`Natm`, `L` and `pad_left`, every `mem` a grid of `Nλ` wavelengths and every atmosphere `Natm`
layers. The second form puts every component on the one grid of `mem` and `cmem`.

`intensity` is the matrix [`render_stacked!`](@ref) writes to: `nothing` allocates one; a
`CuArray{T,2}` of size `(Nλ, B)` is used as it is, and may be a contiguous column range of a
larger matrix taken with `view`.

The wavelength grid of every `mem` and the temperatures and Bézier geometry of every
atmosphere are copied here, once; later changes to `mem.λs`, `zs_gpu` or `Ts_gpu` are not
seen. Nothing is rendered, and a `cmem` need not have been used yet: its Doppler-kernel grid
and `σ_floor` are read on the first render.
"""
function StackedRenderMemory(mems::AbstractVector{<:GPUMemory{T}},
                             cmems::AbstractVector{<:ConvolutionMemory{T}},
                             atms::AbstractVector{<:AtmosphereGPU{T}}, μ::T;
                             intensity=nothing) where T<:AF
    B = length(atms)
    @assert B >= 1 "a stacked render needs at least one atmosphere"
    @assert length(mems) == B && length(cmems) == B "mems, cmems and atms need one entry per component"
    Natm = first(cmems).Natm
    Nλ = first(cmems).Nλ
    L = first(cmems).L
    pad_left = first(cmems).pad_left
    @assert Natm >= 3 "the Bézier τ integration needs at least three layers"
    shapes_match = all(c -> c.Nλ == Nλ && c.Natm == Natm && c.L == L && c.pad_left == pad_left,
                       cmems)
    @assert shapes_match "every cmem must have the same Nλ, Natm, L and pad_left"
    @assert all(mem -> length(mem.λs) == Nλ, mems) "every mem must hold a grid of the $Nλ wavelengths of the cmems"
    layers_match = all(atm -> length(atm.zs_gpu) == Natm && length(atm.Ts_gpu) == Natm, atms)
    @assert layers_match "every atmosphere must have the $Natm layers of the cmems"
    out = _stacked_intensity(intensity, T, Nλ, B)
    nfreq = fld(L, 2) + 1
    BNatm = B * Natm

    # the grid of the contribution function is fixed by each mem; the Doppler-kernel grid and
    # its σ_floor exist in a cmem only after its first use
    λs = CuVector{T}(undef, B * Nλ)
    for (b, mem) in enumerate(mems)
        copyto!(λs, (b - 1) * Nλ + 1, mem.λs, 1, Nλ)
    end
    xs = CUDA.zeros(T, B * Nλ)
    σ_floor = CUDA.zeros(T, B)

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

    return StackedRenderMemory{T, typeof(v_los), typeof(τs), typeof(out), typeof(signal_ft),
                               typeof(plan_fwd), typeof(plan_bwd)}(
        B, Natm, Nλ, L, pad_left, μ, iseven(Natm) || iseven(L),
        any(mem -> mem.use_anchored, mems),
        collect(ConvolutionMemory{T}, cmems), λs, xs, σ_floor, false,
        v_los, v_mic, Ts, tau_ds, tau_alphaC,
        signal_ft, falses(B), kernel_ft, conv_ft, conv_gpu, plan_fwd, plan_bwd,
        τs, cfunc, out, 0)
end

StackedRenderMemory(mem::GPUMemory{T}, cmem::ConvolutionMemory{T},
                    atms::AbstractVector{<:AtmosphereGPU{T}}, μ::T;
                    intensity=nothing) where T<:AF =
    StackedRenderMemory(fill(mem, length(atms)), fill(cmem, length(atms)), atms, μ; intensity)
