# Stacked render: the emergent intensity of B components in one set of kernel launches. Every
# stage repeats the arithmetic of the serial render (calc_intensity_cfunc! followed by
# reduce_intensity!) operation for operation, over B*Natm rows, so a component's intensity is
# the same bits either way. A change to a serial stage must be mirrored here. Each component
# is rendered on its own wavelength grid: a row's kernel reads the block of the stacked grids
# that belongs to the row's component.

"""
    component_rows(smem, b)

Rows of the stacked buffers of `smem` that hold component `b`: `(b-1)*Natm+1 : b*Natm`.
"""
component_rows(smem::StackedRenderMemory, b::Integer) =
    ((b - 1) * smem.Natm + 1):(b * smem.Natm)

"""
    set_signal_ft!(smem, b, signal_ft)

Copy the padded-signal FFT of component `b` into `smem`. `signal_ft` is `(Natm, nfreq)`: what
`cmem.signal_ft_gpu` holds after a serial render of that component's opacities on its own
`ConvolutionMemory`, the `cmems[b]` that `smem` was built from. Call it again whenever the
component's opacities change.
"""
function set_signal_ft!(smem::StackedRenderMemory{T}, b::Integer,
                        signal_ft::CA{Complex{T},2}) where T<:AF
    @assert 1 <= b <= smem.B "component $b is outside 1:$(smem.B)"
    nfreq = size(smem.signal_ft, 2)
    @assert size(signal_ft) == (smem.Natm, nfreq) "signal_ft must be (Natm, nfreq)"
    view(smem.signal_ft, component_rows(smem, b), :) .= signal_ft
    smem.signal_set[b] = true
    return nothing
end

# Copy the Doppler-kernel grid, the kernel offsets and the Gaussian width floor of every
# component's ConvolutionMemory into smem.xs, smem.dx and smem.σ_floor. A ConvolutionMemory
# holds them only once it has rendered or been initialized, and they are constant from then on,
# so this runs once, on the first stacked render. The floor is read through _sigma_floor_cached,
# as the serial kernel build reads it.
function _latch_grids!(smem::StackedRenderMemory{T}) where T<:AF
    Nλ = smem.Nλ
    σ_floor = Vector{T}(undef, smem.B)
    for (b, cmem) in enumerate(smem.cmems)
        @assert cmem.doppler_ready "the ConvolutionMemory of component $b has not rendered " *
            "yet: its wavelength grid and σ_floor are not set"
        copyto!(smem.xs, (b - 1) * Nλ + 1, cmem.xs_gpu, 1, Nλ)
        copyto!(smem.dx, (b - 1) * Nλ + 1, cmem.dx_gpu, 1, Nλ)
        σ_floor[b] = _sigma_floor_cached(cmem, cmem.xs_cpu)
    end
    copyto!(smem.σ_floor, σ_floor)
    smem.grids_latched = true
    return nothing
end

# kernel_to_dft_layout_2d_gpu! with the wavelength grid, the kernel offsets and the width
# floor of the row's component: row `row` belongs to component b = (row - 1) ÷ Natm + 1, whose
# grid and offsets are elements (b - 1) * Nλ + 1 : b * Nλ of `xs` and `dx` and whose floor is
# σ_floor[b]. The floating-point operations and their order are those of
# kernel_to_dft_layout_2d_gpu!.
function kernel_to_dft_layout_stacked_kernel!(kbuf::CDM{T}, xs::CDV{T}, dx::CDV{T},
                                              v_los::CDV{T}, v_mic::CDV{T}, σ_floor::CDV{T},
                                              i0::Int32, Nλ::Int32, L::Int32, Natm::Int32,
                                              BNatm::Int32) where T<:AF
    j = Int32(threadIdx().x) + Int32(blockDim().x) * (Int32(blockIdx().x) - Int32(1))
    row = Int32(threadIdx().y) + Int32(blockDim().y) * (Int32(blockIdx().y) - Int32(1))
    # guard on BNatm, not size(kbuf, 1): the launch ceil-rounds to the block size
    (row > BNatm || j > Nλ) && return nothing
    b = (row - Int32(1)) ÷ Natm + Int32(1)
    off = (b - Int32(1)) * Nλ

    @inbounds begin
        xj = xs[off + j]
        λ0 = xs[off + i0]
        σx = max(xj * v_mic[row] / T(c_ms), σ_floor[b])
        # dx and (v_los/c)*λ0 are both small; subtracting λc from xj would cancel
        Δx = dx[off + j] - v_los[row] / T(c_ms) * λ0
        val = exp(-(Δx / σx)^2)
        d = j - i0
        idx = d >= Int32(0) ? d + Int32(1) : L + d + Int32(1)
        kbuf[row, idx] = val
    end
    return nothing
end

# calc_intensity_cfunc! with a per-row temperature and the wavelength grid of the row's
# component, elements (b - 1) * Nλ + 1 : b * Nλ of `λs`. The thread at (row, j) evaluates the
# interval between rows `row` and `row + 1`; the last row of a component has no interval below
# it and is skipped, which leaves that row of cfunc unused.
function calc_intensity_cfunc_stacked_kernel!(cfunc::CDM{T}, τs::CDM{T}, Ts::CDV{T},
                                              λs::CDV{T}, Natm::Int32, Nλ::Int32) where T<:AF
    j = Int32(threadIdx().x) + Int32(blockDim().x) * (Int32(blockIdx().x) - Int32(1))
    row = Int32(threadIdx().y) + Int32(blockDim().y) * (Int32(blockIdx().y) - Int32(1))
    (j > Nλ || row > size(cfunc, 1)) && return nothing
    row % Natm == Int32(0) && return nothing
    off = ((row - Int32(1)) ÷ Natm) * Nλ

    # Gauss-Legendre two-point abscissa constant
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    @inbounds begin
        # convert to cm
        λ_cm = λs[off + j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        # endpoints in τ-space
        τ0 = τs[row, j]
        τ1 = τs[row + Int32(1), j]
        Δτ = τ1 - τ0
        τ_mid = T(0.5) * (τ0 + τ1)

        # Gauss nodes
        τp1 = τ_mid - T(0.5) * Δτ * one_over_sqrt3
        τp2 = τ_mid + T(0.5) * Δτ * one_over_sqrt3

        # linear T interp at fixed 2-point Gauss nodes
        dT = Ts[row + Int32(1)] - Ts[row]
        T1 = Ts[row] + dT * frac1
        T2 = Ts[row] + dT * frac2

        # evaluate integrand f = B(T,λ) * exp(-τ)
        B1 = bb_num / (exp(bb_x / T1) - one(T))
        B2 = bb_num / (exp(bb_x / T2) - one(T))
        f1 = B1 * exp(-τp1)
        f2 = B2 * exp(-τp2)

        # store contribution
        cfunc[row, j] = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
    end
    return nothing
end

# reduce_intensity_kernel! with one thread per (component, wavelength): out[j, b] is the sum of
# cfunc * Δτ over the Natm - 1 intervals of component b, accumulated from the top layer down.
function reduce_intensity_stacked_kernel!(out::CDM{T}, cfunc::CDM{T}, τs::CDM{T},
                                          Natm::Int32, Nλ::Int32, total::Int32) where T<:AF
    lin = Int32(threadIdx().x) + Int32(blockDim().x) * (Int32(blockIdx().x) - Int32(1))
    lin > total && return nothing
    b = (lin - Int32(1)) ÷ Nλ + Int32(1)
    j = (lin - Int32(1)) % Nλ + Int32(1)
    off = (b - Int32(1)) * Natm

    acc = zero(T)
    @inbounds for k in Int32(1):(Natm - Int32(1))
        Δτ = τs[off + k + Int32(1), j] - τs[off + k, j]
        acc = muladd(cfunc[off + k, j], Δτ, acc)
    end
    @inbounds out[j, b] = acc
    return nothing
end

"""
    render_stacked!(smem)

Render every component of `smem` at the velocities in `smem.v_los` and `smem.v_mic` (m/s,
length `B*Natm`) and write the emergent intensities to `smem.intensity` (`Nλ × B`).

Column `b` of `smem.intensity` is bit-identical to the serial render of component `b`
(`calc_intensity_cfunc!` with its signal FFT cached, then `reduce_intensity!`) on its own
`GPUMemory` and `ConvolutionMemory`, the `mems[b]` and `cmems[b]` that `smem` was built from.
Every one of those `ConvolutionMemory`s must have rendered at least once before the first
call, which is what sets its wavelength grid, kernel offsets and `σ_floor`; the first call
copies them into `smem`, and no later call reads a `ConvolutionMemory`. Every component's
signal FFT must have been handed over with [`set_signal_ft!`](@ref).

Afterwards the valid columns of `smem.conv_gpu` hold the convolved opacities and `smem.τs` and
`smem.cfunc` the optical depths and contribution functions, in the stacked row layout.

Allocates no device memory on a call after the first whose underflow check is not due.
"""
function render_stacked!(smem::StackedRenderMemory{T}) where T<:AF
    @assert smem.rows_independent "cuFFT transforms the rows of an odd-length real plan in " *
        "consecutive pairs: with an odd Natm at an odd padded length the pairs straddle " *
        "components, and a stacked component is not its serial render"
    @assert !smem.use_anchored "the stacked render integrates τ with the Bézier scheme only"
    @assert all(smem.signal_set) "set_signal_ft! has not been called for every component"
    smem.grids_latched || _latch_grids!(smem)

    Natm = smem.Natm
    Nλ = smem.Nλ
    L = smem.L
    BNatm = smem.B * Natm
    total = smem.B * Nλ

    # per-row Doppler kernels, built and normalized as _build_per_row_kernels! does, each row
    # on the grid and with the σ_floor of its component
    i0 = Nλ ÷ 2 + 1
    fill!(smem.conv_gpu, zero(T))
    ts = (32, 32)
    bs = (cld(Nλ, ts[1]), cld(BNatm, ts[2]))
    @cuda threads=ts blocks=bs kernel_to_dft_layout_stacked_kernel!(
        smem.conv_gpu, smem.xs, smem.dx, smem.v_los, smem.v_mic, smem.σ_floor,
        Int32(i0), Int32(Nλ), Int32(L), Int32(Natm), Int32(BNatm))
    _normalize_rows!(smem.conv_gpu, BNatm)
    _maybe_report_underflow!(smem)

    # convolve every row with its kernel
    mul!(smem.kernel_ft, smem.plan_fwd, smem.conv_gpu)
    smem.conv_ft .= smem.signal_ft .* smem.kernel_ft
    mul!(smem.conv_gpu, smem.plan_bwd, smem.conv_ft)
    αs_conv = @view smem.conv_gpu[:, (smem.pad_left + 1):(smem.pad_left + Nλ)]

    # compute taus: one thread per (component, wavelength), each a serial loop over the layers,
    # in small blocks as calc_tau_bezier_cached! launches them
    t_tau = 32
    @cuda threads=t_tau blocks=cld(total, t_tau) calc_tau_bezier_batched_kernel!(
        αs_conv, smem.τs, smem.tau_ds, smem.tau_alphaC,
        Int32(Natm), Int32(Nλ), Int32(smem.B), Int32(total))

    # compute the contribution function
    ts = (32, 16)
    bs = (cld(Nλ, ts[1]), cld(BNatm, ts[2]))
    @cuda threads=ts blocks=bs calc_intensity_cfunc_stacked_kernel!(
        smem.cfunc, smem.τs, smem.Ts, smem.λs, Int32(Natm), Int32(Nλ))

    # reduce over layers
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks reduce_intensity_stacked_kernel!(
        smem.intensity, smem.cfunc, smem.τs, Int32(Natm), Int32(Nλ), Int32(total))
    return nothing
end
