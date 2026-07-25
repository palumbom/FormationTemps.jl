# Ring-by-ring μ-quadrature disk integration: a CPU/GPU alternative to the explicit
# tile-based disk integration (method=:disk, which remains the reference). The disk integral
# is evaluated as a Gauss-Legendre quadrature in μ, so the radiative transfer — which depends
# on a surface element only through μ — is solved once per μ node rather than once per tile,
# and rotation enters as a per-ring azimuthal Doppler convolution.
#
# Applying rotation after the RT solve is exact rather than an approximation: the transfer is
# wavelength-local (nothing couples wavelengths), so Doppler-shifting the input opacity
# α(λ) → α(λ(1-v/c)) shifts the emergent intensity identically. The azimuthal average over a
# ring is then a convolution of the ring's zero-Doppler spectrum with the ring's LOS velocity
# distribution. The macro and ring-Doppler kernels are both linear, so their order is free.

# Truncated weight below this fraction is roundoff, not a real window problem.
const _RING_TRUNC_WARN = 1e-6

"""
    _ring_kernel_rigid!(K, v_max, Δv, i0, Nλ) -> dropped

Exact bin-integrated ring Doppler kernel for solid-body rotation, written into `K`.
Returns the weight falling outside the grid.

For `α₂ = α₄ = 0` the LOS velocity is `v(az) = -v_max·cos(az)`, `v_max = vsini·r_k`,
independent of inclination. With azimuth uniform on `[0, 2π)`, `u = v/v_max` follows the
arcsine distribution on `[-1, 1]`, with CDF `G(u) = asin(u)/π + 1/2`. Bin `n` spans
`(n - i0 ± 1/2)·Δv`, so its weight is the CDF difference across its edges.

Evaluating the CDF analytically rather than sampling makes the bin weights exact, keeps the
kernel exactly symmetric (`asin` is odd, so no radial-velocity shift is representable), and
costs `O(support)` arcsin evaluations. The integrable `1/√(1-u²)` singularity at `±v_max`
needs no special handling, since the CDF is finite there.
"""
function _ring_kernel_rigid!(K::AA{T,1}, v_max::T, Δv::T, i0::Int, Nλ::Int) where T<:AF
    G(u) = asin(clamp(u, -one(T), one(T))) / T(π) + T(0.5)
    inv_vmax = one(T) / v_max

    # only bins overlapping [-v_max, v_max] carry weight
    half_px = ceil(Int, v_max / Δv) + 1
    lo = max(1, i0 - half_px)
    hi = min(Nλ, i0 + half_px)
    @inbounds for n in lo:hi
        K[n] = G((n - i0 + T(0.5)) * Δv * inv_vmax) - G((n - i0 - T(0.5)) * Δv * inv_vmax)
    end
    return max(zero(T), one(T) - sum(K))     # analytic total is 1; shortfall = truncated
end

"""
    _ring_kernel_diffrot!(K, μ_k, r_k, vsini, iₛ, α₂, α₄, Δv, i0, Nλ, N_az) -> dropped

Bin-integrated ring Doppler kernel for **differential** rotation, written into `K`.
Returns the weight that fell outside the grid.

With `f(ϕ) ≠ 1` the LOS velocity `v(az) = -vsini·f(sinϕ(az))·r_k·cos(az)` (latitude
`sinϕ = r_k·sin(az)·cos iₛ + μ_k·sin iₛ`) is not a pure cosine and its pushforward has no
closed form, so azimuth is sampled. Each arc `[az_k, az_{k+1}]` carries weight `1/N_fine`
and maps to the velocity interval spanned by its endpoints; that weight is distributed
across the bins that interval overlaps.

This is not a linear/CIC deposit, which treats each sample as a point and splits it between
neighbouring bins by linear weights — that converges to the true kernel convolved with a
spurious ~`Δv` triangle and over-broadens narrow kernels. Integrating each arc's actual
velocity extent is exact bin integration, second-order in the arc width rather than first.

`N_fine` is forced even. The `v ↔ -v` symmetry of the exact distribution comes from
`az → π - az`, which flips `x_sky` while leaving the latitude (hence `f`) fixed; on the
discrete arc partition that map closes only for even `N_fine`. An odd count leaves `K`
slightly asymmetric, i.e. a spurious radial-velocity shift of order `Δv/N_fine`.
"""
function _ring_kernel_diffrot!(K::AA{T,1}, μ_k::T, r_k::T, vsini::T, iₛ::T,
                               α₂::T, α₄::T, Δv::T, i0::Int, Nλ::Int,
                               N_az::Int) where T<:AF
    cosiₛ = cos(iₛ)
    siniₛ = sin(iₛ)

    # azimuth → continuous bin coordinate of the LOS velocity
    px(az) = begin
        x_sky = r_k * cos(az)
        sinϕ = r_k * sin(az) * cosiₛ + μ_k * siniₛ
        i0 - vsini * diff_rot_factor(sinϕ, α₂, α₄) * x_sky / Δv
    end

    # ~32 arcs per velocity pixel across the ~2·vsini·r_k span (f ≤ 1), floored at N_az;
    # even, so the az → π-az pairing closes on the arc partition (see docstring)
    span_px = 2 * vsini * r_k / Δv
    N_fine = max(N_az, ceil(Int, 32 * span_px))
    N_fine += isodd(N_fine)

    w = one(T) / N_fine
    dropped = zero(T)
    p_prev = px(zero(T))
    @inbounds for k in 1:N_fine
        # close the loop on the exact same value it started from, so the partition is
        # periodic to the bit and the symmetry is not spoiled by 2π round-off
        p_curr = k == N_fine ? px(zero(T)) : px(T(2π) * k / N_fine)
        plo, phi = minmax(p_prev, p_curr)
        p_prev = p_curr

        nlo = round(Int, plo)
        nhi = round(Int, phi)
        if nlo == nhi
            # arc falls inside one bin; also the dv/daz → 0 turning points at az = 0, π
            (1 <= nlo <= Nλ) ? (K[nlo] += w) : (dropped += w)
        else
            scale = w / (phi - plo)     # nlo != nhi ⇒ phi > plo strictly
            for n in nlo:nhi
                overlap = min(phi, n + T(0.5)) - max(plo, n - T(0.5))
                overlap <= zero(T) && continue
                (1 <= n <= Nλ) ? (K[n] += scale * overlap) : (dropped += scale * overlap)
            end
        end
    end
    return dropped
end

"""
    _ring_doppler_kernel(μ_k, vsini, iₛ, α₂, α₄, λs, N_az)

Build the normalized line-of-sight-velocity distribution for a ring at projected
radius `r_k = √(1-μ_k²)`, as a length-`Nλ` kernel centered at zero velocity
(index `i0 = Nλ÷2+1`) on the wavelength grid `λs`. Convolving a ring's spectrum with this
kernel performs the azimuthal average of Doppler-shifted spectra.

The kernel is the area-exact bin-integrated LOS velocity distribution, computed two ways
depending on the rotation law:

- `α₂ = α₄ = 0` (solid body, the default): analytically via the arcsine CDF,
  [`_ring_kernel_rigid!`](@ref). Exact, symmetric, and inclination-independent.
- otherwise: by arc-overlap deposition over an even number of azimuthal arcs,
  [`_ring_kernel_diffrot!`](@ref). `N_az` floors the arc count and only matters here.

Both are symmetric in velocity, so the kernel carries no spurious radial-velocity shift.
Weight Doppler-shifted outside the wavelength window is discarded and the kernel
renormalized, which narrows the broadening; that is warned about rather than left silent,
since it means the synthesis window is too narrow for `vsini`.

Accuracy is limited by the kernel living on the wavelength pixel grid: the bin weights are
exact, but their positions are quantized to `Δv`, which matters most at low `vsini` where
the kernel spans only a few pixels. See the "Integration Methods" guide for the resulting
tolerances against `method=:disk`.
"""
function _ring_doppler_kernel(μ_k::T, vsini::T, iₛ::T, α₂::T, α₄::T,
                              λs::AA{T,1}, N_az::Int) where T<:AF
    Nλ = length(λs)
    i0 = Nλ ÷ 2 + 1
    λ0 = λs[i0]
    # velocity-grid spacing (m/s), uniform. c_ms is Float64, so narrow back to T: the
    # helpers below take Δv::T, and without this a Float32 grid yields a Float64 Δv and no
    # matching method. Identity when T === Float64.
    Δv = T(c_ms * (λs[2] - λs[1]) / λ0)
    K = zeros(T, Nλ)
    r_k = sqrt(max(one(T) - μ_k^2, zero(T)))
    v_max = vsini * r_k

    # degenerate ring (pole-on node, or no rotation): all weight at zero velocity
    if v_max <= zero(T)
        K[i0] = one(T)
        return K
    end

    # TODO(bessel-transfer-function): the solid-body branch can skip the real-space kernel
    # entirely. The characteristic function of the arcsine distribution is a single Bessel
    # function, so H(f) = J₀(2π·f·v_max/Δv) on the padded FFT frequency grid is the exact
    # transfer function — no pixel quantization, and real-valued so it cannot carry an RV
    # shift. Needs a transfer-function variant of _padded_convolve plus a GPU path that
    # fills kernel_row_ft_1d directly; differential rotation still needs the sampled path.
    dropped = if iszero(α₂) & iszero(α₄)
        _ring_kernel_rigid!(K, v_max, Δv, i0, Nλ)
    else
        _ring_kernel_diffrot!(K, μ_k, r_k, vsini, iₛ, α₂, α₄, Δv, i0, Nλ, N_az)
    end

    # Truncated weight is renormalized away below, silently narrowing the rotational
    # broadening. Reachable when the kernel support exceeds the window: 2·v_max/Δv > Nλ.
    # Surface it so a too-narrow window (or a cm/s vs m/s unit error in vsini) is visible
    # rather than showing up as an unexplained shallow line profile.
    if dropped > T(_RING_TRUNC_WARN)
        @warn "ring Doppler kernel truncated by the wavelength window: $(round(100*dropped, digits=2))% " *
              "of the azimuthal weight fell outside the grid (kernel support " *
              "$(round(2 * v_max / Δv, digits=1)) px vs Nλ = $Nλ). Rotational broadening is " *
              "under-estimated; widen the window (minλ/maxλ) or check the units of vsini (m/s)." maxlog=3
    end

    s = sum(K)
    if s > zero(T)
        K ./= s
    else
        K[i0] = one(T)      # degenerate (all weight shifted out of window)
    end
    return K
end

"""
    _calc_formation_temp_quadrature_cpu(star, linelist; Δλ, minλ, maxλ, buffer,
                                        Nμ=32, N_az=256, kwargs...)

Ring-by-ring μ-quadrature evaluation of the disk-integrated flux formation
temperatures (CPU). Produces the same `FormTempResult` as the explicit tiling path
(`method=:disk`); intended as a fast, validated supplement. `Nμ` sets the number of
Gauss–Legendre μ nodes; `N_az` the azimuthal sampling of the per-ring Doppler kernel.
"""
function _calc_formation_temp_quadrature_cpu(star::StellarProps, linelist; Δλ::T=0.01,
                                             minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                             Nμ::Int=32, N_az::Int=256,
                                             showprogress::Bool=true,
                                             r_thresh::Real=BOUNDARY_R_THRESH,
                                             kwargs...) where T<:AF
    # --- setup (mirrors _calc_formation_temp_cpu) ---
    wls = [l.wl * CM_TO_ANGSTROM for l in linelist]
    minλ = isnan(minλ) ? first(wls) - buffer : minλ
    maxλ = isnan(maxλ) ? last(wls) + buffer : maxλ
    λs_korg = range(minλ, maxλ, step=Δλ)

    atm_cpu = AtmosphereCPU(Korg.interpolate_marcs(star.Teff, star.logg, star.A_X))
    zs = atm_cpu.zs
    Ts = atm_cpu.Ts

    Natm = length(zs)
    Nλ = length(λs_korg)
    αs = zeros(T, Natm, Nλ)
    αs_cont = zeros(T, Natm, Nλ)
    α_ref = zeros(T, Natm)
    compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg),
                   linelist, atm_cpu, star.A_X; α_ref_out=α_ref, kwargs...)

    star.ξ isa AbstractVector && @assert length(star.ξ) == Natm "v_micro vector length ($(length(star.ξ))) must match Natm ($Natm)"
    if star.ξ isa AbstractVector
        copyto!(atm_cpu.v_mic, T.(star.ξ))
    else
        fill!(atm_cpu.v_mic, star.ξ)
    end

    _calc_tau_cpu! = _make_tau_integrator(atm_cpu, zs, α_ref)

    # --- Gauss–Legendre μ nodes on [0,1] (reuse Korg; no new dependency) ---
    μ_grid, μ_weights = Korg.RadiativeTransfer.generate_mu_grid(Nμ)

    # --- ring-by-ring accumulation ---
    # Size the convolution padding from the kernel support: the ring Doppler kernel reaches
    # vsini and the macro kernel ~3ζ, and an under-padded linear convolution wraps silently.
    λ0_pad = λs_korg[Nλ ÷ 2 + 1]
    Npad = conv_npad_for_velocity(λ0_pad, Δλ,
                                  conv_kernel_vmax(star.vsini, star.ζ, star.ξ))

    ws = CPUTileWorkspace(T, Natm, Nλ; Npad=Npad)
    v0 = copy(atm_cpu.v_los)                 # zero-rotation base velocity (length Natm)
    iₛ = deg2rad(T(90) - star.istar)
    i0 = Nλ ÷ 2 + 1

    cfunc_dt_flux = zeros(T, Natm - 1, Nλ)
    cfunc_dt_flux_cont = zeros(T, Natm - 1, Nλ)
    G_k = zeros(T, Natm - 1, Nλ)
    G_k_cont = zeros(T, Natm - 1, Nλ)
    ring_out = zeros(T, Natm - 1, Nλ)        # ring-convolved cfunc_dt, reused per node

    # Microturbulence depends only on v0 and v_mic, both fixed across the μ loop, so it is
    # applied once here. Inside the loop ws.αs_broad / ws.αs_cont_broad are only read (the
    # macro convolution and τ integration use separate buffers), so these stay valid.
    _convolve_micro_inplace!(ws.αs_broad, λs_korg, αs, v0, atm_cpu.v_mic, ws)
    _convolve_micro_inplace!(ws.αs_cont_broad, λs_korg, αs_cont, v0, atm_cpu.v_mic, ws)

    for k in eachindex(μ_grid)
        μ_k = T(μ_grid[k])
        wq = T(μ_weights[k]) * μ_k            # projected-area weight (∫ μ dμ)

        # G_k(z,λ; μ_k), zero Doppler: τ(μ_k) → intensity → cfunc_dt → macro(μ_k)
        _calc_tau_cpu!(μ_k, ws.αs_broad, ws.τs_int)
        calc_intensity_cfunc_cpu!(ws.cfunc_int, Ts, λs_korg, ws.τs_int)
        @views ws.cfunc_dt_int .= ws.cfunc_int .* (ws.τs_int[2:end, :] .- ws.τs_int[1:end-1, :])
        _convolve_macro_inplace!(G_k, λs_korg, ws.cfunc_dt_int, star.ζ, μ_k, ws)

        # continuum
        _calc_tau_cpu!(μ_k, ws.αs_cont_broad, ws.τs_int_cont)
        calc_intensity_cfunc_cpu!(ws.cfunc_int_cont, Ts, λs_korg, ws.τs_int_cont)
        @views ws.cfunc_dt_int_cont .= ws.cfunc_int_cont .* (ws.τs_int_cont[2:end, :] .- ws.τs_int_cont[1:end-1, :])
        _convolve_macro_inplace!(G_k_cont, λs_korg, ws.cfunc_dt_int_cont, star.ζ, μ_k, ws)

        # azimuthal Doppler convolution (identity when vsini==0)
        if iszero(star.vsini)
            cfunc_dt_flux .+= wq .* G_k
            cfunc_dt_flux_cont .+= wq .* G_k_cont
        else
            K = _ring_doppler_kernel(μ_k, star.vsini, iₛ, star.α₂, star.α₄, λs_korg, N_az)
            # same padded linear convolution as _padded_convolve, through the workspace
            # plans and buffers: one kernel FT serves both signals, and no per-row allocation
            _kernel_to_dft_layout!(ws.kernel_real, K, i0)
            mul!(ws.kernel_ft, ws.fft_plan, ws.kernel_real)
            _apply_fft_kernel!(ring_out, G_k, ws.kernel_ft, ws, Natm - 1)
            cfunc_dt_flux .+= wq .* ring_out
            _apply_fft_kernel!(ring_out, G_k_cont, ws.kernel_ft, ws, Natm - 1)
            cfunc_dt_flux_cont .+= wq .* ring_out
        end
    end

    # --- reduction (identical to the tiling path) ---
    flux_norm = vec(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))

    # formation temperature at 50% cumulative flux contribution (node-anchored CDF)
    form_temps = form_temps_from_cfunc(cfunc_dt_flux, Ts; r_thresh=r_thresh)

    cont_func = cfunc_dt_flux
    return FormTempResult(collect(λs_korg), flux_norm, form_temps, cont_func, atm_cpu;
                          r_thresh=r_thresh)
end

# ── GPU ─────────────────────────────────────────────────────────────────────────

"""
    _ring_kernel_ft_gpu(macmem, K, i0)

Build the Fourier transform of an arbitrary (already normalized) real-space ring
Doppler kernel `K` (length `Nλ`, centered at `i0`) in the padded/DFT layout expected by
[`convolve_rt_macro_gpu_cached`](@ref). Places `K` into the valid region of
`macmem.padded_kernel_gpu` and reuses [`_finalize_kernel_ft!`](@ref).
"""
function _ring_kernel_ft_gpu(macmem::MacroConvolutionMemory{T}, K::AA{T,1}, i0::Int) where {T<:AF}
    fill!(macmem.padded_kernel_gpu, zero(T))
    dst = view(macmem.padded_kernel_gpu, macmem.pad_left + 1 : macmem.pad_left + macmem.Nλ)
    copyto!(dst, K)                      # K already normalized (sum = 1)
    return _finalize_kernel_ft!(macmem, i0)
end

"""
    _calc_formation_temp_quadrature_gpu(star, linelist; Δλ, minλ, maxλ, buffer,
                                        gpu_precision=Float64, Nμ=32, N_az=256, kwargs...)

GPU port of the ring-by-ring μ-quadrature disk integration. Loops the `Nμ`
Gauss–Legendre μ nodes using the single-tile GPU intensity path and the GPU RT-macro
convolution, applying rotation as a per-ring Doppler convolution. Produces the same
`FormTempResult` as `_calc_formation_temp_quadrature_cpu`.
"""
function _calc_formation_temp_quadrature_gpu(star::StellarProps, linelist; Δλ::T=0.01,
                                             gpu_precision::Type{<:AF}=Float64,
                                             minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                             Nμ::Int=32, N_az::Int=256,
                                             showprogress::Bool=true,
                                             r_thresh::Real=BOUNDARY_R_THRESH,
                                             kwargs...) where T<:AF
    G = gpu_precision

    # --- setup (mirrors _calc_formation_temp_gpu) ---
    wls = [l.wl * CM_TO_ANGSTROM for l in linelist]
    minλ = isnan(minλ) ? first(wls) - buffer : minλ
    maxλ = isnan(maxλ) ? last(wls) + buffer : maxλ
    λs_korg = range(minλ, maxλ, step=Δλ)

    korg_atm = Korg.interpolate_marcs(star.Teff, star.logg, star.A_X)
    atm_f64 = AtmosphereGPU(korg_atm; T=Float64)
    Natm = length(atm_f64.zs)
    Nλ = length(λs_korg)
    αs = zeros(Float64, Natm, Nλ)
    αs_cont = zeros(Float64, Natm, Nλ)
    α_ref = zeros(Float64, Natm)
    compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg),
                   linelist, atm_f64, star.A_X; α_ref_out=α_ref, kwargs...)

    if G !== Float64
        atm_gpu = AtmosphereGPU(korg_atm; T=G)
        αs = G.(αs); αs_cont = G.(αs_cont); α_ref = G.(α_ref)
    else
        atm_gpu = atm_f64
    end

    λs_G = G.(collect(λs_korg))
    _make_gpu_mem = isempty(atm_gpu.τs) ? (() -> GPUMemory(λs_G, atm_gpu)) :
                                          (() -> GPUMemory(λs_G, atm_gpu, α_ref))
    gpu_mem = _make_gpu_mem()
    gpu_mem_cont = _make_gpu_mem()

    # size the convolution padding from the actual kernel support (ring Doppler kernel
    # reaches vsini, macro kernel ~3ζ) so no padded linear convolution can wrap
    Npad = conv_npad_for_velocity(λs_korg[Nλ ÷ 2 + 1], Δλ,
                                  conv_kernel_vmax(star.vsini, star.ζ, star.ξ))

    star.ξ isa AbstractVector && @assert length(star.ξ) == Natm "v_micro vector length ($(length(star.ξ))) must match Natm ($Natm)"
    if star.ξ isa AbstractVector
        copyto!(atm_gpu.v_mic, G.(star.ξ))
    else
        fill!(atm_gpu.v_mic, G(star.ξ))
    end

    # separate micro-convolution memory for total vs continuum (distinct signals)
    cmem = ConvolutionMemory(Nλ, Natm, Npad; T=G)
    cmem_cont = ConvolutionMemory(Nλ, Natm, Npad; T=G)
    macmem = MacroConvolutionMemory(Nλ, Natm - 1, Npad; T=G)

    # --- μ-quadrature ---
    μ_grid, μ_weights = Korg.RadiativeTransfer.generate_mu_grid(Nμ)
    iₛ = deg2rad(G(90) - G(star.istar))
    i0 = Nλ ÷ 2 + 1
    λs_h = G.(collect(λs_korg))                     # host grid for ring-kernel build
    ζ = G(star.ζ)

    cfunc_dt_flux = CUDA.zeros(G, Natm - 1, Nλ)
    cfunc_dt_flux_cont = CUDA.zeros(G, Natm - 1, Nλ)

    # Microturbulence depends only on v_mic and the (zero) rotation velocity, both fixed
    # across the μ loop, so it is applied once. convolve_wavelength_axis_gpu returns a view
    # into cmem.conv_gpu, which the per-row kernel build reuses as scratch, so copy out.
    copyto!(gpu_mem.αs, αs)
    αs_b = copy(convolve_wavelength_axis_gpu(cmem, gpu_mem.λs, gpu_mem.αs,
                                             gpu_mem.v_los_zeros, atm_gpu.v_mic))
    copyto!(gpu_mem_cont.αs, αs_cont)
    αs_cont_b = copy(convolve_wavelength_axis_gpu(cmem_cont, gpu_mem_cont.λs, gpu_mem_cont.αs,
                                                  gpu_mem_cont.v_los_zeros, atm_gpu.v_mic))

    for k in eachindex(μ_grid)
        μ_k = G(μ_grid[k])
        wq = G(μ_weights[k]) * μ_k                  # projected-area weight (∫ μ dμ)

        # per-μ depth-resolved intensity cfunc_dt at zero Doppler (τ→intensity). The results
        # alias gpu_mem/gpu_mem_cont and are consumed by the macro convolution below.
        G_k  = calc_intensity_quantities_broadened!(αs_b,      atm_gpu, gpu_mem,      μ_k).cfunc_dt
        G_kc = calc_intensity_quantities_broadened!(αs_cont_b, atm_gpu, gpu_mem_cont, μ_k).cfunc_dt

        # μ-dependent macroturbulence; copy out of the shared macmem.out_gpu buffer
        Gm  = copy(convolve_rt_macro_gpu(macmem, λs_G, G_k,  ζ, μ_k))
        Gmc = copy(convolve_rt_macro_gpu(macmem, λs_G, G_kc, ζ, μ_k))

        if iszero(star.vsini)
            cfunc_dt_flux      .+= wq .* Gm
            cfunc_dt_flux_cont .+= wq .* Gmc
        else
            K = _ring_doppler_kernel(μ_k, G(star.vsini), iₛ, G(star.α₂), G(star.α₄), λs_h, N_az)
            kft = _ring_kernel_ft_gpu(macmem, K, i0)
            # cached conv returns macmem.out_gpu; accumulate immediately before reuse
            cfunc_dt_flux      .+= wq .* convolve_rt_macro_gpu_cached(macmem, Gm,  kft)
            cfunc_dt_flux_cont .+= wq .* convolve_rt_macro_gpu_cached(macmem, Gmc, kft)
        end
    end

    # --- reduction (host-side; mirrors _calc_formation_temp_gpu) ---
    flux_norm = G.(vec(Array(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))))

    # formation temperature at 50% cumulative flux contribution (node-anchored CDF);
    # extraction is host-side, so pass host copies
    form_temps = form_temps_from_cfunc(Array(cfunc_dt_flux), Array(atm_gpu.Ts); r_thresh=r_thresh)

    cont_func = Array(cfunc_dt_flux)
    return FormTempResult(G.(collect(λs_korg)), flux_norm, form_temps, cont_func, atm_gpu;
                          r_thresh=r_thresh)
end
