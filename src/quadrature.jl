# Ring-by-ring μ-quadrature disk integration.
#
# A CPU supplement to the explicit tile-based disk integration
# (`_calc_formation_temp_cpu`, method=:disk). Instead of looping ~10^4 surface
# tiles, it evaluates the disk integral as a Gauss–Legendre quadrature in μ: the
# expensive radiative transfer (which depends on the tile only through μ) is solved
# once per μ node, and rotation enters as a per-ring azimuthal Doppler convolution.
# Supports inclination and differential rotation. See
# `disk_integration_quadrature_notes.md`. The explicit tiling remains the reference.

"""
    _ring_doppler_kernel(μ_k, vsini, iₛ, α₂, α₄, λs, N_az)

Build the normalized line-of-sight-velocity distribution for a ring at projected
radius `r_k = √(1-μ_k²)`, as a length-`Nλ` kernel centered at zero velocity
(index `i0 = Nλ÷2+1`) on the wavelength grid `λs`. Azimuth is sampled uniformly; each
sample's LOS velocity `v = -vsini·f(ϕ)·x_sky` (with `f = diff_rot_factor(sinϕ, α₂, α₄)`)
is deposited into its **nearest** velocity bin. This yields the area-exact
bin-integrated kernel — the correct discrete representation, well-behaved at the
singular ±v_max edges. (A linear-interpolation deposit would converge instead to that
kernel convolved with a spurious ~Δv triangle, over-broadening narrow kernels.) The
azimuth is oversampled so each velocity pixel across the kernel support receives many
samples (`N_az` is a floor); the extra cost is trivial (one 1-D sweep per μ node).
Convolving a ring's spectrum with this kernel performs the azimuthal average of
Doppler-shifted spectra.

NOTE (accuracy floor at small vsini): the kernel is represented on the wavelength
pixel grid, so a Doppler profile only a few pixels wide (small vsini) is resolved only
to ~pixel accuracy. This leaves a localized worst-pixel formation-temperature
difference vs the explicit tiling of ~1–2 K at vsini ~2 km/s (mean <0.1 K); it is a
genuine sub-pixel discretization difference, not a bug, and shrinks with resolution.
Driving it below ~1 K would require sub-pixel Doppler (FFT phase-shift per azimuth
sample), which costs ~N_az× more FFTs and erodes the quadrature's speed advantage —
deliberately not done. See `disk_integration_quadrature_notes.md`.
"""
function _ring_doppler_kernel(μ_k::T, vsini::T, iₛ::T, α₂::T, α₄::T,
                              λs::AA{T,1}, N_az::Int) where T<:AF
    Nλ = length(λs)
    i0 = Nλ ÷ 2 + 1
    λ0 = λs[i0]
    Δv = c_ms * (λs[2] - λs[1]) / λ0        # velocity-grid spacing (m/s), uniform
    K = zeros(T, Nλ)
    r_k = sqrt(max(one(T) - μ_k^2, zero(T)))
    cosiₛ = cos(iₛ)
    siniₛ = sin(iₛ)

    # oversample azimuth so every velocity pixel across the ~2·vsini·r_k span (f ≤ 1)
    # gets many nearest-bin hits (≈32/pixel); floor at N_az.
    span_px = 2 * vsini * r_k / Δv
    N_fine = max(N_az, ceil(Int, 32 * span_px))
    w = one(T) / N_fine
    @inbounds for j in 0:(N_fine - 1)
        az = T(2π) * j / N_fine
        x_sky = r_k * cos(az)
        y_sky = r_k * sin(az)
        sinϕ = y_sky * cosiₛ + μ_k * siniₛ          # stellar latitude of the point
        f = diff_rot_factor(sinϕ, α₂, α₄)
        v = -vsini * f * x_sky                       # LOS velocity (m/s)
        pn = round(Int, i0 + v / Δv)                 # nearest velocity bin
        (1 <= pn <= Nλ) && (K[pn] += w)
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
                                        Nμ=16, N_az=256, kwargs...)

Ring-by-ring μ-quadrature evaluation of the disk-integrated flux formation
temperatures (CPU). Produces the same `FormTempResult` as the explicit tiling path
(`method=:disk`); intended as a fast, validated supplement. `Nμ` sets the number of
Gauss–Legendre μ nodes; `N_az` the azimuthal sampling of the per-ring Doppler kernel.
"""
function _calc_formation_temp_quadrature_cpu(star::StellarProps, linelist; Δλ::T=0.01,
                                             minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                             Nμ::Int=16, N_az::Int=256,
                                             showprogress::Bool=true,
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
    ws = CPUTileWorkspace(T, Natm, Nλ)
    v0 = copy(atm_cpu.v_los)                 # zero-rotation base velocity (length Natm)
    iₛ = deg2rad(T(90) - star.istar)

    cfunc_dt_flux = zeros(T, Natm - 1, Nλ)
    cfunc_dt_flux_cont = zeros(T, Natm - 1, Nλ)
    G_k = zeros(T, Natm - 1, Nλ)
    G_k_cont = zeros(T, Natm - 1, Nλ)

    for k in eachindex(μ_grid)
        μ_k = T(μ_grid[k])
        wq = T(μ_weights[k]) * μ_k            # projected-area weight (∫ μ dμ)

        # G_k(z,λ; μ_k), zero Doppler: micro(v_mic) → τ(μ_k) → intensity → cfunc_dt → macro(μ_k)
        _convolve_micro_inplace!(ws.αs_broad, λs_korg, αs, v0, atm_cpu.v_mic, ws)
        _calc_tau_cpu!(μ_k, ws.αs_broad, ws.τs_int)
        calc_intensity_cfunc_cpu!(ws.cfunc_int, Ts, λs_korg, ws.τs_int)
        @views ws.cfunc_dt_int .= ws.cfunc_int .* (ws.τs_int[2:end, :] .- ws.τs_int[1:end-1, :])
        _convolve_macro_inplace!(G_k, λs_korg, ws.cfunc_dt_int, star.ζ, μ_k, ws)

        # continuum
        _convolve_micro_inplace!(ws.αs_cont_broad, λs_korg, αs_cont, v0, atm_cpu.v_mic, ws)
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
            cfunc_dt_flux .+= wq .* _padded_convolve(G_k, K)
            cfunc_dt_flux_cont .+= wq .* _padded_convolve(G_k_cont, K)
        end
    end

    # --- reduction (identical to the tiling path) ---
    cum_cfunc_flux = cumsum(cfunc_dt_flux, dims=1)
    cum_cfunc_flux ./= maximum(cum_cfunc_flux, dims=1)

    flux_norm = vec(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))

    form_temps = zeros(T, Nλ)
    mid_temps = elav(Ts)
    for i in eachindex(λs_korg)
        xs = view(cum_cfunc_flux, :, i)
        itp = linear_interp(xs, mid_temps)
        form_temps[i] = itp(0.5)
    end

    cont_func = cfunc_dt_flux
    return FormTempResult(collect(λs_korg), flux_norm, form_temps, cont_func, atm_cpu)
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
                                        gpu_precision=Float64, Nμ=16, N_az=256, kwargs...)

GPU port of the ring-by-ring μ-quadrature disk integration. Loops the `Nμ`
Gauss–Legendre μ nodes using the single-tile GPU intensity path and the GPU RT-macro
convolution, applying rotation as a per-ring Doppler convolution. Produces the same
`FormTempResult` as `_calc_formation_temp_quadrature_cpu`.
"""
function _calc_formation_temp_quadrature_gpu(star::StellarProps, linelist; Δλ::T=0.01,
                                             gpu_precision::Type{<:AF}=Float64,
                                             minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                             Nμ::Int=16, N_az::Int=256,
                                             showprogress::Bool=true,
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
    Npad = 512

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

    for k in eachindex(μ_grid)
        μ_k = G(μ_grid[k])
        wq = G(μ_weights[k]) * μ_k                  # projected-area weight (∫ μ dμ)

        # per-μ depth-resolved intensity cfunc_dt at zero Doppler (micro→τ→intensity);
        # calc_intensity_quantities returns independent copies.
        G_k  = calc_intensity_quantities(αs,      atm_gpu, gpu_mem,      cmem,      μ_k, gpu_mem.v_los_zeros,      atm_gpu.v_mic).cfunc_dt
        G_kc = calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem_cont, cmem_cont, μ_k, gpu_mem_cont.v_los_zeros, atm_gpu.v_mic).cfunc_dt

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
    cum_cfunc_flux = Array(cumsum(cfunc_dt_flux, dims=1))
    cum_cfunc_flux ./= maximum(cum_cfunc_flux, dims=1)

    flux_norm = G.(vec(Array(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))))

    form_temps = zeros(G, length(λs_korg))
    mid_temps = elav(atm_gpu.Ts)
    for i in eachindex(λs_korg)
        xs = view(cum_cfunc_flux, :, i)
        itp = linear_interp(xs, mid_temps)
        form_temps[i] = G(itp(G(0.5)))
    end

    cont_func = Array(cfunc_dt_flux)
    return FormTempResult(G.(collect(λs_korg)), flux_norm, form_temps, cont_func, atm_gpu)
end
