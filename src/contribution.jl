E_2(τ) = exponential_integral_2_gpu(τ)

"""
    calc_intensity_quantities(αs_init, atm, mem, cmem, μ_tile, v_los, v_mic)

Compute the specific intensity contribution function and its optical-depth-weighted
differential for a single disk tile.

Applies microturbulent broadening (with per-layer Doppler shift `v_los`), integrates the
optical depth using either the anchored or Bézier scheme (as configured in `mem`), and
evaluates the Planck-weighted contribution function at each layer boundary.

Arguments:
- `αs_init::AbstractMatrix{<:Real}`: Absorption coefficients, shape `(Natm, Nλ)`.
- `atm::AtmosphereGPU`: GPU atmosphere.
- `mem::GPUMemory`: Pre-allocated GPU working arrays.
- `cmem::AbstractConvolutionMemory`: Pre-allocated GPU convolution memory.
- `μ_tile::Real`: Cosine of the local zenith angle for this disk tile.
- `v_los::CuArray{<:Real,1}`: Per-layer line-of-sight velocity from rotation (m/s).
- `v_mic::Union{T, CuArray{T,1}}`: Microturbulent broadening width (m/s). Scalar for
  uniform broadening; CuVector for per-layer broadening.

Returns:
- `IntensityContFunc` with fields:
  - `cfunc`: Intensity contribution function `C_I`, shape `(Natm-1, Nλ)`.
  - `cfunc_dt`: `cfunc .* Δτ`, the differential contribution.

See also: [`calc_flux_quantities`](@ref)
"""
function calc_intensity_quantities(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory,
                                   cmem::AbstractConvolutionMemory, μ_tile::T, v_los::CA{T,1},
                                   v_mic::Union{T, CA{T,1}}) where T<:AF
    # get contribution function
    calc_intensity_cfunc!(αs_init, atm, mem, cmem, μ_tile, v_los, v_mic)

    # multiply by differential for cum. cont. & intensity
    # returns independent copies so callers can hold the result across multiple calls
    compute_cfunc_dt!(mem.cfunc_dt, mem.cfunc, mem.τs)
    return IntensityContFunc(copy(mem.cfunc), copy(mem.cfunc_dt))
end

# Zero-allocation variant for the disk integration hot loop.
# Returned cfunc and cfunc_dt alias mem.cfunc / mem.cfunc_dt — caller must consume
# (copy or pass to a function that copies internally) before the next call that shares mem.
function calc_intensity_quantities_inplace!(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory,
                                            cmem::AbstractConvolutionMemory, μ_tile::T, v_los::CA{T,1},
                                            v_mic::Union{T, CA{T,1}}) where T<:AF
    # micro-broadening + tau (same as calc_intensity_cfunc!)
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, v_los, v_mic)
    if mem.use_anchored
        calc_tau_anchored_gpu!(μ_tile, mem.log_τ_ref, mem.ifactor_base, αs_gpu, mem.τs)
    else
        calc_tau_bezier_cached!(μ_tile, atm.zs_gpu, αs_gpu, mem.τs,
                                mem.tau_ds, mem.tau_alphaC)
    end

    # fused cfunc + cfunc_dt in a single kernel launch
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_intensity_cfunc_dt!(μ_tile, atm.Ts_gpu, mem.λs, mem.τs,
                                                        mem.cfunc, mem.cfunc_dt)
    return IntensityContFunc(mem.cfunc, mem.cfunc_dt)
end

"""
    calc_flux_quantities(αs_init, atm, mem, cmem, v_mic)

Compute the disk-center (μ=1, zero rotation) flux contribution function and its
optical-depth-weighted differential.

Applies microturbulent broadening (zero Doppler shift), integrates the optical depth
using either the anchored or Bézier scheme (as configured in `mem`), and evaluates the
Planck-weighted E₂(τ) contribution function at each layer boundary.

Arguments:
- `αs_init::AbstractMatrix{<:Real}`: Absorption coefficients, shape `(Natm, Nλ)`.
- `atm::AtmosphereGPU`: GPU atmosphere.
- `mem::GPUMemory`: Pre-allocated GPU working arrays.
- `cmem::AbstractConvolutionMemory`: Pre-allocated GPU convolution memory.
- `v_mic::Union{T, CuArray{T,1}}`: Microturbulent broadening width (m/s). Scalar for
  uniform broadening; CuVector for per-layer broadening.

Returns:
- `FluxContFunc` with fields:
  - `cfunc`: Flux contribution function `C_F`, shape `(Natm-1, Nλ)`.
  - `cfunc_dt`: `cfunc .* Δτ`, the differential contribution.

See also: [`calc_intensity_quantities`](@ref)
"""
function calc_flux_quantities(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory,
                              cmem::AbstractConvolutionMemory, v_mic::Union{T, CA{T,1}}) where T<:AF
    # micro-broadening + tau (stationary frame: zero v_los)
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, mem.v_los_zeros, v_mic)
    if mem.use_anchored
        calc_tau_anchored_gpu!(one(T), mem.log_τ_ref, mem.ifactor_base, αs_gpu, mem.τs)
    else
        calc_tau_bezier_cached!(one(T), atm.zs_gpu, αs_gpu, mem.τs,
                                mem.tau_ds, mem.tau_alphaC)
    end

    # fused cfunc + cfunc_dt in a single kernel launch
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_flux_cfunc_dt!(atm.Ts_gpu, mem.λs, mem.τs,
                                                    mem.cfunc, mem.cfunc_dt)

    # copy since this is called once (not in hot loop) and the caller
    # may hold the result across a second call that overwrites buffers
    return FluxContFunc(copy(mem.cfunc), copy(mem.cfunc_dt))
end

function calc_intensity_cfunc!(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory,
                               cmem::AbstractConvolutionMemory, μ_tile::T, v_los::CA{T,1},
                               v_mic::Union{T, CA{T,1}}) where T<:AF
    # copy opacities (skip when signal FFT is cached — αs unchanged)
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, v_los, v_mic)

    # compute taus
    if mem.use_anchored
        calc_tau_anchored_gpu!(μ_tile, mem.log_τ_ref, mem.ifactor_base, αs_gpu, mem.τs)
    else
        calc_tau_bezier_cached!(μ_tile, atm.zs_gpu, αs_gpu, mem.τs,
                                mem.tau_ds, mem.tau_alphaC)
    end

    # compute the contribution function
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_intensity_cfunc!(μ_tile, atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    return nothing
end

# Like calc_intensity_cfunc! but writes intensity directly (fused cfunc+reduce).
# Does NOT populate mem.cfunc. Bézier path also skips mem.τs.
# Use calc_intensity_cfunc! if you need the cfunc or τs matrices.
function calc_intensity_direct!(out::CA{T,1}, αs_init::AA{T,2}, atm::AtmosphereGPU{T},
                                mem::GPUMemory, cmem::AbstractConvolutionMemory, μ_tile::T,
                                v_los::CA{T,1}, v_mic::Union{T, CA{T,1}}) where T<:AF
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, v_los, v_mic)
    if mem.use_anchored
        calc_tau_anchored_gpu!(μ_tile, mem.log_τ_ref, mem.ifactor_base, αs_gpu, mem.τs)
        cfunc_reduce_intensity!(out, μ_tile, atm.Ts_gpu, mem.λs, mem.τs)
    else
        calc_tau_cfunc_reduce_bezier!(out, μ_tile, atm.zs_gpu, αs_gpu,
                                      mem.tau_ds, mem.tau_alphaC,
                                      atm.Ts_gpu, mem.λs)
    end
    return nothing
end

function calc_flux_cfunc!(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory,
                         cmem::AbstractConvolutionMemory, v_mic::Union{T, CA{T,1}}) where T<:AF
    # micro-broadening (stationary frame: zero v_los)
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, mem.v_los_zeros, v_mic)

    # compute taus
    if mem.use_anchored
        calc_tau_anchored_gpu!(one(T), mem.log_τ_ref, mem.ifactor_base, αs_gpu, mem.τs)
    else
        calc_tau_bezier_cached!(one(T), atm.zs_gpu, αs_gpu, mem.τs,
                                mem.tau_ds, mem.tau_alphaC)
    end

    # compute the contribution function
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_flux_cfunc!(atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    return nothing
end

function compute_cfunc_dt_kernel!(out::CDM{T}, cfunc::CDM{T}, τs::CDM{T}) where T<:AF
    j = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    k = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    if j <= size(out, 2) && k <= size(out, 1)
        @inbounds out[k, j] = cfunc[k, j] * (τs[k + 1, j] - τs[k, j])
    end
    return nothing
end

function compute_cfunc_dt!(out::CA{T,2}, cfunc::CA{T,2}, τs::CA{T,2}) where T<:AF
    Natm1, Nλ = size(cfunc)
    ts = (32, 16)
    bs = (cld(Nλ, ts[1]), cld(Natm1, ts[2]))
    @cuda threads=ts blocks=bs compute_cfunc_dt_kernel!(out, cfunc, τs)
    return nothing
end

function calc_intensity_cfunc!(μ_i::T, Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM) where T<:AF
    # thread indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    # Gauss-Legendre two-point abscissa constant
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    # loop over lambda
    for j in idx:sdx:length(λs)
        # convert to cm
        λ_cm = λs[j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        # loop over atmosphere
        for k in idy:sdy:length(Ts)-1
            # endpoints in τ-space
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            # Gauss nodes
            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            # linear T interp at fixed 2-point Gauss nodes
            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            # evaluate integrand f = B(T,λ) * exp(-τ)
            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * exp(-τp1)
            f2 = B2 * exp(-τp2)

            # store contribution
            @inbounds cfunc[k, j] = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
        end
    end
    return nothing
end

# fused cfunc + cfunc_dt: computes both in a single pass, avoiding a second kernel launch
# and an extra global memory round-trip for the cfunc matrix
function calc_intensity_cfunc_dt!(μ_i::T, Ts::CDV, λs::CDV, τs::CDM,
                                  cfunc::CDM, cfunc_dt::CDM) where T<:AF
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    for j in idx:sdx:length(λs)
        λ_cm = λs[j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        for k in idy:sdy:length(Ts)-1
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = T(0.5) * (τ0 + τ1)

            τp1 = τ_mid - T(0.5) * Δτ * one_over_sqrt3
            τp2 = τ_mid + T(0.5) * Δτ * one_over_sqrt3

            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * exp(-τp1)
            f2 = B2 * exp(-τp2)

            cf = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
            @inbounds cfunc[k, j] = cf
            @inbounds cfunc_dt[k, j] = cf * Δτ
        end
    end
    return nothing
end

# Fused cfunc + atmosphere reduction: computes intensity directly without
# materializing the cfunc matrix.  2D thread block (x=wavelength, y=atmosphere
# chunk) with shared-memory reduction across y.
function cfunc_reduce_intensity_kernel!(out::CDV{T}, Ts::CDV, λs::CDV,
                                        τs::CDM, Natm1::Int32,
                                        Nλ::Int32) where T<:AF
    shmem = CuDynamicSharedArray(T, Int(blockDim().x) * Int(blockDim().y))

    tx = threadIdx().x
    ty = threadIdx().y
    bdx = Int32(blockDim().x)
    bdy = Int32(blockDim().y)
    j = tx + bdx * (blockIdx().x - Int32(1))

    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    partial = zero(T)
    if j <= Nλ
        λ_cm = λs[j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        k = ty
        while k <= Natm1
            @inbounds begin
                τ0 = τs[k, j]
                τ1 = τs[k + Int32(1), j]
                Δτ = τ1 - τ0
                τ_mid = T(0.5) * (τ0 + τ1)

                τp1 = τ_mid - T(0.5) * Δτ * one_over_sqrt3
                τp2 = τ_mid + T(0.5) * Δτ * one_over_sqrt3

                dT = Ts[k + Int32(1)] - Ts[k]
                T1 = Ts[k] + dT * frac1
                T2 = Ts[k] + dT * frac2

                B1 = bb_num / (exp(bb_x / T1) - one(T))
                B2 = bb_num / (exp(bb_x / T2) - one(T))
                f1 = B1 * exp(-τp1)
                f2 = B2 * exp(-τp2)

                cfunc_val = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
                partial = muladd(cfunc_val, Δτ, partial)
            end
            k += bdy
        end
    end

    # shared memory reduction across atmosphere (y) dimension
    sidx = Int(tx) + (Int(ty) - Int32(1)) * Int(bdx)
    @inbounds shmem[sidx] = partial
    sync_threads()

    s = Int(bdy) >> 1
    while s >= 1
        if Int(ty) <= s
            @inbounds shmem[sidx] += shmem[sidx + s * Int(bdx)]
        end
        sync_threads()
        s >>= 1
    end

    if ty == Int32(1) && j <= Nλ
        @inbounds out[j] = shmem[Int(tx)]
    end
    return nothing
end

# Unfused reduction: out[j] = sum_k cfunc[k,j] * Δτ[k,j].
# Use when cfunc matrix is already computed (e.g., for diagnostics).
function reduce_intensity_kernel!(out::CDV{T}, cfunc::CDM{T}, τs::CDM{T},
                                  Natm1::Int32, Nλ::Int32) where T<:AF
    j = Int32(threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x)
    s = Int32(blockDim().x * gridDim().x)
    while j <= Nλ
        acc = zero(T)
        @inbounds for k in Int32(1):Natm1
            Δτ = τs[k + Int32(1), j] - τs[k, j]
            acc = muladd(cfunc[k, j], Δτ, acc)
        end
        @inbounds out[j] = acc
        j += s
    end
    return nothing
end

function reduce_intensity!(out::CA{T,1}, cfunc::CA{T,2}, τs::CA{T,2}) where T<:AF
    Nλ = Int32(size(cfunc, 2))
    Natm1 = Int32(size(cfunc, 1))
    size(out, 1) == Nλ || error("reduce_intensity!: output length mismatch")
    size(τs, 1) == Natm1 + 1 || error("reduce_intensity!: τs/cfunc atmosphere dimension mismatch")
    size(τs, 2) == Nλ || error("reduce_intensity!: τs/cfunc wavelength dimension mismatch")

    threads = 256
    blocks = cld(Int(Nλ), threads)
    @cuda threads=threads blocks=blocks reduce_intensity_kernel!(out, cfunc, τs, Natm1, Nλ)
    return nothing
end

function cfunc_reduce_intensity!(out::CA{T,1}, μ_i::T, Ts::CA{T,1},
                                  λs::CA{T,1}, τs::CA{T,2}) where T<:AF
    Nλ = Int32(length(λs))
    Natm1 = Int32(size(τs, 1) - 1)
    length(out) == Nλ || error("cfunc_reduce_intensity!: output length mismatch")

    ts = (32, 16)
    bs = (cld(Int(Nλ), ts[1]), 1)
    shmem_bytes = prod(ts) * sizeof(T)
    @cuda threads=ts blocks=bs shmem=shmem_bytes cfunc_reduce_intensity_kernel!(
        out, Ts, λs, τs, Natm1, Nλ)
    return nothing
end

# ── Fused Bézier τ + cfunc + intensity-reduction ─────────────────────────────

# GL cfunc integrand accumulated in-register. Pure (no mutation).
@inline function _accumulate_cfunc(I_acc::T, τ0::T, τ1::T, T0::T, T1::T,
                                    bb_num::T, bb_x::T, half::T,
                                    one_over_sqrt3::T, frac1::T, frac2::T) where T
    Δτ = τ1 - τ0
    τ_mid = half * (τ0 + τ1)
    τp1 = τ_mid - half * Δτ * one_over_sqrt3
    τp2 = τ_mid + half * Δτ * one_over_sqrt3
    dT = T1 - T0
    Tg1 = T0 + dT * frac1
    Tg2 = T0 + dT * frac2
    B1 = bb_num / (exp(bb_x / Tg1) - one(T))
    B2 = bb_num / (exp(bb_x / Tg2) - one(T))
    f1 = B1 * exp(-τp1)
    f2 = B2 * exp(-τp2)
    cfunc_k = half * (f1 + f2) * T(ANGSTROM_TO_CM)
    return muladd(cfunc_k, Δτ, I_acc)
end

# Fused kernel: Bézier τ integration + cfunc evaluation + intensity reduction
# in a single pass. One thread per wavelength, sequential z-loop.
# Mirrors calc_tau_bezier_cached_kernel! (tau.jl) with in-register cfunc accumulation.
# Writes only out[j] — no τs or cfunc matrices.
function calc_tau_cfunc_reduce_bezier_kernel!(out::CDV{T}, αs::CDM{T},
                                              ds::CDV{T}, alphaC::CDV{T},
                                              Ts::CDV{T}, λs::CDV{T},
                                              Nλ::Int32) where T<:AF
    j = Int32(threadIdx().x) + Int32(blockDim().x) * (Int32(blockIdx().x) - Int32(1))
    j > Nλ && return nothing

    N = Int32(size(αs, 1))
    one_third = inv(T(3))
    half = T(0.5)
    zeroT = zero(T)
    oneT = one(T)

    # Planck and GL constants (loop-invariant, hoisted)
    λ_cm = λs[j] * T(ANGSTROM_TO_CM)
    λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
    bb_num = T(2) * T(h) * T(c)^2 / λ5
    bb_x = T(h) * T(c) / (λ_cm * T(kB))
    one_over_sqrt3 = oneT / sqrt(T(3))
    frac1 = half * (oneT - one_over_sqrt3)
    frac2 = half * (oneT + one_over_sqrt3)

    # αmax scan — Bézier overshoot clamping
    @inbounds α_prev = αs[1, j]
    αmax = α_prev
    @inbounds for p in Int32(2):N
        v = αs[p, j]
        αmax = max(αmax, v)
    end
    hi = max(T(2) * αmax, zeroT)

    # init
    τ_prev = T(TAU_FLOOR)
    I_acc = zeroT
    @inbounds T_prev = Ts[1]

    # first iteration (c=2): load α1, α2, α3 fresh
    @inbounds ds_prev = ds[1]
    @inbounds ds_curr = ds[2]
    @inbounds α_curr = αs[2, j]
    @inbounds α_next = αs[3, j]
    prev_dC = (α_curr - α_prev) / ds_prev
    dC = (α_next - α_curr) / ds_curr

    @inbounds αC = alphaC[2]
    ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                  (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
    C0 = α_curr + half * ds_prev * ybar
    C1 = α_curr - half * ds_curr * ybar
    Cf = min(max(C0, zeroT), hi)
    τ_curr = τ_prev - ds_prev * one_third * (α_prev + α_curr + Cf)

    # accumulate cfunc for interval [1, 2]
    @inbounds T_curr = Ts[2]
    I_acc = _accumulate_cfunc(I_acc, τ_prev, τ_curr, T_prev, T_curr,
                              bb_num, bb_x, half, one_over_sqrt3, frac1, frac2)

    prev_dC = dC
    prev_C1 = C1
    α_prev = α_curr
    α_curr = α_next
    ds_prev = ds_curr
    τ_prev = τ_curr
    T_prev = T_curr

    # main loop (lyr=3..N-1). Named 'lyr' to avoid shadowing speed-of-light constant 'c'.
    @inbounds for lyr in Int32(3):(N - Int32(1))
        ds_curr = ds[lyr]
        αC = alphaC[lyr]
        α_next = αs[lyr + Int32(1), j]
        dC = (α_next - α_curr) / ds_curr

        ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                      (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
        C0 = α_curr + half * ds_prev * ybar
        C1 = α_curr - half * ds_curr * ybar
        Cf = min(max(half * (C0 + prev_C1), zeroT), hi)
        τ_curr = τ_prev - ds_prev * one_third * (α_prev + α_curr + Cf)

        # accumulate cfunc for interval [lyr-1, lyr]
        T_curr = Ts[lyr]
        I_acc = _accumulate_cfunc(I_acc, τ_prev, τ_curr, T_prev, T_curr,
                                  bb_num, bb_x, half, one_over_sqrt3, frac1, frac2)

        prev_dC = dC
        prev_C1 = C1
        α_prev = α_curr
        α_curr = α_next
        ds_prev = ds_curr
        τ_prev = τ_curr
        T_prev = T_curr
    end

    # last step (lyr=N)
    Cf = min(max(prev_C1, zeroT), hi)
    τ_curr = τ_prev - ds_prev * one_third * (α_prev + α_curr + Cf)
    @inbounds T_curr = Ts[N]
    I_acc = _accumulate_cfunc(I_acc, τ_prev, τ_curr, T_prev, T_curr,
                              bb_num, bb_x, half, one_over_sqrt3, frac1, frac2)

    @inbounds out[j] = I_acc
    return nothing
end

function calc_tau_cfunc_reduce_bezier!(out::CA{T,1}, μ_i::T, zs::CA{T,1},
                                       αs::CA{T,2}, ds::CA{T,1}, alphaC::CA{T,1},
                                       Ts::CA{T,1}, λs::CA{T,1}) where T<:AF
    N = length(zs)
    Nλ = Int32(size(αs, 2))
    t_geom = 128
    b_geom = cld(N, t_geom)
    @cuda threads=t_geom blocks=b_geom precompute_bezier_geometry!(μ_i, zs, ds, alphaC)
    threads = 32
    blocks = cld(Int(Nλ), threads)
    @cuda threads=threads blocks=blocks calc_tau_cfunc_reduce_bezier_kernel!(
        out, αs, ds, alphaC, Ts, λs, Nλ)
    return nothing
end

function calc_flux_cfunc!(Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM)
    # thread indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    # Gauss-Legendre two-point abscissa constant
    T = eltype(Ts)
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    # loop over lambda
    for j in idx:sdx:length(λs)
        λ_cm = λs[j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        # loop over atmosphere
        for k in idy:sdy:length(Ts)-1
            # endpoints in τ-space
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            # Gauss nodes
            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            # linear T interp at fixed 2-point Gauss nodes
            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            # evaluate integrand f = B(T,λ) * E_2(τ)
            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * E_2(τp1)
            f2 = B2 * E_2(τp2)

            # store contribution
            @inbounds cfunc[k, j] = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
        end
    end
    return nothing
end

# fused flux cfunc + cfunc_dt
function calc_flux_cfunc_dt!(Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM, cfunc_dt::CDM)
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    T = eltype(Ts)
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    for j in idx:sdx:length(λs)
        λ_cm = λs[j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        for k in idy:sdy:length(Ts)-1
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = T(0.5) * (τ0 + τ1)

            τp1 = τ_mid - T(0.5) * Δτ * one_over_sqrt3
            τp2 = τ_mid + T(0.5) * Δτ * one_over_sqrt3

            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * E_2(τp1)
            f2 = B2 * E_2(τp2)

            cf = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
            @inbounds cfunc[k, j] = cf
            @inbounds cfunc_dt[k, j] = cf * Δτ
        end
    end
    return nothing
end

function calc_intensity_cfunc_cpu!(cfunc::AA{T,2}, Ts::AA{T,1}, λs::AA{T,1},
                                   τs::AA{T,2}) where {T<:AF}
    Natm = length(Ts)
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)
    @inbounds for j in 1:length(λs)
        λ_cm = λs[j] * T(ANGSTROM_TO_CM)
        for k in 1:Natm-1
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            f1 = Korg.blackbody(T1, λ_cm) * exp(-τp1)
            f2 = Korg.blackbody(T2, λ_cm) * exp(-τp2)
            @inbounds cfunc[k, j] = 0.5 * (f1 + f2) * T(ANGSTROM_TO_CM)
        end
    end
    return nothing
end

function calc_flux_cfunc_cpu!(cfunc::AA{T,2}, Ts::AA{T,1}, λs::AA{T,1},
                              τs::AA{T,2}) where {T<:AF}
    Natm = length(Ts)
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)
    E2 = exponential_integral_2_gpu
    @inbounds for j in 1:length(λs)
        λ_cm = λs[j] * T(ANGSTROM_TO_CM)
        for k in 1:Natm-1
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            f1 = Korg.blackbody(T1, λ_cm) * E2(τp1)
            f2 = Korg.blackbody(T2, λ_cm) * E2(τp2)
            @inbounds cfunc[k, j] = 0.5 * (f1 + f2) * T(ANGSTROM_TO_CM)
        end
    end
    return nothing
end

# ── batched kernels ────────────────────────────────────────────────────────────

# fused τ-integration + cfunc_dt: one thread per (tile, wavelength), serial over layers.
# computes τ in registers (rolling pair) and immediately evaluates the contribution function.
# eliminates the τs_batch global memory round-trip.
function calc_tau_cfunc_dt_fused_kernel!(cfunc_dt, αs, log_τ_ref, ifactor_base,
                                         μ_tiles, μ_off, Ts, λs,
                                         Natm, Nλ, Bcur, total)
    idx = threadIdx().x + blockDim().x * (blockIdx().x - Int32(1))
    sdx = gridDim().x * blockDim().x
    T = eltype(cfunc_dt)
    Natm1 = Natm - Int32(1)

    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    for lin in idx:sdx:total
        b = ((lin - Int32(1)) ÷ Nλ) + Int32(1)
        j = ((lin - Int32(1)) % Nλ) + Int32(1)
        inv_μ = one(T) / @inbounds μ_tiles[μ_off + b]
        off_α  = (b - Int32(1)) * Natm
        off_cf = (b - Int32(1)) * Natm1

        λ_cm = @inbounds λs[j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        # rolling τ integration + cfunc_dt in one pass
        τ_prev = zero(T)
        @inbounds for k in Int32(1):Natm1
            f_prev = αs[off_α + k, j]     * ifactor_base[k]     * inv_μ
            f_curr = αs[off_α + k + Int32(1), j] * ifactor_base[k + Int32(1)] * inv_μ
            dlog   = log_τ_ref[k + Int32(1)] - log_τ_ref[k]
            τ_curr = τ_prev + T(0.5) * (f_prev + f_curr) * dlog

            Δτ = τ_curr - τ_prev
            τ_mid = T(0.5) * (τ_prev + τ_curr)

            τp1 = τ_mid - T(0.5) * Δτ * one_over_sqrt3
            τp2 = τ_mid + T(0.5) * Δτ * one_over_sqrt3

            dT = Ts[k + Int32(1)] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * exp(-τp1)
            f2 = B2 * exp(-τp2)

            cf = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
            cfunc_dt[off_cf + k, j] = cf * Δτ

            τ_prev = τ_curr
        end
    end
    return nothing
end

function calc_tau_cfunc_dt_fused!(cfunc_dt::CA{T,2}, αs::AA{T,2},
                                   log_τ_ref::CA{T,1}, ifactor_base::CA{T,1},
                                   μ_tiles::CA{T,1}, Ts::CA{T,1}, λs::CA{T,1},
                                   Natm::Int, Bcur::Int; tile_offset::Int=0) where T<:AF
    Nλ = size(cfunc_dt, 2)
    total = Bcur * Nλ
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks calc_tau_cfunc_dt_fused_kernel!(
        cfunc_dt, αs, log_τ_ref, ifactor_base, μ_tiles, Int32(tile_offset),
        Ts, λs, Int32(Natm), Int32(Nλ), Int32(Bcur), Int32(total))
    return nothing
end

# unfused batched cfunc_dt (kept for Bézier path which has a different τ integrator)
function calc_intensity_cfunc_dt_batched_kernel!(cfunc_dt, τs, Ts, λs,
                                                  Natm, Nλ, Bcur, total)
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    T = eltype(cfunc_dt)
    Natm1 = Natm - 1

    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    for lin in idx:sdx:total
        b = ((lin - 1) ÷ Nλ) + 1
        j = ((lin - 1) % Nλ) + 1
        off_τ  = (b - 1) * Natm   # row offset in τs
        off_cf = (b - 1) * Natm1  # row offset in cfunc_dt

        λ_cm = @inbounds λs[j] * T(ANGSTROM_TO_CM)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * T(c)^2 / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        @inbounds for k in 1:Natm1
            τ0 = τs[off_τ + k, j]
            τ1 = τs[off_τ + k + 1, j]
            Δτ = τ1 - τ0
            τ_mid = T(0.5) * (τ0 + τ1)

            τp1 = τ_mid - T(0.5) * Δτ * one_over_sqrt3
            τp2 = τ_mid + T(0.5) * Δτ * one_over_sqrt3

            dT = Ts[k + 1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * exp(-τp1)
            f2 = B2 * exp(-τp2)

            cf = T(0.5) * (f1 + f2) * T(ANGSTROM_TO_CM)
            cfunc_dt[off_cf + k, j] = cf * Δτ
        end
    end
    return nothing
end

function calc_intensity_cfunc_dt_batched!(cfunc_dt::CA{T,2}, τs::CA{T,2},
                                           Ts::CA{T,1}, λs::CA{T,1},
                                           Natm::Int, Bcur::Int) where T<:AF
    Nλ = size(cfunc_dt, 2)
    total = Bcur * Nλ  # Int product — safe from Int32 overflow
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks calc_intensity_cfunc_dt_batched_kernel!(
        cfunc_dt, τs, Ts, λs, Int32(Natm), Int32(Nλ), Int32(Bcur), Int32(total))
    return nothing
end

# 2D batched accumulation: one thread per (layer, wavelength), loops over B tiles only.
# Kahan compensated summation keeps O(ε) accuracy across ~10^4 tile additions.
# flux is NOT accumulated here — derive it post-loop via sum(cfunc_acc, dims=1).
function accumulate_batch_2d_kernel!(cfunc_acc, cfunc_comp,
                                     cfunc_dt, dA_tiles, dA_off, Natm1, Nλ, Bcur)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    (j > Nλ || k > Natm1) && return nothing
    T = eltype(cfunc_acc)
    @inbounds for b in Int32(1):Bcur
        val = cfunc_dt[(b - Int32(1)) * Natm1 + k, j] * dA_tiles[dA_off + b]
        y = val - cfunc_comp[k, j]
        t = cfunc_acc[k, j] + y
        cfunc_comp[k, j] = (t - cfunc_acc[k, j]) - y
        cfunc_acc[k, j] = t
    end
    return nothing
end

function accumulate_batch!(cfunc_acc::CA{T,2}, cfunc_comp::CA{T,2},
                           cfunc_dt::CA{T,2}, dA_tiles::CA{T,1},
                           Natm1::Int, Bcur::Int; tile_offset::Int=0) where T<:AF
    Nλ = Int32(size(cfunc_acc, 2))
    ts = (32, 16)
    bs = (cld(Nλ, ts[1]), cld(Natm1, ts[2]))
    @cuda threads=ts blocks=bs accumulate_batch_2d_kernel!(
        cfunc_acc, cfunc_comp,
        cfunc_dt, dA_tiles, Int32(tile_offset), Int32(Natm1), Nλ, Int32(Bcur))
    return nothing
end

# legacy 1D accumulation (kept for macro path compatibility)
function accumulate_batch_kernel!(flux_acc, cfunc_acc, flux_comp, cfunc_comp,
                                  cfunc_dt, dA_tiles, dA_off, Natm1, Nλ, Bcur)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > Nλ && return nothing
    T = eltype(flux_acc)
    @inbounds for b in 1:Bcur
        dA_i = dA_tiles[dA_off + b]
        off = (b - 1) * Natm1
        s = zero(T)
        s_comp = zero(T)
        for k in 1:Natm1
            val = cfunc_dt[off + k, j] * dA_i

            # Kahan step for cfunc_acc
            y = val - cfunc_comp[k, j]
            t = cfunc_acc[k, j] + y
            cfunc_comp[k, j] = (t - cfunc_acc[k, j]) - y
            cfunc_acc[k, j] = t

            # Kahan step for local flux sum
            y_s = val - s_comp
            t_s = s + y_s
            s_comp = (t_s - s) - y_s
            s = t_s
        end
        # Kahan step for flux_acc
        y_f = s - flux_comp[j]
        t_f = flux_acc[j] + y_f
        flux_comp[j] = (t_f - flux_acc[j]) - y_f
        flux_acc[j] = t_f
    end
    return nothing
end

function accumulate_batch!(flux_acc::CA{T,1}, cfunc_acc::CA{T,2},
                           flux_comp::CA{T,1}, cfunc_comp::CA{T,2},
                           cfunc_dt::CA{T,2}, dA_tiles::CA{T,1},
                           Natm1::Int, Bcur::Int; tile_offset::Int=0) where T<:AF
    Nλ = Int32(size(cfunc_acc, 2))
    threads = 256
    blocks = cld(Nλ, threads)
    @cuda threads=threads blocks=blocks accumulate_batch_kernel!(
        flux_acc, cfunc_acc, flux_comp, cfunc_comp,
        cfunc_dt, dA_tiles, Int32(tile_offset), Int32(Natm1), Nλ, Int32(Bcur))
    return nothing
end
