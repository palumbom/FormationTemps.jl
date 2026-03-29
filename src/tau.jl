# calc_tau!(μ_i, zs, αs, τs) = calc_tau_gauss_legendre!(μ_i, zs, αs, τs)
# calc_tau!(μ_i, zs, αs, τs) = calc_tau_bezier!(μ_i, zs, αs, τs)
# calc_tau_cpu!(μ_i, zs, αs, τs) = Korg.RadiativeTransfer.compute_tau_bezier!(τs, zs ./ μ_i, αs)

function calc_tau_bezier_cpu!(μ_i, zs, αs, τs)
    for i in axes(τs,2)
        Korg.RadiativeTransfer.compute_tau_bezier!(view(τs,:,i), zs ./ μ_i, view(αs,:,i))
    end
end

"""
    calc_tau_anchored_cpu!(μ_i, τ_ref, α_ref, αs, τs)

Compute optical depth by integrating in d(log τ_ref) rather than ds, following the
"anchored" scheme of Korg (Wheeler et al. 2022, Appendix A.1).

The change of variables τ = ∫ α ds = ∫ (α/α_ref) dτ_ref eliminates the sensitivity
to non-uniform geometric layer spacing in the model atmosphere.

α_ref must be the physical continuum absorption coefficient at the MARCS reference
wavelength for each layer, computed from chemistry (not estimated from layer geometry).
This is what makes the integration independent of Δz.
"""
function calc_tau_anchored_cpu!(μ_i::T, τ_ref::AA{T,1}, α_ref::AA{T,1},
                                αs::AA{T,2}, τs::AA{T,2}) where T<:AF
    N = length(τ_ref)

    # scale factor: τ_ref / α_ref / μ  (same for all wavelengths)
    integrand_factor = τ_ref ./ α_ref ./ μ_i

    log_τ_ref = log.(τ_ref)

    @inbounds for j in axes(τs, 2)
        τs[1, j] = zero(T)
        for i in 2:N
            f_prev = αs[i-1, j] * integrand_factor[i-1]
            f_curr = αs[i,   j] * integrand_factor[i]
            τs[i, j] = τs[i-1, j] + T(0.5) * (f_prev + f_curr) * (log_τ_ref[i] - log_τ_ref[i-1])
        end
    end
    return nothing
end

"""
    calc_tau_anchored_kernel!(αs, τs, log_τ_ref, ifactor_base, inv_μ, Nλ)

GPU kernel for anchored τ integration.  One thread per wavelength; serial loop over N layers.

Integrates τ[i] = τ[i-1] + 0.5*(f[i-1]+f[i]) * Δ(log τ_ref)[i]
where f[k] = αs[k,j] * ifactor_base[k] * inv_μ  and  ifactor_base[k] = τ_ref[k] / α_ref[k].
"""
function calc_tau_anchored_kernel!(αs::CDM{T}, τs::CDM{T},
                                    log_τ_ref::CDV{T}, ifactor_base::CDV{T},
                                    inv_μ::T, Nλ::Int32) where T<:AF
    j = Int32(threadIdx().x) + Int32(blockDim().x) * (Int32(blockIdx().x) - Int32(1))
    j > Nλ && return nothing

    N    = Int32(size(αs, 1))
    half = T(0.5)

    @inbounds τs[1, j] = zero(T)
    @inbounds for i in Int32(2):N
        f_prev = αs[i - Int32(1), j] * ifactor_base[i - Int32(1)] * inv_μ
        f_curr = αs[i,             j] * ifactor_base[i]             * inv_μ
        dlog   = log_τ_ref[i] - log_τ_ref[i - Int32(1)]
        τs[i, j] = τs[i - Int32(1), j] + half * (f_prev + f_curr) * dlog
    end
    return nothing
end

function calc_tau_anchored_gpu!(μ_i::T, log_τ_ref::CA{T,1}, ifactor_base::CA{T,1},
                                 αs::CA{T,2}, τs::CA{T,2};
                                 threads::Int=1024,
                                 blocks::Int=cld(size(αs, 2), threads)) where T<:AF
    Nλ    = Int32(size(αs, 2))
    inv_μ = one(T) / μ_i
    @cuda threads=threads blocks=blocks calc_tau_anchored_kernel!(
        αs, τs, log_τ_ref, ifactor_base, inv_μ, Nλ)
    return nothing
end

function precompute_bezier_geometry!(μ_i::T, zs::CDV{T}, ds::CDV{T},
                                     alphaC::CDV{T}) where T<:AF
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    N = length(zs)
    inv_μ = one(T) / μ_i
    one_third = inv(T(3))

    @inbounds for p in idx:sdx:N
        if p <= (N - 1)
            ds[p] = (zs[p+1] - zs[p]) * inv_μ
        end
        if p >= 2 && p <= (N - 1)
            ds_left = (zs[p] - zs[p-1]) * inv_μ
            ds_right = (zs[p+1] - zs[p]) * inv_μ
            alphaC[p] = one_third * (one(T) + ds_right / (ds_left + ds_right))
        elseif p == 1 || p == N
            alphaC[p] = zero(T)
        end
    end
    return nothing
end

function calc_tau_bezier_cached_kernel!(αs::CDM{T}, τs::CDM{T},
                                        ds::CDV{T}, alphaC::CDV{T},
                                        Nλ::Int32) where T<:AF
    j = Int32(threadIdx().x) + Int32(blockDim().x) * (Int32(blockIdx().x) - Int32(1))
    j > Nλ && return nothing

    N = Int32(size(αs, 1))
    one_third = inv(T(3))
    half = T(0.5)
    zeroT = zero(T)
    oneT = one(T)

    # αmax scan — needed for Bézier overshoot clamping
    @inbounds α_prev = αs[1, j]
    αmax = α_prev
    @inbounds for p in Int32(2):N
        v = αs[p, j]
        αmax = max(αmax, v)
    end
    hi = max(T(2) * αmax, zeroT)
    @inbounds τs[1, j] = T(1e-5)

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
    @inbounds τs[2, j] = τs[1, j] - ds_prev * one_third * (α_prev + α_curr + Cf)

    prev_dC = dC
    prev_C1 = C1

    # carry forward registers: α_prev ← α_curr, α_curr ← α_next, ds_prev ← ds_curr
    α_prev = α_curr
    α_curr = α_next
    ds_prev = ds_curr

    # main loop (c=3..N-1): only 3 new global loads per iteration (α_next, ds_curr, αC)
    @inbounds for c in Int32(3):(N - Int32(1))
        ds_curr = ds[c]
        αC = alphaC[c]
        α_next = αs[c + Int32(1), j]
        dC = (α_next - α_curr) / ds_curr

        ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                      (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
        C0 = α_curr + half * ds_prev * ybar
        C1 = α_curr - half * ds_curr * ybar
        Cf = min(max(half * (C0 + prev_C1), zeroT), hi)
        τs[c, j] = τs[c - Int32(1), j] - ds_prev * one_third * (α_prev + α_curr + Cf)

        prev_dC = dC
        prev_C1 = C1
        α_prev = α_curr
        α_curr = α_next
        ds_prev = ds_curr
    end

    # last step (c=N)
    Cf = min(max(prev_C1, zeroT), hi)
    @inbounds τs[N, j] = τs[N - Int32(1), j] - ds_prev * one_third * (α_prev + α_curr + Cf)
    return nothing
end

function calc_tau_bezier_cached!(μ_i::T, zs::CA{T,1}, αs::CA{T,2}, τs::CA{T,2},
                                 ds::CA{T,1}, alphaC::CA{T,1};
                                 threads::Int=32,
                                 blocks::Int=cld(size(αs,2), threads)) where T<:AF
    N = length(zs)
    N >= 3 || error("calc_tau_bezier_cached! requires Natm >= 3")
    length(ds) == N - 1 || error("ds must have length Natm - 1")
    length(alphaC) == N || error("alphaC must have length Natm")
    size(τs, 1) == N || error("τs atmosphere dimension mismatch")
    size(αs, 1) == N || error("αs atmosphere dimension mismatch")
    size(αs, 2) == size(τs, 2) || error("αs/τs wavelength dimension mismatch")

    Nλ = Int32(size(αs, 2))
    t_geom = 128
    b_geom = cld(N, t_geom)
    @cuda threads=t_geom blocks=b_geom precompute_bezier_geometry!(μ_i, zs, ds, alphaC)
    @cuda threads=threads blocks=blocks calc_tau_bezier_cached_kernel!(αs, τs, ds, alphaC, Nλ)
    return nothing
end

function calc_tau_bezier!(μ_i, zs, αs, τs)
    # get indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # length and precompute constants
    N = length(zs)
    T = eltype(τs)
    inv_μ = one(T) / T(μ_i)
    one_third = inv(T(3))
    half = T(0.5)
    zeroT = zero(T)
    oneT = one(T)

    # loop over wavelengths
    @inbounds for j in idx:sdx:size(αs,2)
        # bounds for clamping — opacity is non-negative
        αmax = αs[1, j]
        for p in 2:N
            v = αs[p, j]
            αmax = max(αmax, v)
        end
        lo = zeroT
        hi = max(T(2) * αmax, zeroT)

        # init
        τs[1, j] = T(1e-5)

        # first iteration handle outside loop
        s_prev = zs[1] * inv_μ
        s_t = zs[2] * inv_μ
        s_next = zs[3] * inv_μ
        ds0 = s_t - s_prev
        ds1 = s_next - s_t
        αC = one_third * (oneT + ds1 / (ds0 + ds1))
        prev_dC = (αs[2, j] - αs[1, j]) / ds0
        dC = (αs[3, j] - αs[2, j]) / ds1

        # monotone limiter: zero derivative at local extrema to prevent denominator
        # blow-up (αC*dC + (1-αC)*prev_dC → 0) and Cf overshoot
        ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                      (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
        α2 = αs[2, j]
        C0 = α2 + half * ds0 * ybar
        C1 = α2 - half * ds1 * ybar
        Cf = min(max(C0, lo), hi)

        # update tau
        τs[2, j] = τs[1, j] + (s_prev - s_t) * one_third * (αs[1, j] + αs[2, j] + Cf)

        # for next iteration
        prev_dC = dC
        prev_C1 = C1

        # loop until final step
        @inbounds for t in 2:N-2
            s_prev = s_t
            s_t = s_next
            s_next = zs[t+2] * inv_μ
            ds0 = s_t - s_prev
            ds1 = s_next - s_t

            αC = one_third * (oneT + ds1 / (ds0 + ds1))
            α_t = αs[t+1, j]
            dC = (αs[t+2, j] - α_t) / ds1

            ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                          (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
            C0 = α_t + half * ds0 * ybar
            C1 = α_t - half * ds1 * ybar
            Cf = min(max(half * (C0 + prev_C1), lo), hi)

            # update tau
            τs[t+1, j] = τs[t, j] + (s_prev - s_t) * one_third * (αs[t, j] + α_t + Cf)

            # for next iteration
            prev_dC = dC
            prev_C1 = C1
        end

        # handle last step outside loop
        s_t = zs[N] * inv_μ
        ds0 = s_prev - s_t
        Cf = min(max(prev_C1, lo), hi)
        @inbounds τs[N, j] = τs[N-1, j] + (one_third * ds0) * (αs[N-1, j] + αs[N, j] + Cf)
    end
    return nothing
end

# batched anchored τ: one thread per (tile, wavelength), serial over layers
function calc_tau_anchored_batched_kernel!(αs, τs, log_τ_ref, ifactor_base,
                                           μ_tiles, Natm, Nλ, Bcur, total)
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    for lin in idx:sdx:total
        b = ((lin - 1) ÷ Nλ) + 1
        j = ((lin - 1) % Nλ) + 1
        T = eltype(τs)
        inv_μ = one(T) / @inbounds μ_tiles[b]
        off = (b - 1) * Natm  # row offset for this tile
        @inbounds τs[off + 1, j] = zero(T)
        @inbounds for i in 2:Natm
            f_prev = αs[off + i - 1, j] * ifactor_base[i - 1] * inv_μ
            f_curr = αs[off + i, j]     * ifactor_base[i]     * inv_μ
            τs[off + i, j] = τs[off + i - 1, j] +
                T(0.5) * (f_prev + f_curr) * (log_τ_ref[i] - log_τ_ref[i - 1])
        end
    end
    return nothing
end

function calc_tau_anchored_batched!(μ_tiles::CA{T,1}, log_τ_ref::CA{T,1},
                                    ifactor_base::CA{T,1}, αs::AA{T,2}, τs::CA{T,2},
                                    Natm::Int, Bcur::Int) where T<:AF
    Nλ = size(αs, 2)
    total = Bcur * Nλ  # Int product — safe from Int32 overflow
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks calc_tau_anchored_batched_kernel!(
        αs, τs, log_τ_ref, ifactor_base, μ_tiles, Int32(Natm), Int32(Nλ), Int32(Bcur),
        Int32(total))
    return nothing
end

# batched Bezier geometry: precompute ds and alphaC for B tiles
function precompute_bezier_geometry_batched!(μ_tiles, zs, ds, alphaC, Natm, Bcur)
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    T = eltype(ds)
    one_third = inv(T(3))
    N = Natm
    for lin in idx:sdx:(Bcur * N)
        b = ((lin - 1) ÷ N) + 1
        p = ((lin - 1) % N) + 1
        inv_μ = one(T) / @inbounds μ_tiles[b]
        off = (b - 1) * N
        @inbounds if p <= (N - 1)
            ds[off + p] = (zs[p + 1] - zs[p]) * inv_μ
        end
        @inbounds if p >= 2 && p <= (N - 1)
            ds_left  = (zs[p] - zs[p - 1]) * inv_μ
            ds_right = (zs[p + 1] - zs[p]) * inv_μ
            alphaC[off + p] = one_third * (one(T) + ds_right / (ds_left + ds_right))
        elseif p == 1 || p == N
            alphaC[off + p] = zero(T)
        end
    end
    return nothing
end

# batched Bezier tau: one thread per (tile, wavelength), serial over layers
function calc_tau_bezier_batched_kernel!(αs, τs, ds, alphaC, Natm, Nλ, Bcur, total)
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    T = eltype(τs)
    N = Natm
    one_third = inv(T(3))
    half = T(0.5)
    zeroT = zero(T)
    oneT = one(T)

    for lin in idx:sdx:total
        b = ((lin - 1) ÷ Nλ) + 1
        j = ((lin - 1) % Nλ) + 1
        off = (b - 1) * N

        # αmax scan for overshoot clamping
        @inbounds α_prev = αs[off + 1, j]
        αmax = α_prev
        @inbounds for p in 2:N
            v = αs[off + p, j]
            αmax = max(αmax, v)
        end
        hi = max(T(2) * αmax, zeroT)
        @inbounds τs[off + 1, j] = T(1e-5)

        # first iteration (c=2)
        @inbounds ds_prev = ds[off + 1]
        @inbounds ds_curr = ds[off + 2]
        @inbounds α_curr = αs[off + 2, j]
        @inbounds α_next = αs[off + 3, j]
        prev_dC = (α_curr - α_prev) / ds_prev
        dC = (α_next - α_curr) / ds_curr

        @inbounds αC = alphaC[off + 2]
        ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                      (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
        C0 = α_curr + half * ds_prev * ybar
        C1 = α_curr - half * ds_curr * ybar
        Cf = min(max(C0, zeroT), hi)
        @inbounds τs[off + 2, j] = τs[off + 1, j] - ds_prev * one_third * (α_prev + α_curr + Cf)

        prev_dC = dC
        prev_C1 = C1
        α_prev = α_curr
        α_curr = α_next
        ds_prev = ds_curr

        # main loop (c=3..N-1)
        for c in 3:(N - 1)
            @inbounds ds_curr = ds[off + c]
            @inbounds αC = alphaC[off + c]
            @inbounds α_next = αs[off + c + 1, j]
            dC = (α_next - α_curr) / ds_curr

            ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                          (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
            C0 = α_curr + half * ds_prev * ybar
            C1 = α_curr - half * ds_curr * ybar
            Cf = min(max(half * (C0 + prev_C1), zeroT), hi)
            @inbounds τs[off + c, j] = τs[off + c - 1, j] - ds_prev * one_third * (α_prev + α_curr + Cf)

            prev_dC = dC
            prev_C1 = C1
            α_prev = α_curr
            α_curr = α_next
            ds_prev = ds_curr
        end

        # last step (c=N)
        Cf = min(max(prev_C1, zeroT), hi)
        @inbounds τs[off + N, j] = τs[off + N - 1, j] - ds_prev * one_third * (α_prev + α_curr + Cf)
    end
    return nothing
end

function calc_tau_bezier_batched!(μ_tiles::CA{T,1}, zs::CA{T,1},
                                  αs::AA{T,2}, τs::CA{T,2},
                                  ds::CA{T,1}, alphaC::CA{T,1},
                                  Natm::Int, Bcur::Int) where T<:AF
    Natm >= 3 || error("calc_tau_bezier_batched! requires Natm >= 3")
    Nλ = size(αs, 2)

    # precompute geometry for B tiles
    t_geom = 256
    b_geom = cld(Bcur * Natm, t_geom)
    @cuda threads=t_geom blocks=b_geom precompute_bezier_geometry_batched!(
        μ_tiles, zs, ds, alphaC, Int32(Natm), Int32(Bcur))

    # main Bezier integration
    total = Bcur * Nλ  # Int product — safe from Int32 overflow
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks calc_tau_bezier_batched_kernel!(
        αs, τs, ds, alphaC, Int32(Natm), Int32(Nλ), Int32(Bcur), Int32(total))
    return nothing
end
