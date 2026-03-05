# calc_tau!(μ_i, zs, αs, τs) = calc_tau_gauss_legendre!(μ_i, zs, αs, τs)
calc_tau!(μ_i, zs, αs, τs) = calc_tau_bezier!(μ_i, zs, αs, τs)
# calc_tau_cpu!(μ_i, zs, αs, τs) = Korg.RadiativeTransfer.compute_tau_bezier!(τs, zs ./ μ_i, αs)

function calc_tau_cpu!(μ_i, zs, αs, τs) 
    for i in axes(τs,2)
        Korg.RadiativeTransfer.compute_tau_bezier!(view(τs,:,i), zs ./ μ_i, view(αs,:,i))
    end
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
                                        ds::CDV{T}, alphaC::CDV{T}) where T<:AF
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    N = size(αs, 1)
    one_third = inv(T(3))
    half = T(0.5)
    zeroT = zero(T)
    oneT = one(T)

    @inbounds for j in idx:sdx:size(αs,2)
        αmax = αs[1, j]
        for p in 2:N
            αmax = max(αmax, αs[p, j])
        end
        lo = zeroT
        hi = max(T(2) * αmax, zeroT)
        τs[1, j] = T(1e-5)

        ds0 = ds[1]
        ds1 = ds[2]
        αC = alphaC[2]
        α1 = αs[1, j]
        α2 = αs[2, j]
        α3 = αs[3, j]
        prev_dC = (α2 - α1) / ds0
        dC = (α3 - α2) / ds1

        ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                      (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
        C0 = α2 + half * ds0 * ybar
        C1 = α2 - half * ds1 * ybar
        Cf = min(max(C0, lo), hi)
        τs[2, j] = τs[1, j] - ds0 * one_third * (α1 + α2 + Cf)

        prev_dC = dC
        prev_C1 = C1

        for c in 3:(N - 1)
            ds0 = ds[c - 1]
            ds1 = ds[c]
            αC = alphaC[c]
            α_prev = αs[c-1, j]
            α_t = αs[c, j]
            α_next = αs[c+1, j]
            dC = (α_next - α_t) / ds1

            ybar = ifelse(prev_dC * dC <= zeroT, zeroT,
                          (prev_dC * dC) / (αC * dC + (oneT - αC) * prev_dC))
            C0 = α_t + half * ds0 * ybar
            C1 = α_t - half * ds1 * ybar
            Cf = min(max(half * (C0 + prev_C1), lo), hi)
            τs[c, j] = τs[c-1, j] - ds0 * one_third * (α_prev + α_t + Cf)

            prev_dC = dC
            prev_C1 = C1
        end

        Cf = min(max(prev_C1, lo), hi)
        ds_last = ds[N - 1]
        τs[N, j] = τs[N-1, j] - ds_last * one_third * (αs[N-1, j] + αs[N, j] + Cf)
    end
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

    t_geom = 128
    b_geom = cld(N, t_geom)
    @cuda threads=t_geom blocks=b_geom precompute_bezier_geometry!(μ_i, zs, ds, alphaC)
    @cuda threads=threads blocks=blocks calc_tau_bezier_cached_kernel!(αs, τs, ds, alphaC)
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

function calc_tau_trapezoid!(μ_i, zs, αs, τs)
    # get indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # length and precompute constants
    N = length(zs)
    inv_μ = 1.0 / μ_i
    one_third = 1.0 / 3.0

    # loop over wavelength
    @inbounds for j in idx:sdx:size(αs,2)
        τs[1,j] = 1e-5
        @inbounds for p in 2:N
            ds = inv_μ * (zs[p-1,j] - zs[p,j])
            τs[p,j] = τs[p-1,j] + 0.5 * (αs[p-1,j] + αs[p,j]) * ds
        end
    end 
    return nothing
end

function calc_tau_trap_cpu!(μ_i::T, zs::AA{T,1}, αs::AA{T,2}, τs::AA{T,2}) where T<:AF
    N = length(zs)
    inv_μ = one(T) / μ_i
    @inbounds for j in 1:size(αs, 2)
        τs[1, j] = T(1e-5)
        for p in 2:N
            ds = inv_μ * (zs[p-1] - zs[p])
            τs[p, j] = τs[p-1, j] + 0.5 * (αs[p-1, j] + αs[p, j]) * ds
        end
    end
    return nothing
end

function calc_tau_simpson!(μ_i, zs, αs, τs)
    # get indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # length and precompute constants
    N = length(zs)
    inv_μ = 1.0 / μ_i
    one_third = 1.0 / 3.0

    # loop over wavelength
    @inbounds for j in idx:sdx:size(αs,2)
        τs[1,j] = 1e-5
        τs[2,j] = 1e-5 + 0.5 * inv_μ * (αs[1,j]+αs[2,j]) * (zs[1]-zs[2])
        @inbounds for p in 3:2:N
            h = zs[p-2] - zs[p]
            τs[p,j] = τs[p-2,j] + (h/(6.0 * μ_i))*(αs[p-2,j] + 4.0 * αs[p-1,j] + αs[p,j])
            τs[p-1,j] = τs[p-2,j] + (0.5 * inv_μ * (zs[p-2] - zs[p-1])) * (αs[p-2,j] + αs[p-1,j])
        end

        # final trapezoid step
        if iseven(N)
            @inbounds τs[N,j] = τs[N-1,j] + 0.5 * (αs[N-1,j] + αs[N,j]) * ((zs[N-1] - zs[N]) * inv_μ)
        end
    end
    return nothing
end

function calc_tau_gauss_legendre!(μ_i, zs, αs, τs)
    # get indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # length and precompute constants
    N = length(zs)
    inv_μ = 1.0 / μ_i

    # standard 3-point nodes & weights on [-1,1]
    ξ = sqrt(3.0/5.0)
    w1 = 5.0/9.0
    w2 = 8.0/9.0
    w3 = 5.0/9.0

    # loop over wavelength
    @inbounds for j in idx:sdx:size(αs,2)
        # initialize
        τs[1,j] = 1e-5

        # loop over atmosphere layers
        @inbounds for p in 2:N
            # endpoints of this slab
            z0 = zs[p]
            z1 = zs[p-1]
            h = z1 - z0
            m = 0.5 * (z0 + z1)

            # real-space GL nodes
            zgl1 = m - 0.5 * ξ * h
            zgl2 = m
            zgl3 = m + 0.5 * ξ * h

            # linear interpolation slope
            α0 = αs[p-1,j]
            α1 = αs[p,j]
            slope = (α1 - α0) / h

            # α at the three nodes
            αg1 = α0 + slope * (zgl1 - z0)
            αg2 = α0 + slope * (zgl2 - z0)
            αg3 = α0 + slope * (zgl3 - z0)

            # 6th-order increment
            slab = (0.5 * h * inv_μ) * (w1 * αg1 + w2 * αg2 + w3 * αg3)
            τs[p,j] = τs[p-1,j] + slab
        end
    end
    return nothing
end
