calc_tau!(μ_i, zs, αs, τs) = calc_tau_gauss_legendre!(μ_i, zs, αs, τs)
# calc_tau!(μ_i, zs, αs, τs) = calc_tau_bezier!(μ_i, zs, αs, τs)

function calc_tau_bezier!(μ_i, zs, αs, τs)
    # get indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # length and precompute constants
    N = length(zs)
    inv_μ = 1.0 / μ_i
    one_third = 1.0 / 3.0

    # loop over wavelengths
    @inbounds for j in idx:sdx:size(αs,2)
            # views of arrays
        αv = @view αs[:,j]
        τv = @view τs[:,j]

        # bounds for clamping
        αmin = αv[1]; αmax = αmin
        for p in 2:N
            v = αv[p]
            αmin = v < αmin ? v : αmin
            αmax = v > αmax ? v : αmax
        end
        lo = 0.5 * αmin
        hi = 2.0 * αmax

        # init
        τv[1] = 1e-5

        # first iteration handle outside loop
        ds0 = (zs[2] * inv_μ - zs[1] * inv_μ)
        ds1 = (zs[3] * inv_μ - zs[2] * inv_μ)
        αC = one_third * (1.0 + ds1 / (ds0 + ds1))
        prev_dC = (αv[2] - αv[1]) / ds0
        dC = (αv[3] - αv[2]) / ds1

        ybar = (prev_dC * dC) / (αC * dC + (1.0 - αC) * prev_dC)
        C0 = αv[2] + 0.5 * ds0 * ybar
        C1 = αv[2] - 0.5 * ds1 * ybar
        Cf = C0
        Cf = Cf < lo ? lo : (Cf > hi ? hi : Cf)

        # update tau
        s_prev = zs[1] * inv_μ
        s_t = zs[2] * inv_μ
        τv[2] = τv[1] + (s_prev - s_t) * one_third * (αv[1] + αv[2] + Cf)

        # for next iteration
        prev_dC = dC
        prev_C0 = C0
        prev_C1 = C1
        s_prev = s_t

        # loop until final step
        @inbounds for t in 2:N-2
            s_t = zs[t+1] * inv_μ
            s_next = zs[t+2] * inv_μ
            ds0 = s_t - s_prev 
            ds1 = s_next - s_t

            αC = one_third * (1.0 + ds1 / (ds0 + ds1))
            dC = (αv[t+2] - αv[t+1]) / ds1

            ybar = (prev_dC * dC) / (αC * dC + (1.0 - αC) * prev_dC)
            C0 = αv[t+1] + 0.5 * ds0 * ybar
            C1 = αv[t+1] - 0.5 * ds1 * ybar
            Cf = 0.5 * (C0 + prev_C1)
            Cf = Cf < lo ? lo : (Cf > hi ? hi : Cf)

            # update tau
            τv[t+1] = τv[t] + (s_prev - s_t) * one_third * (αv[t] + αv[t+1] + Cf)

            # for next iteration
            prev_dC = dC
            prev_C0 = C0
            prev_C1 = C1
            s_prev = s_t
        end

        # handle last step outside loop
        s_t = zs[N] * inv_μ
        ds0 = s_prev - s_t
        Cf = prev_C1
        Cf = Cf < lo ? lo : (Cf > hi ? hi : Cf)
        @inbounds τv[N] = τv[N-1] + (one_third * ds0) * (αv[N-1] + αv[N] + Cf)
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

