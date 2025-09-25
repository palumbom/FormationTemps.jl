function calc_stellar_grid!(gprops::GridProperties, μs::CuArray{T,2},
                            ld::CuArray{T,2}, dA::CuArray{T,2},
                            z_rot::CuArray{T,2}, z_cbs::CuArray{T,2}) where T<:AF
    # get precision from GPU allocs
    precision = eltype(μs)

    # convert scalars from disk params to desired precision
    ρs = convert(precision, gprops.ρs)
    A = convert(precision, gprops.A)
    B = convert(precision, gprops.B)
    C = convert(precision, gprops.C)
    v0 = convert(precision, gprops.v0)

    # get size of sub-tiled grid
    Nϕ = gprops.N
    Nθ_max = maximum(gprops.Nθ)

    # copy data to GPU
    @cusync begin
        # get observer vector and rotation matrix
        O⃗ = CuArray{precision}(gprops.O⃗)
        Nθ = CuArray{Int32}(gprops.Nθ)
        R_x = CuArray{precision}(gprops.R_x)
    end

    # compute geometric parameters, average over subtiles
    threads1 = 512
    blocks1 = cld(Nϕ * Nθ_max, prod(threads1))
    @cusync @captured @cuda threads=threads1 blocks=blocks1 calc_stellar_grid!(μs, ld, dA, z_rot, Nϕ,
                                                                               Nθ_max, Nθ, R_x, O⃗,
                                                                               ρs, A, B, C, v0, u1, u2)

    return nothing
end

function calc_stellar_grid!(μs, ld, dA, z_rot, Nϕ, Nθ_max, Nθ, R_x, O⃗, ρs, A, B, C, v0, u1, u2)
    # get indices from GPU blocks + threads
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # total number of elements output array
    num_tiles = Nϕ * Nθ_max

    # get latitude step size
    dϕ = π / Nϕ

    # linear index over course grid tiles
    for t in idx:sdx:num_tiles
        # get index for output array
        row = (t - 1) ÷ Nθ_max
        col = (t - 1) % Nθ_max
        m = row + 1
        n = col + 1

        # don't do nonsense tile
        n > Nθ[m] && continue

        # get coordinates of latitude tile center
        ϕc = -π/2 + (dϕ/2.0) + (m - 1) * dϕ

        # get longitude tile step size
        N_θ_edges = Nθ[m]
        dθ = 2π / (N_θ_edges)

        # get longitude
        θc = (dθ/2.0) + (n - 1) * dθ

        # get cartesian coords
        x, y, z = sphere_to_cart_gpu(ρs, ϕc, θc)

        # get vector from spherical circle center to surface patch
        a = x
        b = CUDA.zero(CUDA.eltype(μs))
        c = z

        # take cross product to get vector in direction of rotation
        d = - ρs * c
        e = CUDA.zero(CUDA.eltype(μs))
        f = ρs * a

        # make it a unit vector
        def_norm = CUDA.sqrt(d^2.0 + e^2.0 + f^2.0)
        d /= def_norm
        e /= def_norm
        f /= def_norm

        # set magnitude by differential rotation
        rp = 2π * ρs * CUDA.cos(ϕc) / GRASS.rotation_period_gpu(ϕc, A, B, C)

        # get in units of c
        rp /= c_Rsun_day

        # set magnitude of vector
        d *= rp
        e *= rp
        f *= rp

        # rotate xyz by inclination
        x, y, z = rotate_vector_gpu(x, y, z, R_x)

        # rotate xyz by inclination and calculate mu
        μ_tile = calc_mu_gpu(x, y, z, O⃗)
        if μ_tile <= 0.0
            continue
        end
        @inbounds μs[m,n] = μ_tile

        # get projected area element
        @inbounds dA[m,n] = calc_dA_gpu(ρs, ϕc, dϕ, dθ) * μ_tile

        # rotate the velocity vectors by inclination
        d, e, f = rotate_vector_gpu(d, e, f, R_x)

        # get vector pointing from observer to surface patch
        a = x - O⃗[1]
        b = y - O⃗[2]
        c = z - O⃗[3]

        # get angle between them
        n1 = CUDA.sqrt(a^2.0 + b^2.0 + c^2.0)
        n2 = CUDA.sqrt(d^2.0 + e^2.0 + f^2.0)
        angle = (a * d + b * e + c * f) / (n1 * n2)
        @inbounds z_rot[m,n] = n2 * angle
    end
    return nothing
end

