"""
    calc_stellar_grid(ρs, i, vsini, Nϕ)

Compute a stellar surface grid on the GPU for disk integration. Geometry follows 
that from  S. S. Vogt et al. (1987) and N. Piskunov & O. Kochukhov (2002). 

Arguments:
- `ρs::Real`: Stellar radius (in solar radii)
- `i::Real`: Inclination in degrees, in the range [-90, 90].
- `vsini::Real`: Projected rotational velocity.
- `Nϕ::Int`: Number of latitude bins; longitude bins vary with latitude.

Returns:
- `μs::CuArray`: Cosine of the angle between the surface normal and the line of sight per tile.
- `dA::CuArray`: Projected surface area per tile.
- `z_rot::CuArray`: Line-of-sight rotational velocity per tile.
- `z_cbs::CuArray`: Additional per-tile velocity term. Disused in this implementation. 
"""
function calc_stellar_grid(ρs::T1, i::T1, vsini::T1, Nϕ::Int) where T1<:AF
    # allocate on GPU
    μs = CUDA.zeros(T1, Nϕ, 2 * Nϕ)
    dA = CUDA.zeros(T1, Nϕ, 2 * Nϕ)
    z_rot = CUDA.zeros(T1, Nϕ, 2 * Nϕ)
    z_cbs = CUDA.zeros(T1, Nϕ, 2 * Nϕ)

    # calculate in place and return
    calc_stellar_grid!(ρs, i, vsini, Nϕ, μs, dA, z_rot, z_cbs)
    return μs, dA, z_rot, z_cbs
end

function calc_stellar_grid_cpu(ρs::T, i::T, vsini::T, Nϕ::Int) where T<:AF
    ϕe = range(deg2rad(-90.0), deg2rad(90.0), length=Nϕ + 1)
    ϕc = get_grid_centers(ϕe)

    Nθ = ceil.(Int, 2π .* cos.(ϕc) ./ step(ϕe))
    Nθ_max = maximum(Nθ)

    μs = zeros(T, Nϕ, Nθ_max)
    dA = zeros(T, Nϕ, Nθ_max)
    z_rot = zeros(T, Nϕ, Nθ_max)

    dϕ = π / Nϕ
    iₛ = deg2rad(90.0 - i)
    R_x = [one(T) zero(T) zero(T);
           zero(T) cos(iₛ) -sin(iₛ);
           zero(T) sin(iₛ) cos(iₛ)]
    O⃗ = T[zero(T), zero(T), T(1e12)]

    for m in 1:Nϕ
        ϕc_m = ϕc[m]
        dθ = 2π / Nθ[m]
        for n in 1:Nθ[m]
            θc = (dθ / 2.0) + (n - 1) * dθ
            coords = sphere_to_cart(ρs, ϕc_m, θc)
            x = coords[1]
            y = coords[2]
            z = coords[3]

            # sky-plane x-coordinate (⊥ projected spin axis); invariant under R_x
            x_sky = x

            x, y, z = rotate_vector(x, y, z, R_x)
            μ_tile = calc_mu(x, y, z, O⃗)
            if μ_tile <= 0.0
                continue
            end
            μs[m, n] = μ_tile
            dA[m, n] = calc_dA(ρs, ϕc_m, dϕ, dθ) * μ_tile

            # projected solid-body LOS velocity; vsini is the projected velocity,
            # so this is independent of inclination: v_los = vsini * x_sky / ρs
            z_rot[m, n] = -(vsini / c_ms) * (x_sky / ρs)
        end
    end
    if iszero(vsini)
        z_rot .= 0.0
    end
    return μs, dA, z_rot
end

function calc_stellar_grid!(ρs::T1, inclination::T1, vsini::T1, Nϕ::Int,
                            μs::CuArray{T2,2}, dA::CuArray{T2,2},
                            z_rot::CuArray{T2,2}, z_cbs::CuArray{T2,2}) where {T1<:AF, T2<:AF}
    # get precision from GPU allocs
    precision = eltype(μs)

    # convert scalars from disk params to desired precision
    ρs = convert(precision, ρs)
    vsini = convert(precision, vsini)

    # get latitude grid
    ϕe = range(deg2rad(-90.0), deg2rad(90.0), length=Nϕ+1)
    ϕc = get_grid_centers(ϕe)

    # make longitude grid
    Nθ = ceil.(Int, 2π .* cos.(ϕc) ./ step(ϕe))
    θe = zeros(Nϕ+1, maximum(Nθ)+1)
    θc = zeros(Nϕ, maximum(Nθ))
    for i in eachindex(Nθ)
        # initialize edges
        edges = collect(range(0.0, 2π, length=Nθ[i]+1))

        # assign grid center and edges values
        θc[i, 1:Nθ[i]] .= get_grid_centers(edges)
        θe[i, 1:Nθ[i]+1] .= collect(edges)
    end
    Nθ_max = maximum(Nθ)

    # create rotation matrix for inclination
    @assert -90.0 <= inclination <= 90.0
    iₛ = deg2rad(90.0 - inclination)
    R_x = [1.0 0.0 0.0;
           0.0 cos(iₛ) -sin(iₛ);
           0.0 sin(iₛ) cos(iₛ)]

    # copy data to GPU
    @cusync begin
        # get observer vector and rotation matrix
        O⃗ = CuArray{precision}([0.0, 0.0, 1e12]) # CuArray{precision}(gprops.O⃗)
        Nθ = CuArray{Int32}(Nθ)
        R_x = CuArray{precision}(R_x)
    end

    # compute geometric parameters
    threads1 = 512
    blocks1 = cld(Nϕ * Nθ_max, prod(threads1))
    @cusync @captured @cuda threads=threads1 blocks=blocks1 calc_stellar_grid!(μs, dA, z_rot, Nϕ,
                                                                               Nθ_max, Nθ, R_x, O⃗,
                                                                               ρs, vsini)

    # safety for vsini = 1
    if iszero(vsini)
        z_rot .= 0.0
    end
    return nothing
end

function calc_stellar_grid!(μs, dA, z_rot, Nϕ, Nθ_max, Nθ, R_x, O⃗, ρs, vsini)
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

        # sky-plane x-coordinate (⊥ projected spin axis); invariant under R_x
        x_sky = x

        # rotate xyz by inclination and calculate mu
        x, y, z = rotate_vector(x, y, z, R_x)
        μ_tile = calc_mu(x, y, z, O⃗)
        if μ_tile <= 0.0
            continue
        end
        @inbounds μs[m,n] = μ_tile

        # get projected area element
        @inbounds dA[m,n] = calc_dA(ρs, ϕc, dϕ, dθ) * μ_tile

        # projected solid-body LOS velocity; vsini is the projected velocity, so
        # this is independent of inclination: v_los = vsini * x_sky / ρs
        # TODO differential rotation
        @inbounds z_rot[m,n] = -(vsini / c_ms) * (x_sky / ρs)
    end
    return nothing
end
