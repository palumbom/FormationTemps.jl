function make_grid(N::Integer)
    # create grid edges
    ϕe = range(deg2rad(-90.0), deg2rad(90.0), length=N)
    θe = range(deg2rad(0.0), deg2rad(360.0), length=N)
    return ϕe, θe
end

function get_grid_centers(grid::StepRangeLen)
    start = first(grid) + 0.5 * step(grid)
    stop = last(grid) - 0.5 * step(grid)
    return range(start, stop, length=length(grid)-1)
end

function get_grid_centers(grid::AA{T,1}) where T
    idx = findlast(x -> x .> 0.0, grid)
    return grid[1:idx-1] .+ (grid[2:idx] .- grid[1:idx-1])/2.0
end

function sphere_to_cart(ρ::T, ϕ::T, θ::T) where T
    # compute trig quantitites
    sinϕ = sin(ϕ)
    sinθ = sin(θ)
    cosϕ = cos(ϕ)
    cosθ = cos(θ)

    # now get cartesian coords
    x = ρ * cosϕ * sinθ
    y = ρ * sinϕ
    z = ρ * cosϕ * cosθ
    return [x, y, z]
end

function calc_mu(x, y, z, O⃗)
    dp = x * O⃗[1] + y * O⃗[2] + z * O⃗[3]
    n1 = sqrt(O⃗[1]^2.0 + O⃗[2]^2.0 + O⃗[3]^2.0)
    n2 = sqrt(x^2.0 + y^2.0 + z^2.0)
    return dp / (n1 * n2)
end

function sphere_to_cart_gpu(ρ, ϕ, θ)
    # compute trig quantities
    sinϕ = sin(ϕ)
    sinθ = sin(θ)
    cosϕ = cos(ϕ)
    cosθ = cos(θ)

    # now get cartesian coords
    x = ρ * cosϕ * sinθ
    y = ρ * sinϕ
    z = ρ * cosϕ * cosθ
    return x, y, z
end

function rotate_vector(x0, y0, z0, R_x)
    # do dot product
    x1 = x0 * R_x[1,1] + y0 * R_x[1,2] + z0 * R_x[1,3]
    y1 = x0 * R_x[2,1] + y0 * R_x[2,2] + z0 * R_x[2,3]
    z1 = x0 * R_x[3,1] + y0 * R_x[3,2] + z0 * R_x[3,3]
    return x1, y1, z1
end

function rotation_period(ϕ, A, B, C)
    sinϕ = sin(ϕ)
    return 360.0/(A + B * sinϕ^2.0 + C * sinϕ^4.0)
end

# normalized differential-rotation rate factor f(ϕ) = Ω(ϕ)/Ω_eq, given sin(ϕ).
# α₂=α₄=0 → solid body (f≡1); positive α = equator faster than poles (solar-like).
diff_rot_factor(sinϕ, α₂, α₄) = one(sinϕ) - α₂ * sinϕ^2 - α₄ * sinϕ^4

function calc_dA(ρs, ϕc, dϕ, dθ)
    return ρs^2.0 * cos(ϕc) * dϕ * dθ
end

