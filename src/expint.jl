@inline function exponential_integral_2_gpu(x::T) where {T<:AbstractFloat}
    x <= zero(T) && return one(T)  # E2(0)=1; domain here is x ≥ 0
    E1 = exponential_integral_1_gpu(x)
    return exp(-x) - x*E1
end

@inline function exponential_integral_1_gpu(x::T) where {T<:AbstractFloat}
    # E1(x) = -γ - log(x) - Σ_{k≥1} [(-x)^k / (k·k!)]  for x < 1
    # For x ≥ 1, use continued fraction evaluated by modified Lentz' method.
    if x < one(T)
        γ = T(0.57721566490153286060651209008240243104215933593992)  # Euler–Mascheroni
        s = zero(T)
        t = -x                         # (-x)^1 / 1!
        k = 1
        tol = sqrt(eps(T))
        while true
            add = t / T(k)             # term_k / k
            s += add
            abs(add) <= tol*(abs(s) + one(T)) && break
            k += 1
            t = t * (-x) / T(k)        # update to (-x)^k / k!
            k > 200 && break           # hard guard
        end
        return -γ - log(x) - s
    else
        # Continued fraction for E1(x); returns E1(x) = exp(-x) * h
        tiny = T(1e-30)                # avoid zero denominators (safe for Float32/64)
        b = x + one(T)
        c = inv(tiny)
        d = inv(b)
        h = d
        tol = sqrt(eps(T))
        @inbounds for n in 1:100
            a = -T(n*n)
            b += T(2)
            d = inv(a*d + b)
            c = b + a/c
            δ = c*d
            h *= δ
            abs(δ - one(T)) <= tol && break
        end
        return h * exp(-x)
    end
end
