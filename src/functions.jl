function blackbody_gpu(T::Float64, λ::Float64)
    # refactor exponential
    λ5 = λ * λ * λ * λ * λ

    # planck law
    num = fma(2.0, h, 0.0) * (c^2.0) / λ5
    den = (exp(h * c / λ / kB / T) - 1.0)
    return num / den
end
