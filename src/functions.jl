function blackbody_gpu(Temp::F, λ::F) where F<:AF
    λ5 = λ * λ * λ * λ * λ
    num = F(2) * F(h) * F(c)^2 / λ5
    den = exp(F(h) * F(c) / (λ * F(kB) * Temp)) - one(F)
    return num / den
end
