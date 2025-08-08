
function round_to_power(x::Real)
    iszero(x) && return 0
    p = floor(Int, log10(abs(x)))
    return round(x, digits = -p)
end
