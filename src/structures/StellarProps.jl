struct StellarProps{T<:AF}
    Teff::T
    logg::T
    Fe_H::T
    A_X::AA{T,1}
    vsini::T
    ζ::T
    ξ::T
end

function StellarProps(Teff=NaN, logg=NaN, Fe_H=NaN, vsini=0.0, v_macro=NaN, v_micro=NaN)
    # get the abundances
    A_X = Korg.format_A_X(Fe_H)

    # get macro
    if isnan(v_macro)
        ζ = vmac_fit(Teff, logg)
    end

    # get micro
    if isnan(v_micro)
        ξ = vmic_fit(Teff)
    end

    return StellarProps(Teff, logg, Fe_H, A_X, vsini, ζ, ξ)
end