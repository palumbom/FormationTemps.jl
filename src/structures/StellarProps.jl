struct StellarProps{T<:AF}
    Teff::T
    logg::T
    Fe_H::T
    A_X::AA{T,1}
    vsini::T
    ζ::T
    ξ::T
    ρstar::T
    istar::T
end

"""
    StellarProps(; Teff=NaN, logg=NaN, Fe_H=NaN, vsini=0.0, v_macro=NaN, v_micro=NaN,
                  ρstar=1.0, istar=90.0)

Container for stellar parameters and broadening properties used by FormationTemps.

Keyword arguments:
- `Teff`: effective temperature (K).
- `logg`: log10 surface gravity (cgs).
- `Fe_H`: metallicity [Fe/H] (dex), used to build `A_X`.
- `vsini`: projected rotation velocity (m/s).
- `v_macro`: macroturbulent velocity (m/s); if `NaN`, uses `vmac_fit(Teff, logg)`.
- `v_micro`: microturbulent velocity (m/s); if `NaN`, uses `vmic_fit(Teff)`.
- `ρstar`: stellar radius scale factor for disk integration (dimensionless).
- `istar`: inclination angle in degrees (90 = equator-on).
"""
function StellarProps(;Teff=NaN, logg=NaN, Fe_H=NaN, vsini=0.0, v_macro=NaN, v_micro=NaN,
                      ρstar=1.0, istar=90.0)
    # get the abundances
    A_X = Korg.format_A_X(Fe_H)

    # get macro
    if isnan(v_macro)
        ζ = vmac_fit(Teff, logg)
    else
        ζ = v_macro
    end

    # get micro
    if isnan(v_micro)
        ξ = vmic_fit(Teff)
    else 
        ξ = v_micro
    end

    return StellarProps(Teff, logg, Fe_H, A_X, vsini, ζ, ξ, ρstar, istar)
end
