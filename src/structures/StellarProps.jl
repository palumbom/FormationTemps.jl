struct StellarProps{T<:AF}
    Teff::T
    logg::T
    Fe_H::T
    A_X::Vector{T}
    vsini::T
    ζ::T
    ξ::T
    ρstar::T
    istar::T
end

"""
    StellarProps(; Teff=NaN, logg=NaN, Fe_H=NaN, vsini=0.0, v_macro=NaN, v_micro=NaN,
                  ρstar=1.0, istar=90.0)

Container for stellar parameters and broadening properties used by `calc_formation_temp`.

Keyword arguments:
- `Teff`: effective temperature (K).
- `logg`: log10 surface gravity (cgs).
- `Fe_H`: metallicity [Fe/H] (dex); used to build the abundance vector `A_X`.
- `vsini`: projected rotational velocity (m/s).
- `v_macro`: macroturbulent velocity scale ζ (m/s); if `NaN`, uses `vmac_fit(Teff, logg)`.
- `v_micro`: microturbulent velocity ξ (m/s); if `NaN`, uses `vmic_fit(Teff)`.
- `ρstar`: stellar radius scale factor for disk integration (dimensionless; default 1).
- `istar`: stellar inclination (degrees; 90 = equator-on).

Struct fields:
- `Teff`, `logg`, `Fe_H`, `A_X`: atmosphere parameters (A_X is the full abundance vector).
- `vsini`, `ζ`, `ξ`: rotational, macroturbulent, and microturbulent velocities (m/s).
- `ρstar`, `istar`: disk integration parameters.

See also: [`vmac_fit`](@ref), [`vmic_fit`](@ref), [`calc_formation_temp`](@ref)
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
