struct StellarProps{T<:AF, V<:Union{T, AbstractVector{T}}}
    Teff::T
    logg::T
    Fe_H::T
    A_X::Vector{T}
    vsini::T
    ζ::T
    ξ::V
    ρstar::T
    istar::T
    α₂::T
    α₄::T
end

"""
    StellarProps(; Teff=NaN, logg=NaN, Fe_H=NaN, vsini=0.0, v_macro=NaN, v_micro=NaN,
                  ρstar=1.0, istar=90.0, α₂=0.0, α₄=0.0)

Container for stellar parameters and broadening properties used by `calc_formation_temp`.

Keyword arguments:
- `Teff`: effective temperature (K).
- `logg`: log10 surface gravity (cgs).
- `Fe_H`: metallicity [Fe/H] (dex); used to build the abundance vector `A_X`.
- `vsini`: projected rotational velocity (m/s).
- `v_macro`: macroturbulent velocity scale ζ (m/s); if `NaN`, uses `vmac_fit(Teff, logg)`.
- `v_micro`: microturbulent velocity ξ (m/s). Scalar for uniform broadening; vector of
  length `Natm` for per-layer broadening. If scalar `NaN`, uses `vmic_fit(Teff)`.
- `ρstar`: stellar radius scale factor for disk integration (dimensionless; default 1).
- `istar`: stellar inclination (degrees; 90 = equator-on). No physical effect for rigid
  rotation (`α₂=α₄=0`), where `v_los = -vsini·x_sky/ρstar` and the broadening follows from the
  projected `vsini` alone. It matters once differential rotation is on, where it selects which
  latitude bands are visible and how they are weighted. Numerically it is an exact no-op for
  `method=:quadrature`; for `method=:disk` at finite `Nϕ` it changes which tiles are visible,
  shifting the result at the discretization level.
- `α₂`, `α₄`: differential-rotation coefficients in the normalized rate law
  `Ω(ϕ)/Ω_eq = f(ϕ) = 1 - α₂·sin²ϕ - α₄·sin⁴ϕ`. Default 0 (solid body); positive values make
  the equator rotate faster than the poles. `vsini` stays the projected equatorial velocity
  (`f(0)=1`).

Struct fields:
- `Teff`, `logg`, `Fe_H`, `A_X`: atmosphere parameters (A_X is the full abundance vector).
- `vsini`, `ζ`, `ξ`: rotational, macroturbulent, and microturbulent velocities (m/s).
  `ξ` is `T` (scalar) or `AbstractVector{T}` (per-layer).
- `ρstar`, `istar`, `α₂`, `α₄`: disk integration parameters (`α₂`, `α₄` are the
  differential-rotation coefficients).

See also: [`vmac_fit`](@ref), [`vmic_fit`](@ref), [`calc_formation_temp`](@ref)
"""
function StellarProps(;Teff=NaN, logg=NaN, Fe_H=NaN, vsini=0.0, v_macro=NaN, v_micro=NaN,
                      ρstar=1.0, istar=90.0, α₂=0.0, α₄=0.0)
    # get the abundances
    A_X = Korg.format_A_X(Fe_H)

    # get macro
    if isnan(v_macro)
        ζ = vmac_fit(Teff, logg)
    else
        ζ = v_macro
    end

    # get micro
    if v_micro isa AbstractVector
        ξ = v_micro
    elseif isnan(v_micro)
        ξ = vmic_fit(Teff)
    else
        ξ = v_micro
    end

    return StellarProps(Teff, logg, Fe_H, A_X, vsini, ζ, ξ, ρstar, istar, α₂, α₄)
end
