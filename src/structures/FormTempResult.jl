"""
    FormTempResult(wavs, flux, form_temps, cont_func, atmosphere)

Container for `calc_formation_temp` outputs.

Fields:
- `wavs`: wavelength grid (Angstrom).
- `flux`: normalized flux across the grid.
- `form_temps`: formation temperature (K) at cumulative flux contribution of 0.5.
- `cont_func`: differential contribution function (C × Δτ), size `(Natm - 1, Nλ)`.
- `atmosphere`: atmosphere structure used for the calculation.
"""
struct FormTempResult{T<:AF}
    wavs::Vector{T}
    flux::Vector{T}
    form_temps::Vector{T}
    cont_func::Matrix{T}
    atmosphere::Atmosphere{T}
end
