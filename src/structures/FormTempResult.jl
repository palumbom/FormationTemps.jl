"""
    FormTempResult(wavs, flux, form_temps, cont_func)

Container for `calc_formation_temp` outputs.

Fields:
- `wavs`: wavelength grid (Angstrom).
- `flux`: normalized flux across the grid.
- `form_temps`: formation temperature (K) at cumulative flux contribution of 0.5.
- `cont_func`: differential contribution function (dC/dtau), size `(Natm - 1, Nλ)`.
"""
struct FormTempResult{T<:AF}
    wavs::AA{T,1}
    flux::AA{T,1}
    form_temps::AA{T,1}
    cont_func::AA{T,2}
end
