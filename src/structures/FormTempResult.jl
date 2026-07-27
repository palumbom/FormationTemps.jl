"""
    FormTempResult(wavs, flux, form_temps, cont_func, atmosphere; r_thresh=BOUNDARY_R_THRESH)
    FormTempResult(wavs, flux, form_temps, cont_func, ceiling_ratio, r_thresh, atmosphere)

Container for `calc_formation_temp` outputs.

Fields:
- `wavs`: wavelength grid (Angstrom).
- `flux`: normalized flux across the grid.
- `form_temps`: formation temperature (K) at cumulative flux contribution of 0.5.
- `cont_func`: differential contribution function (C × Δτ), size `(Natm - 1, Nλ)`.
- `ceiling_ratio`: per-wavelength top-of-atmosphere contamination statistic; see
  [`ceiling_ratio`](@ref).
- `r_thresh`: the contamination threshold this result was computed with. [`boundary_mask`](@ref)
  defaults to it, so the mask matches the wavelengths the calculation warned about.
- `atmosphere`: atmosphere structure used for the calculation.

The five-argument form derives `ceiling_ratio` from `cont_func` and is what the internal paths
use; prefer it over the seven-argument form, which cannot enforce that the two agree.
"""
struct FormTempResult{T<:AF}
    wavs::Vector{T}
    flux::Vector{T}
    form_temps::Vector{T}
    cont_func::Matrix{T}
    ceiling_ratio::Vector{T}
    r_thresh::T
    atmosphere::Atmosphere{T}
end

# ceiling_ratio is derived, not passed, so it cannot disagree with cont_func
function FormTempResult(wavs::Vector{T}, flux, form_temps, cont_func, atmosphere;
                        r_thresh::Real=BOUNDARY_R_THRESH) where T<:AF
    return FormTempResult(wavs, flux, form_temps, cont_func, ceiling_ratio(cont_func),
                          T(r_thresh), atmosphere)
end
