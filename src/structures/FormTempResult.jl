"""
    FormTempResult(wavs, flux, form_temps, cont_func, atmosphere)
    FormTempResult(wavs, flux, form_temps, cont_func, ceiling_ratio, atmosphere)

Container for `calc_formation_temp` outputs.

Fields:
- `wavs`: wavelength grid (Angstrom).
- `flux`: normalized flux across the grid.
- `form_temps`: formation temperature (K) at cumulative flux contribution of 0.5.
- `cont_func`: differential contribution function (C × Δτ), size `(Natm - 1, Nλ)`.
- `ceiling_ratio`: per-wavelength top-of-atmosphere contamination statistic; see
  [`ceiling_ratio`](@ref) and [`boundary_mask`](@ref).
- `atmosphere`: atmosphere structure used for the calculation.

The five-argument form derives `ceiling_ratio` from `cont_func`, which is how the internal
paths build results; it is the only way to guarantee the two stay consistent.
"""
struct FormTempResult{T<:AF}
    wavs::Vector{T}
    flux::Vector{T}
    form_temps::Vector{T}
    cont_func::Matrix{T}
    ceiling_ratio::Vector{T}
    atmosphere::Atmosphere{T}
end

# derive ceiling_ratio from cont_func rather than accepting it separately, so a caller
# cannot supply a statistic that disagrees with the contribution function it came from
FormTempResult(wavs, flux, form_temps, cont_func, atmosphere::Atmosphere) =
    FormTempResult(wavs, flux, form_temps, cont_func, ceiling_ratio(cont_func), atmosphere)
