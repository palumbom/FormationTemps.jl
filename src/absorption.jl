"""
    compute_alpha!(αs, wls, linelist, atm, A_X; partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1)
    compute_alpha!(αs, wls, linelist, zs, Ts, nds, nes, A_X; partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1)

Compute total (continuum + line) absorption coefficients in-place.

Arguments:
- `αs::AbstractArray{<:Real}`: Output array for absorption coefficients, sized `(Nlayers, Nλ)`.
- `wls::Korg.Wavelengths`: Wavelength grid for the absorption calculation.
- `linelist`: Line list passed to `Korg.line_absorption!`.
- `atm::Atmosphere` or `(zs, Ts, nds, nes)`: Atmospheric structure (heights, temperatures, number densities, electron densities).
- `A_X::AbstractVector{<:Real}`: Elemental abundances on the usual astronomical scale.
- `partition_funcs=Korg.default_partition_funcs`: Partition function table for chemical equilibrium.
- `ne_warn_thresh=0.1`: Relative warning threshold for electron density updates.
- `hydrogen_lines=true`: Include hydrogen (Balmer/Brackett) lines, which Korg computes from
  dedicated Stark/MHD physics rather than from the linelist.
- `hydrogen_line_window_size_Å=150.0`: Per-line hydrogen window (matches Korg's default).
- `use_MHD=nothing`: MHD occupation-probability formalism for the hydrogen lines. `nothing`
  enables it only below 13000 Å; pass `true` to reproduce Korg's default at all wavelengths.
  See [`_add_hydrogen_line_absorption!`](@ref).

Returns:
- `nothing`: `αs` is filled in-place.

Notes:
- Adapted from `Korg.line_absorption!`.
"""
function compute_alpha!(αs, wls::Korg.Wavelengths, linelist,
                        atm::Atmosphere{T}, A_X::AA{T,1};
                        α_ref_out=nothing, vmic_ref_cms::Float64=0.0,
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1, ne_warn_min=1e-4,
                        line_buffer_Å::Float64=10.0,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing) where T<:AF
    compute_alpha!(αs, wls, linelist, atm.zs, atm.Ts, atm.nd, atm.nₑ,
                   A_X; α_ref_out=α_ref_out, ref_wl_cm=atm.reference_wavelength,
                   vmic_ref_cms=vmic_ref_cms,
                   partition_funcs=partition_funcs, ne_warn_thresh=ne_warn_thresh,
                   ne_warn_min=ne_warn_min, line_buffer_Å=line_buffer_Å,
                   hydrogen_lines=hydrogen_lines,
                   hydrogen_line_window_size_Å=hydrogen_line_window_size_Å,
                   use_MHD=use_MHD)
    return nothing
end

"""
    compute_alpha!(αs, αs_cont, wls, linelist, atm, A_X; partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1)
    compute_alpha!(αs, αs_cont, wls, linelist, zs, Ts, nds, nes, A_X; partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1)

Compute continuum and total absorption coefficients in-place.

Arguments:
- `αs::AbstractArray{<:Real}`: Output array for total (continuum + line) absorption.
- `αs_cont::AbstractArray{<:Real}`: Output array for continuum-only absorption.
- `wls::Korg.Wavelengths`: Wavelength grid for the absorption calculation.
- `linelist`: Line list passed to `Korg.line_absorption!`.
- `atm::Atmosphere` or `(zs, Ts, nds, nes)`: Atmospheric structure (heights, temperatures, number densities, electron densities).
- `A_X::AbstractVector{<:Real}`: Elemental abundances on the usual astronomical scale.
- `partition_funcs=Korg.default_partition_funcs`: Partition function table for chemical equilibrium.
- `ne_warn_thresh=0.1`: Relative warning threshold for electron density updates.
- `hydrogen_lines=true`, `hydrogen_line_window_size_Å=150.0`, `use_MHD=nothing`: hydrogen
  line treatment; see the single-output method above. Hydrogen opacity is added to `αs`
  only, never to `αs_cont`, so Balmer lines appear as features against the true continuum.

Returns:
- `nothing`: `αs` and `αs_cont` are filled in-place.
"""

function compute_alpha!(αs, wls::Korg.Wavelengths, linelist, zs, Ts, nds, nes, A_X;
                        α_ref_out=nothing, ref_wl_cm::Float64=5000.0*ANGSTROM_TO_CM,
                        vmic_ref_cms::Float64=0.0,
                        partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1,
                        ne_warn_min=1e-4,
                        line_buffer_Å::Float64=10.0,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing)
    # deal with abundances
    abs_abundances = @. 10^(A_X - 12) # n(X) / n_tot
    abs_abundances ./= sum(abs_abundances) #normalize so that sum(n(X)/n_tot) = 1

    # work in cm
    cntm_step = ANGSTROM_TO_CM
    line_buffer = line_buffer_Å * ANGSTROM_TO_CM

    # wavelengths at which to calculate the continuum
    cntm_wls = range(first(wls) - line_buffer, last(wls) + line_buffer, step=cntm_step)
    cntm_wls = Korg.Wavelengths(cntm_wls)

    # allocate for chemical equilibrium solver
    N = length(zs)
    triples = Vector{Tuple{Float64, Dict, typeof(Korg.linear_interpolation(cntm_wls, zeros(length(cntm_wls))))}}(undef, N)

    # reference frequency for α_ref (c_cgs in cm/s, ref_wl_cm in cm → Hz)
    ref_freq = Korg.c_cgs / ref_wl_cm

    # loop over layers and do chemical equilibrium
    Threads.@threads for i in 1:N
        # index the layers
        temp = Ts[i]
        nd = nds[i]
        ne = nes[i]

        # compute equilibrium
        nₑ, n_dict = Korg.chemical_equilibrium(temp, nd, ne, abs_abundances,
                                               Korg.ionization_energies,
                                               partition_funcs,
                                               Korg.default_log_equilibrium_constants,
                                               electron_number_density_warn_threshold=ne_warn_thresh,
                                               electron_number_density_warn_min_value=ne_warn_min)

        # continuum absorption at reference wavelength — reuses (nₑ, n_dict), no extra solver call
        if !isnothing(α_ref_out)
            α_ref_out[i] = Korg.total_continuum_absorption([ref_freq], temp, nₑ, n_dict, partition_funcs)[1]
        end

        # continuum absorption
        α_cntm_vals = reverse(Korg.total_continuum_absorption(Korg.eachfreq(cntm_wls),
                              temp, nₑ, n_dict, partition_funcs))
        α_cntm_layer = Korg.linear_interpolation(cntm_wls, α_cntm_vals)

        # write into shared array (distinct rows → no races)
        αs[i, :] .= α_cntm_layer(wls)

        # collect results
        triples[i] = (nₑ, n_dict, α_cntm_layer)
    end

    # slice out the results
    nₑs = first.(triples)

    # put number densities in a dict of vectors, rather than a vector of dicts.
    n_dicts = getindex.(triples, 2)
    nds = Dict([spec => [n[spec] for n in n_dicts]
               for spec in keys(n_dicts[1])
               if spec != Korg.Species("H III")])

    #vector of continuum-absorption interpolators
    α_cntm = last.(triples)

    # now do the line absorption
    Korg.line_absorption!(αs, linelist, wls, Ts, nₑs, nds, partition_funcs,
                          0.0, α_cntm; cutoff_threshold=3e-4)

    # hydrogen (Balmer/Brackett) lines — Korg treats these separately from the linelist
    if hydrogen_lines
        _add_hydrogen_line_absorption!(αs, wls, Ts, nₑs,
                                       [nd[_HI_SPECIES] for nd in n_dicts],
                                       [nd[_HeI_SPECIES] for nd in n_dicts],
                                       partition_funcs, hydrogen_line_window_size_Å; use_MHD=use_MHD)
    end

    # mirror Korg synthesize.jl lines 253-265: add line opacity at the reference wavelength.
    # Korg filters the user linelist to lines near ref_wl then merges with its internal
    # _alpha_5000_default_linelist; for MARCS (ref_wl = 5000 Å) that internal list is always used.
    # Korg also calls interpolate_molecular_cross_sections! here, which is a no-op when no
    # molecular cross-sections are loaded (the common case).
    if !isnothing(α_ref_out)
        linelist5 = Korg._alpha_5000_default_linelist  # synthesize.jl:195-198
        α_cntm_ref = [_ -> a for a in copy(α_ref_out)]  # synthesize.jl:255
        Korg.line_absorption!(view(α_ref_out, :, 1:1), linelist5,
                              Korg.Wavelengths([ref_wl_cm * CM_TO_ANGSTROM]),
                              Ts, nₑs, nds, partition_funcs,
                              vmic_ref_cms, α_cntm_ref; cutoff_threshold=3e-4)
    end

    return nothing
end

function compute_alpha!(αs, αs_cont, wls::Korg.Wavelengths, linelist, atm, A_X;
                        α_ref_out=nothing, vmic_ref_cms::Float64=0.0,
                        partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1,
                        ne_warn_min=1e-4, line_buffer_Å::Float64=10.0,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing)
    compute_alpha!(αs, αs_cont, wls, linelist, atm.zs, atm.Ts, atm.nd, atm.nₑ,
                   A_X; α_ref_out=α_ref_out, ref_wl_cm=atm.reference_wavelength,
                   vmic_ref_cms=vmic_ref_cms,
                   partition_funcs=partition_funcs, ne_warn_thresh=ne_warn_thresh,
                   ne_warn_min=ne_warn_min, line_buffer_Å=line_buffer_Å,
                   hydrogen_lines=hydrogen_lines,
                   hydrogen_line_window_size_Å=hydrogen_line_window_size_Å,
                   use_MHD=use_MHD)
    return nothing
end

function compute_alpha!(αs, αs_cont, wls::Korg.Wavelengths, linelist, zs, Ts, nds, nes, A_X;
                        α_ref_out=nothing, ref_wl_cm::Float64=5000.0*ANGSTROM_TO_CM,
                        vmic_ref_cms::Float64=0.0,
                        partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1,
                        ne_warn_min=1e-4,
                        line_buffer_Å::Float64=10.0,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing)
    # deal with abundances
    abs_abundances = @. 10^(A_X - 12) # n(X) / n_tot
    abs_abundances ./= sum(abs_abundances) #normalize so that sum(n(X)/n_tot) = 1

    # work in cm
    cntm_step = ANGSTROM_TO_CM
    line_buffer = line_buffer_Å * ANGSTROM_TO_CM

    # wavelengths at which to calculate the continuum
    cntm_wls = range(first(wls) - line_buffer, last(wls) + line_buffer, step=cntm_step)
    cntm_wls = Korg.Wavelengths(cntm_wls)

    # allocate for chemical equilibrium solver
    N = length(zs)
    triples = Vector{Tuple{Float64, Dict, typeof(Korg.linear_interpolation(cntm_wls, zeros(length(cntm_wls))))}}(undef, N)

    # reference frequency for α_ref (c_cgs in cm/s, ref_wl_cm in cm → Hz)
    ref_freq = Korg.c_cgs / ref_wl_cm

    # loop over layers and do chemical equilibrium
    Threads.@threads for i in 1:N
        # index the layers
        temp = Ts[i]
        nd = nds[i]
        ne = nes[i]

        # compute equilibrium
        nₑ, n_dict = Korg.chemical_equilibrium(temp, nd, ne, abs_abundances,
                                               Korg.ionization_energies,
                                               partition_funcs,
                                               Korg.default_log_equilibrium_constants,
                                               electron_number_density_warn_threshold=ne_warn_thresh,
                                               electron_number_density_warn_min_value=ne_warn_min)

        # continuum absorption at reference wavelength — reuses (nₑ, n_dict), no extra solver call
        if !isnothing(α_ref_out)
            α_ref_out[i] = Korg.total_continuum_absorption([ref_freq], temp, nₑ, n_dict, partition_funcs)[1]
        end

        # continuum absorption
        α_cntm_vals = reverse(Korg.total_continuum_absorption(Korg.eachfreq(cntm_wls),
                              temp, nₑ, n_dict, partition_funcs))
        α_cntm_layer = Korg.linear_interpolation(cntm_wls, α_cntm_vals)

        # write into shared array (distinct rows → no races)
        αs[i, :] .= α_cntm_layer(wls)

        # collect results
        triples[i] = (nₑ, n_dict, α_cntm_layer)
    end

    # slice out the results
    nₑs = first.(triples)

    # put number densities in a dict of vectors, rather than a vector of dicts.
    n_dicts = getindex.(triples, 2)
    nds = Dict([spec => [n[spec] for n in n_dicts]
               for spec in keys(n_dicts[1])
               if spec != Korg.Species("H III")])

    #vector of continuum-absorption interpolators
    α_cntm = last.(triples)
    αs_cont .= αs

    # now do the line absorption
    Korg.line_absorption!(αs, linelist, wls, Ts, nₑs, nds, partition_funcs,
                          0.0, α_cntm; cutoff_threshold=3e-4)

    # hydrogen (Balmer/Brackett) lines — added to total αs only (after αs_cont .= αs),
    # so they appear as a feature against the continuum. Korg treats these separately.
    if hydrogen_lines
        _add_hydrogen_line_absorption!(αs, wls, Ts, nₑs,
                                       [nd[_HI_SPECIES] for nd in n_dicts],
                                       [nd[_HeI_SPECIES] for nd in n_dicts],
                                       partition_funcs, hydrogen_line_window_size_Å; use_MHD=use_MHD)
    end

    # mirror Korg synthesize.jl lines 253-265: add line opacity at the reference wavelength.
    # Korg filters the user linelist to lines near ref_wl then merges with its internal
    # _alpha_5000_default_linelist; for MARCS (ref_wl = 5000 Å) that internal list is always used.
    # Korg also calls interpolate_molecular_cross_sections! here, which is a no-op when no
    # molecular cross-sections are loaded (the common case).
    if !isnothing(α_ref_out)
        linelist5 = Korg._alpha_5000_default_linelist  # synthesize.jl:195-198
        α_cntm_ref = [_ -> a for a in copy(α_ref_out)]  # synthesize.jl:255
        Korg.line_absorption!(view(α_ref_out, :, 1:1), linelist5,
                              Korg.Wavelengths([ref_wl_cm * CM_TO_ANGSTROM]),
                              Ts, nₑs, nds, partition_funcs,
                              vmic_ref_cms, α_cntm_ref; cutoff_threshold=3e-4)
    end
    return nothing
end

# hydrogen-line species handles, constructed once at load
const _HI_SPECIES = Korg.Species("H I")
const _HeI_SPECIES = Korg.Species("He I")

"""
    _add_hydrogen_line_absorption!(αs, wls, Ts, nₑs, nH_I, nHe_I, partition_funcs,
                                   window_size_Å; use_MHD=nothing)

Add hydrogen (Balmer/Brackett) line opacity to `αs` in-place, one atmosphere layer per row,
mirroring `Korg.synthesize`. Korg computes hydrogen lines from dedicated Stark/MHD physics
rather than from the linelist, so `Korg.line_absorption!` never emits them.

Notes:
- `ξ = 0`: microturbulence is applied downstream via FFT convolution on `αs` (as for the metal
  lines, which are passed `vmic=0`); passing a nonzero ξ here would double-count it.
- Not added to the 5000 Å reference opacity `α_ref`, matching Korg, which builds `α5` from
  the continuum plus a 5000 Å linelist and adds hydrogen lines only to `α`.
- `use_MHD=nothing` (the default) enables the MHD occupation-probability formalism only for
  `wls[end] < 13000 Å`. Korg's `use_MHD_for_hydrogen_lines` defaults to `true` at all
  wavelengths and warns above 13000 Å that `false` is preferable; this default applies that
  recommendation directly. Pass `use_MHD=true` to match Korg when comparing in the near-IR.
"""
function _add_hydrogen_line_absorption!(αs, wls::Korg.Wavelengths, Ts, nₑs, nH_I, nHe_I,
                                        partition_funcs, window_size_Å::Float64;
                                        use_MHD::Union{Nothing,Bool}=nothing)
    window_cm = window_size_Å * ANGSTROM_TO_CM
    use_MHD = isnothing(use_MHD) ? (wls[end] < 13000 * ANGSTROM_TO_CM) : use_MHD  # wls in cm
    H_I_pf = partition_funcs[_HI_SPECIES]
    Threads.@threads for i in eachindex(Ts)
        Korg.hydrogen_line_absorption!(view(αs, i, :), wls, Ts[i], nₑs[i],
                                       nH_I[i], nHe_I[i], H_I_pf(log(Ts[i])),
                                       0.0, window_cm; use_MHD=use_MHD)
    end
    return nothing
end
