const _HIII_SPECIES = Korg.Species("H III")

"""
    AlphaCache(wls, A_X, Nlayers; cntm_step=ANGSTROM_TO_CM, line_buffer=10*ANGSTROM_TO_CM)
    AlphaCache(wls, A_X, atm; cntm_step=ANGSTROM_TO_CM, line_buffer=10*ANGSTROM_TO_CM)

Reusable cache for accelerated `compute_alpha!` calls.

This cache implements:
- warm-started electron density guesses across columns,
- precomputed abundance normalization and continuum wavelength grid,
- reusable buffers for chemistry and line-absorption assembly.

Arguments:
- `wls::Korg.Wavelengths`: Wavelength grid for line and continuum absorption.
- `A_X::AbstractVector{<:Real}`: Elemental abundances on the usual astronomical scale.
- `Nlayers::Int` or `atm::Atmosphere`: Number of atmosphere layers, or an atmosphere struct.
- `cntm_step`: Continuum wavelength step (cm; default `ANGSTROM_TO_CM`, i.e. 1 Å).
- `line_buffer`: Buffer around the wavelength range for line wings (cm; default 10 Å).

See also: [`compute_alpha!`](@ref), [`set_abundances!`](@ref), [`reset_alpha_cache!`](@ref)
"""
mutable struct AlphaCache{T<:AF, TI}
    Nlayers::Int
    abs_abundances::Vector{T}
    cntm_wls::Korg.Wavelengths
    cntm_step::T
    line_buffer::T

    warm_ne::Vector{T}
    has_warm_ne::Bool
    ne_solved::Vector{T}
    species_layout_initialized::Bool

    species_keys::Vector{Korg.Species}
    species_density_vectors::Vector{Vector{T}}
    nds_by_species::Dict{Korg.Species, Vector{T}}
    α_cntm::Vector{TI}
end

function _normalize_abs_abundances(A_X::AA{T, 1}) where {T<:AF}
    abs_abundances = @. 10^(A_X - 12)
    abs_abundances ./= sum(abs_abundances)
    return abs_abundances
end

function AlphaCache(wls::Korg.Wavelengths, A_X::AA{T, 1}, Nlayers::Integer;
                    cntm_step::Real=ANGSTROM_TO_CM, line_buffer::Real=10*ANGSTROM_TO_CM) where {T<:AF}
    cs = T(cntm_step)
    lb = T(line_buffer)
    cntm_wls = Korg.Wavelengths(range(first(wls) - lb, last(wls) + lb, step=cs))

    dummy_itp = Korg.linear_interpolation(cntm_wls, zeros(T, length(cntm_wls)))
    return AlphaCache{T, typeof(dummy_itp)}(
        Int(Nlayers),
        _normalize_abs_abundances(A_X),
        cntm_wls,
        cs,
        lb,
        zeros(T, Int(Nlayers)),
        false,
        zeros(T, Int(Nlayers)),
        false,
        Korg.Species[],
        Vector{Vector{T}}(),
        Dict{Korg.Species, Vector{T}}(),
        [dummy_itp for _ in 1:Int(Nlayers)],
    )
end

function AlphaCache(wls::Korg.Wavelengths, A_X::AA{T, 1}, atm::Atmosphere{T};
                    cntm_step::Real=ANGSTROM_TO_CM, line_buffer::Real=10*ANGSTROM_TO_CM) where {T<:AF}
    return AlphaCache(wls, A_X, length(atm.zs); cntm_step=cntm_step, line_buffer=line_buffer)
end

"""
    set_abundances!(cache, A_X)

Refresh cached absolute abundances after changing `A_X`.
"""
function set_abundances!(cache::AlphaCache{T}, A_X::AA{T, 1}) where {T<:AF}
    cache.abs_abundances .= _normalize_abs_abundances(A_X)
    return cache
end

"""
    reset_alpha_cache!(cache)

Drop warm-start state while keeping allocated buffers.
"""
function reset_alpha_cache!(cache::AlphaCache)
    cache.has_warm_ne = false
    fill!(cache.warm_ne, zero(eltype(cache.warm_ne)))
    fill!(cache.ne_solved, zero(eltype(cache.ne_solved)))
    return cache
end

function _validate_alpha_cache(cache::AlphaCache, wls::Korg.Wavelengths, N::Int)
    N <= cache.Nlayers || throw(ArgumentError("AlphaCache layer count $(cache.Nlayers) < input layer count $N"))
    if first(wls) < first(cache.cntm_wls) || last(wls) > last(cache.cntm_wls)
        throw(ArgumentError("wavelength grid lies outside cached continuum grid; rebuild AlphaCache for this wls"))
    end
    return nothing
end

function _seed_ne_guesses!(cache::AlphaCache{T}, nes::AA{T, 1}, N::Int) where {T<:AF}
    tiny = eps(T)
    if cache.has_warm_ne
        @inbounds for i in 1:N
            cache.ne_solved[i] = max(cache.warm_ne[i], tiny)
        end
    else
        cache.ne_solved[1] = max(nes[1], tiny)
        @inbounds for i in 2:N
            # Depth-coupled fallback guess: use the previous-layer value.
            cache.ne_solved[i] = max(nes[i - 1], tiny)
        end
    end
    return nothing
end

function _initialize_species_layout!(cache::AlphaCache{T}, n_dict::Dict) where {T<:AF}
    if cache.species_layout_initialized
        return nothing
    end

    empty!(cache.species_keys)
    empty!(cache.species_density_vectors)
    empty!(cache.nds_by_species)
    for spec in keys(n_dict)
        if spec != _HIII_SPECIES
            vec = zeros(T, cache.Nlayers)
            push!(cache.species_keys, spec)
            push!(cache.species_density_vectors, vec)
            cache.nds_by_species[spec] = vec
        end
    end

    cache.species_layout_initialized = true
    return nothing
end

function _fill_species_layer!(cache::AlphaCache{T}, n_dict::Dict, i::Int) where {T<:AF}
    @inbounds for j in eachindex(cache.species_keys)
        cache.species_density_vectors[j][i] = n_dict[cache.species_keys[j]]
    end
    return nothing
end

function _solve_layer_chemistry!(cache::AlphaCache{T}, αs::AA{T, 2}, i::Int, wls::Korg.Wavelengths,
                                 Ts::AA{T, 1}, nds::AA{T, 1}, partition_funcs, ne_warn_thresh;
                                 fill_species::Bool=true) where {T<:AF}
    temp = Ts[i]
    nd = nds[i]
    ne_guess = cache.ne_solved[i]

    ne, n_dict = Korg.chemical_equilibrium(temp, nd, ne_guess, cache.abs_abundances,
                                           Korg.ionization_energies, partition_funcs,
                                           Korg.default_log_equilibrium_constants,
                                           electron_number_density_warn_threshold=ne_warn_thresh)

    α_cntm_vals = reverse(Korg.total_continuum_absorption(Korg.eachfreq(cache.cntm_wls),
                          temp, ne, n_dict, partition_funcs))
    α_cntm_layer = Korg.linear_interpolation(cache.cntm_wls, α_cntm_vals)

    @views αs[i, :] .= α_cntm_layer(wls)
    cache.ne_solved[i] = ne
    cache.α_cntm[i] = α_cntm_layer
    fill_species && _fill_species_layer!(cache, n_dict, i)
    return n_dict
end

function _fill_continuum_from_cache!(αs_cont::AA{T,2}, cache::AlphaCache{T},
                                     wls::Korg.Wavelengths) where {T<:AF}
    N = size(αs_cont, 1)
    N <= cache.Nlayers || throw(ArgumentError("continuum array layer count $(N) > cache layer count $(cache.Nlayers)"))
    @inbounds for i in 1:N
        @views αs_cont[i, :] .= cache.α_cntm[i](wls)
    end
    return nothing
end

function _compute_alpha_cached!(αs::AA{T, 2}, wls::Korg.Wavelengths, linelist, Ts::AA{T, 1},
                                nds::AA{T, 1}, nes::AA{T, 1}, cache::AlphaCache{T};
                                partition_funcs=Korg.default_partition_funcs,
                                ne_warn_thresh=0.1,
                                cutoff_threshold=3e-4,
                                threaded::Bool=true,
                                hydrogen_lines::Bool=true,
                                hydrogen_line_window_size_Å::Float64=150.0,
                                use_MHD::Union{Nothing,Bool}=nothing) where {T<:AF}
    N = length(Ts)
    N == 0 && return nothing

    _validate_alpha_cache(cache, wls, N)
    _seed_ne_guesses!(cache, nes, N)

    # solve one layer first to initialize the species layout once
    n_dict_first = _solve_layer_chemistry!(cache, αs, 1, wls, Ts, nds, partition_funcs,
                                           ne_warn_thresh; fill_species=false)
    _initialize_species_layout!(cache, n_dict_first)
    _fill_species_layer!(cache, n_dict_first, 1)

    if N > 1
        if threaded && Threads.nthreads() > 1
            Threads.@threads for i in 2:N
                _solve_layer_chemistry!(cache, αs, i, wls, Ts, nds, partition_funcs, ne_warn_thresh)
            end
        else
            @inbounds for i in 2:N
                _solve_layer_chemistry!(cache, αs, i, wls, Ts, nds, partition_funcs, ne_warn_thresh)
            end
        end
    end

    vmic = zero(T)
    ne_view     = view(cache.ne_solved, 1:N)
    nds_by_spec = Dict(k => view(v, 1:N) for (k, v) in cache.nds_by_species)
    α_cntm_view = view(cache.α_cntm,    1:N)
    Korg.line_absorption!(αs, linelist, wls, Ts, ne_view,
                          nds_by_spec, partition_funcs,
                          vmic, α_cntm_view; cutoff_threshold=cutoff_threshold)

    # hydrogen (Balmer/Brackett) lines — Korg treats these separately from the linelist
    if hydrogen_lines
        _add_hydrogen_line_absorption!(αs, wls, Ts, ne_view,
                                       nds_by_spec[_HI_SPECIES], nds_by_spec[_HeI_SPECIES],
                                       partition_funcs, hydrogen_line_window_size_Å; use_MHD=use_MHD)
    end

    # Persist solved n_e profile for warm-starting the next column.
    nes .= ne_view
    view(cache.warm_ne, 1:N) .= ne_view
    cache.has_warm_ne = true
    return nothing
end

"""
    compute_alpha!(αs, wls, linelist, atm, A_X, cache; kwargs...)
    compute_alpha!(αs, wls, linelist, zs, Ts, nds, nes, A_X, cache; kwargs...)

Cache-accelerated overloads of `FormationTemps.compute_alpha!`.

These overloads are additive: existing FormationTemps methods remain unchanged.
"""
function compute_alpha!(αs::AA{T, 2}, wls::Korg.Wavelengths, linelist,
                        atm::Atmosphere{T}, A_X::AA{T, 1}, cache::AlphaCache{T};
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1,
                        cutoff_threshold=3e-4,
                        threaded::Bool=true,
                        refresh_abundances::Bool=false,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing) where {T<:AF}
    refresh_abundances && set_abundances!(cache, A_X)
    _compute_alpha_cached!(αs, wls, linelist, atm.Ts, atm.nd, atm.nₑ, cache;
                           partition_funcs=partition_funcs,
                           ne_warn_thresh=ne_warn_thresh,
                           cutoff_threshold=cutoff_threshold, threaded=threaded,
                           hydrogen_lines=hydrogen_lines,
                           hydrogen_line_window_size_Å=hydrogen_line_window_size_Å,
                           use_MHD=use_MHD)
    return nothing
end

function compute_alpha!(αs::AA{T, 2}, wls::Korg.Wavelengths, linelist,
                        zs, Ts::AA{T, 1}, nds::AA{T, 1}, nes::AA{T, 1},
                        A_X::AA{T, 1}, cache::AlphaCache{T};
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1,
                        cutoff_threshold=3e-4,
                        threaded::Bool=true,
                        refresh_abundances::Bool=false,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing) where {T<:AF}
    refresh_abundances && set_abundances!(cache, A_X)
    _compute_alpha_cached!(αs, wls, linelist, Ts, nds, nes, cache;
                           partition_funcs=partition_funcs,
                           ne_warn_thresh=ne_warn_thresh,
                           cutoff_threshold=cutoff_threshold, threaded=threaded,
                           hydrogen_lines=hydrogen_lines,
                           hydrogen_line_window_size_Å=hydrogen_line_window_size_Å,
                           use_MHD=use_MHD)
    return nothing
end

function compute_alpha!(αs::AA{T, 2}, αs_cont::AA{T, 2}, wls::Korg.Wavelengths, linelist,
                        atm::Atmosphere{T}, A_X::AA{T, 1}, cache::AlphaCache{T};
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1,
                        cutoff_threshold=3e-4,
                        threaded::Bool=true,
                        refresh_abundances::Bool=false,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing) where {T<:AF}
    compute_alpha!(αs, wls, linelist, atm, A_X, cache;
                   partition_funcs=partition_funcs,
                   ne_warn_thresh=ne_warn_thresh,
                   cutoff_threshold=cutoff_threshold,
                   threaded=threaded, refresh_abundances=refresh_abundances,
                   hydrogen_lines=hydrogen_lines,
                   hydrogen_line_window_size_Å=hydrogen_line_window_size_Å,
                   use_MHD=use_MHD)
    _fill_continuum_from_cache!(αs_cont, cache, wls)
    return nothing
end

function compute_alpha!(αs::AA{T, 2}, αs_cont::AA{T, 2}, wls::Korg.Wavelengths, linelist,
                        zs, Ts::AA{T, 1}, nds::AA{T, 1}, nes::AA{T, 1},
                        A_X::AA{T, 1}, cache::AlphaCache{T};
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1, cutoff_threshold=3e-4,
                        threaded::Bool=true, refresh_abundances::Bool=false,
                        hydrogen_lines::Bool=true,
                        hydrogen_line_window_size_Å::Float64=150.0,
                        use_MHD::Union{Nothing,Bool}=nothing) where {T<:AF}
    compute_alpha!(αs, wls, linelist, zs, Ts, nds, nes, A_X, cache;
                   partition_funcs=partition_funcs, ne_warn_thresh=ne_warn_thresh,
                   cutoff_threshold=cutoff_threshold, threaded=threaded,
                   refresh_abundances=refresh_abundances,
                   hydrogen_lines=hydrogen_lines,
                   hydrogen_line_window_size_Å=hydrogen_line_window_size_Å,
                   use_MHD=use_MHD)
    _fill_continuum_from_cache!(αs_cont, cache, wls)
    return nothing
end
