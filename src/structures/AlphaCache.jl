const _HIII_SPECIES = Korg.Species("H III")

"""
    AlphaCache(wls, A_X, Nlayers; cntm_step=1e-8, line_buffer=10e-8)
    AlphaCache(wls, A_X, atm; cntm_step=1e-8, line_buffer=10e-8)

Reusable cache for accelerated `compute_alpha!` calls.

This cache implements:
- warm-started electron density guesses across columns,
- precomputed abundance normalization and continuum wavelength grid,
- reusable buffers for chemistry and line-absorption assembly.
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
    n_dicts::Vector{Dict{Korg.Species, T}}

    species_keys::Vector{Korg.Species}
    nds_by_species::Dict{Korg.Species, Vector{T}}
    α_cntm::Vector{TI}
end

function _normalize_abs_abundances(A_X::AA{T, 1}) where {T<:AF}
    abs_abundances = @. 10^(A_X - 12)
    abs_abundances ./= sum(abs_abundances)
    return abs_abundances
end

function AlphaCache(wls::Korg.Wavelengths, A_X::AA{T, 1}, Nlayers::Integer;
                    cntm_step::Real=1e-8, line_buffer::Real=10e-8) where {T<:AF}
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
        [Dict{Korg.Species, T}() for _ in 1:Int(Nlayers)],
        Korg.Species[],
        Dict{Korg.Species, Vector{T}}(),
        [dummy_itp for _ in 1:Int(Nlayers)],
    )
end

function AlphaCache(wls::Korg.Wavelengths, A_X::AA{T, 1}, atm::Atmosphere{T};
                    cntm_step::Real=1e-8, line_buffer::Real=10e-8) where {T<:AF}
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
    N == cache.Nlayers || throw(ArgumentError("AlphaCache layer count $(cache.Nlayers) != input layer count $N"))
    if first(wls) < first(cache.cntm_wls) || last(wls) > last(cache.cntm_wls)
        throw(ArgumentError("wavelength grid lies outside cached continuum grid; rebuild AlphaCache for this wls"))
    end
    return nothing
end

function _seed_ne_guesses!(cache::AlphaCache{T}, nes::AA{T, 1}) where {T<:AF}
    tiny = eps(T)
    if cache.has_warm_ne
        @inbounds for i in eachindex(cache.ne_solved, cache.warm_ne)
            cache.ne_solved[i] = max(cache.warm_ne[i], tiny)
        end
    else
        cache.ne_solved[1] = max(nes[1], tiny)
        @inbounds for i in 2:length(cache.ne_solved)
            # Depth-coupled fallback guess: use the previous-layer value.
            cache.ne_solved[i] = max(nes[i - 1], tiny)
        end
    end
    return nothing
end

function _build_species_density_view!(cache::AlphaCache{T}) where {T<:AF}
    empty!(cache.species_keys)
    first_layer = cache.n_dicts[1]
    for spec in keys(first_layer)
        if spec != _HIII_SPECIES
            push!(cache.species_keys, spec)
            vec = get!(cache.nds_by_species, spec, zeros(T, cache.Nlayers))
            length(vec) == cache.Nlayers || (cache.nds_by_species[spec] = zeros(T, cache.Nlayers))
        end
    end

    # Remove stale species not present in the current chemistry solution.
    keep = Set(cache.species_keys)
    for spec in collect(keys(cache.nds_by_species))
        spec in keep || delete!(cache.nds_by_species, spec)
    end

    for spec in cache.species_keys
        vec = cache.nds_by_species[spec]
        @inbounds for i in 1:cache.Nlayers
            vec[i] = cache.n_dicts[i][spec]
        end
    end
    return nothing
end

function _solve_layer_chemistry!(cache::AlphaCache{T}, αs::AA{T, 2}, i::Int, wls::Korg.Wavelengths,
                                 Ts::AA{T, 1}, nds::AA{T, 1}, partition_funcs, ne_warn_thresh) where {T<:AF}
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
    cache.n_dicts[i] = n_dict
    cache.α_cntm[i] = α_cntm_layer
    return nothing
end

function _compute_alpha_cached!(αs::AA{T, 2}, wls::Korg.Wavelengths, linelist, Ts::AA{T, 1},
                                nds::AA{T, 1}, nes::AA{T, 1}, cache::AlphaCache{T};
                                partition_funcs=Korg.default_partition_funcs,
                                ne_warn_thresh=0.1,
                                cutoff_threshold=3e-4,
                                threaded::Bool=true) where {T<:AF}
    N = length(Ts)
    _validate_alpha_cache(cache, wls, N)
    _seed_ne_guesses!(cache, nes)

    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:N
            _solve_layer_chemistry!(cache, αs, i, wls, Ts, nds, partition_funcs, ne_warn_thresh)
        end
    else
        @inbounds for i in 1:N
            _solve_layer_chemistry!(cache, αs, i, wls, Ts, nds, partition_funcs, ne_warn_thresh)
        end
    end

    _build_species_density_view!(cache)
    vmic = zero(T)
    Korg.line_absorption!(αs, linelist, wls, Ts, cache.ne_solved, 
                          cache.nds_by_species, partition_funcs, 
                          vmic, cache.α_cntm; cutoff_threshold=cutoff_threshold)

    # Persist solved n_e profile for warm-starting the next column.
    nes .= cache.ne_solved
    cache.warm_ne .= cache.ne_solved
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
                        refresh_abundances::Bool=false) where {T<:AF}
    refresh_abundances && set_abundances!(cache, A_X)
    _compute_alpha_cached!(αs, wls, linelist, atm.Ts, atm.nd, atm.nₑ, cache; 
                           partition_funcs=partition_funcs, 
                           ne_warn_thresh=ne_warn_thresh, 
                           cutoff_threshold=cutoff_threshold, threaded=threaded)
    return nothing
end

function compute_alpha!(αs::AA{T, 2}, wls::Korg.Wavelengths, linelist,
                        zs, Ts::AA{T, 1}, nds::AA{T, 1}, nes::AA{T, 1},
                        A_X::AA{T, 1}, cache::AlphaCache{T};
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1,
                        cutoff_threshold=3e-4,
                        threaded::Bool=true,
                        refresh_abundances::Bool=false) where {T<:AF}
    refresh_abundances && set_abundances!(cache, A_X)
    _compute_alpha_cached!(αs, wls, linelist, Ts, nds, nes, cache; 
                           partition_funcs=partition_funcs, 
                           ne_warn_thresh=ne_warn_thresh, 
                           cutoff_threshold=cutoff_threshold, threaded=threaded)
    return nothing
end

function compute_alpha!(αs::AA{T, 2}, αs_cont::AA{T, 2}, wls::Korg.Wavelengths, linelist,
                        atm::Atmosphere{T}, A_X::AA{T, 1}, cache::AlphaCache{T};
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1,
                        cutoff_threshold=3e-4,
                        threaded::Bool=true,
                        refresh_abundances::Bool=false) where {T<:AF}
    compute_alpha!(αs, wls, linelist, atm, A_X, cache; 
                   partition_funcs=partition_funcs, 
                   ne_warn_thresh=ne_warn_thresh, 
                   cutoff_threshold=cutoff_threshold, 
                   threaded=threaded, refresh_abundances=refresh_abundances)
    αs_cont .= αs
    return nothing
end

function compute_alpha!(αs::AA{T, 2}, αs_cont::AA{T, 2}, wls::Korg.Wavelengths, linelist,
                        zs, Ts::AA{T, 1}, nds::AA{T, 1}, nes::AA{T, 1},
                        A_X::AA{T, 1}, cache::AlphaCache{T};
                        partition_funcs=Korg.default_partition_funcs,
                        ne_warn_thresh=0.1, cutoff_threshold=3e-4,
                        threaded::Bool=true, refresh_abundances::Bool=false) where {T<:AF}
    compute_alpha!(αs, wls, linelist, zs, Ts, nds, nes, A_X, cache; 
                   partition_funcs=partition_funcs, ne_warn_thresh=ne_warn_thresh, 
                   cutoff_threshold=cutoff_threshold, threaded=threaded, 
                   refresh_abundances=refresh_abundances)
    αs_cont .= αs
    return nothing
end