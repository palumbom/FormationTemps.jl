"""
Adapted from Korg.jl -> line_absorption!()
"""
function compute_alpha!(αs, wls::Korg.Wavelengths, linelist, atm, A_X; partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1)
    compute_alpha!(αs, wls, linelist, atm.zs, atm.Ts, atm.nd, atm.nₑ, 
                   A_X; partition_funcs=Korg.default_partition_funcs,
                   ne_warn_thresh=ne_warn_thresh)
    return nothing
end

function compute_alpha!(αs, wls::Korg.Wavelengths, linelist, zs, Ts, nds, nes, A_X; 
                        partition_funcs=Korg.default_partition_funcs, ne_warn_thresh=0.1)
    # deal with abundances 
    abs_abundances = @. 10^(A_X - 12) # n(X) / n_tot
    abs_abundances ./= sum(abs_abundances) #normalize so that sum(n(X)/n_tot) = 1

    # work in cm
    cntm_step = 1e-8
    line_buffer = 10.0 * 1e-8

    # wavelengths at which to calculate the continuum
    cntm_wls = range(first(wls) - line_buffer, last(wls) + line_buffer, step=cntm_step)
    cntm_wls = Korg.Wavelengths(cntm_wls)

    # allocate for chemical equilibrium solver
    N = length(zs)
    triples = Vector{Tuple{Float64, Dict, typeof(Korg.linear_interpolation(cntm_wls, zeros(length(cntm_wls))))}}(undef, N)

    # loop over layers and do chemical equilibrium
    # Threads.@threads for i in 1:N
    for i in 1:N
        # index the layers
        temp = Ts[i]
        nd = nds[i]
        ne = nes[i]

        # compute equilibrium
        nₑ, n_dict = Korg.chemical_equilibrium(temp, nd, ne, abs_abundances, 
                                               Korg.ionization_energies, 
                                               partition_funcs, 
                                               Korg.default_log_equilibrium_constants,
                                               electron_number_density_warn_threshold=ne_warn_thresh)

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
    vmic = 0.0
    Korg.line_absorption!(αs, linelist, wls, Ts, nₑs, nds, partition_funcs, 
                          vmic, α_cntm; cutoff_threshold=3e-4, verbose=false)

    return nothing
end