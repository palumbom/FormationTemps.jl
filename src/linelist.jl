function get_grass_linelist(;return_names::Bool=false)
    linelist = Korg.read_linelist(joinpath(datdir, "Sun_VALD.lin"))
    linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
    wls = [l.wl for l in linelist]
    specs = [replace(string(l.species), " "=>"") for l in linelist]

    # get GRASS lines
    lp = GRASS.LineProperties()
    wls_grass = lp.λrest
    species_grass = lp.species
    names_grass = GRASS.get_name(lp)

    # loop over 
    idxs = []
    line_names = []
    for i in eachindex(wls_grass)
        idx = findfirst(isapprox.(wls * 1e8, wls_grass[i], atol=1e-1))

        if !isnothing(idx) && species_grass[i] == specs[idx]
            push!(idxs, idx)
            push!(line_names, names_grass[i])
        end
    end
    linelist = linelist[idxs]

    # fix FeI 6173 manually
    wl = 6.17333e-5
    log_gf = -2.880
    species = Korg.Species("Fe I")
    E_lower = 2.223
    factor = 1.0
    gamma_rad = factor * exp10(8.31)
    gamma_stark = factor * exp10(-6.16)
    gamma_vdw = log10(factor * exp10(-7.69))

    # replace 
    idx = findfirst(isapprox.([l.wl * 1e8 for l in linelist], 6173.333, atol=1e-1))
    linelist[idx] = Korg.Line(wl, log_gf, species, E_lower, gamma_rad, gamma_stark, gamma_vdw)
    if return_names
        return linelist, line_names
    else
        return linelist
    end
end