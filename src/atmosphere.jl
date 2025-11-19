function get_marcs_atm(Teff::T, logg::T, A_X::AA{T,1}; n_layers::Int=56) where T<:AF
    # get the model atmosphere
    marcs_atm = Korg.interpolate_marcs(Teff, logg, A_X)
    τ_500 = Korg.get_tau_refs(marcs_atm)
    zs = Korg.get_zs(marcs_atm)
    Ts = Korg.get_temps(marcs_atm)
    ne = Korg.get_electron_number_densities(marcs_atm)
    nd = Korg.get_number_densities(marcs_atm)

    # interpolate in zs 
    itp_τs = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(τ_500))
    itp_Ts = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(Ts))
    itp_ne = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(ne))
    itp_nd = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(nd))

    zs_new = range(last(zs), first(zs), length=n_layers)
    τs_new = reverse(itp_τs.(zs_new))
    Ts_new = reverse(itp_Ts.(zs_new))
    ne_new = reverse(itp_ne.(zs_new))
    nd_new = reverse(itp_nd.(zs_new))
    zs_new = reverse(collect(zs_new))

    ls = Array{Korg.PlanarAtmosphereLayer{Float64, Float64, Float64, Float64, Float64}}(undef, length(zs_new))
    for i in eachindex(zs_new)
        ls[i] = Korg.PlanarAtmosphereLayer(τs_new[i], zs_new[i], Ts_new[i], ne_new[i], nd_new[i])
    end
    return Korg.PlanarAtmosphere(ls, 5000.0 / 1e8)
end