using Korg
using FormationTemps; FT = FormationTemps

# set some stellar parameters
logg = 4.44
Teff = 5777
A_X = deepcopy(Korg.asplund_2020_solar_abundances)
vmic = 1200.0 # in m/s

# mu positions to evaluate
μs = range(0.1, 1.0, step=0.1)

# get the linelist 
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))

form_temps_intensity, form_temps_flux = calc_formation_temp(μs, linelist, Teff, logg, A_X, vmic)