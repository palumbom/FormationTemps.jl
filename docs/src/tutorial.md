# Basic Tutorial

In the simplest use case, a model flux spectrum and associated formation temperatures can be calculated given stellar parameters as input. 

```julia
using Korg
using PyPlot
using FormationTemps; FT = FormationTemps

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]

# set stellar parameters
Teff = 5777.0
logg = 4.44
A_X = Korg.asplund_2020_solar_abundances
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0   # radial-tangential macroturbulent broadening 
ξ = 850.0       # microturbulent broadenign

# create StellarProps composite type to hold everything 
star_props = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# get the flux + formation temperature spectra
form_temp_result = FT.calc_formation_temp(star_props, linelist; Δλ=0.01)

# parse the result
wavs = form_temp_result.wavs
flux = form_temp_result.flux
temp = form_temp_result.form_temps

# plot the result
fig, ax1 = plt.subplots()
ax1.plot(wavs, temp, c="k")
ax1.set_xlabel("Vacuum Wavelength [Å]")
ax1.set_ylabel("Formation Temperature [K]")
plt.show()
```
![formation_temps](examples/temp_example_jl.png)