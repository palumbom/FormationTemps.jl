"""
    vmac_fit(teff, logg)
    vmac_fit(teff)

Empirical macroturbulent velocity fits.

`vmac_fit(teff, logg)` uses the Doyle et al. (2014) relation with `teff` in K and
`logg` in cgs. `vmac_fit(teff)` uses the Bruntt et al. (2010) relation with `teff`
in K. Both return the macroturbulent velocity in m/s.
"""
# from Doyle et al. 2014
vmac_fit(teff, logg) = 1.0e3 * (3.21 + 2.33e-3 * (teff - 5777) + 2e-6 * (teff - 5777)^2.0 - 2.0 * (logg - 4.44))

# from Bruntt et al. 2010
vmac_fit(teff) = 1.0e3 * (2.26 + 2.90e-3 * (teff - 5777) + 5.86e-7 * (teff - 5777)^2.0)

"""
    vmic_fit(teff)

Empirical microturbulent velocity fit from Bruntt et al. (2010).

Input `teff` is in K; returns microturbulent velocity in m/s.
"""
vmic_fit(teff) = 1.0e3 * (1.01 + 4.56e-4 * (teff - 5777) + 2.75e-7 * (teff - 5777)^2.0)
