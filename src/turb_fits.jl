# from Doyle et al. 2014
vmac_fit(teff, logg) = 3.21 + 2.33e-3 * (teff - 5777) + 2e-6 * (teff - 5777)^2.0 - 2.0 * (logg - 4.44)

# from Bruntt et al. 2010
vmac_fit(teff) = 2.26 + 2.90e-3 * (teff - 5777) + 5.86e-7 * (teff - 5777)^2.0
vmic_fit(teff) = 1.01 + 4.56e-4 * (teff - 5777) + 2.75e-7 * (teff - 5777)^2.0