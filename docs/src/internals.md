# Public Functions 

```@meta
CurrentModule = FormationTemps
```

The public functions exported by FormationTemps.jl are documented on this page. The high-level convenience functions  should meet the needs of most users. However, some potentially useful, slightly lower-level methods are exposed by FormationTemps.jl and documented below. A full index of all methods defined by FormationTemps.jl is available in the [Full Index](@ref "Full Index").

## High-level Convenience Functions

FormationTemps.jl provides a few high-level convenience wrappers to produce flux spectra and formation temperatures.

```@docs
calc_formation_temp
calc_formation_temp_chunked
StellarProps
FormTempResult
```

## Atmosphere Types

```FormTempResult``` includes an ```atmosphere``` field which contains the model atmosphere used in the spectrum modeling. 

```@docs
Atmosphere
get_τs
get_zs
get_Ts
```


## Empirical Relations

A few works have reported empirical relationships between fundamental stellar parameters and the measured micro- and macroturbulent velocities. FormationTemps.jl implements these for cases where micro- and macroturbulence are not available. 

```@docs
vmac_fit
vmic_fit
```

## Resolution Degradation + LSF Modeling

Spectra are synthesized at infinite spectral resolving power (but finite sampling). Since instruments imprint some LSF on spectra, model spectra are generally convolved with model LSFs. Functions for performing these operations are documented here. 

```@docs
convolve_instrument_gauss
rebin_spectrum
```

## Convolutions & Kernels

Convolution operations are often used in the modeling of spectra, even when explicit disk integration is performed (e.g., for microturbulent broadening). FormationTemps.jl exposes these convolution methods.

```@docs
convolve_gray_rotation
convolve_hirano_rotmacro
convolve_iso_rt_macro
convolve_rt_macro
```

The broadening kernels (which are evaluated internally in the above convolution functions) can also be directly calculated.


```@docs
gray_rot_kernel
hirano_rotmacro_ft_kernel 
gray_iso_rt_macro_kernel
rt_macro_kernel
```

## Absorption Cache

`AlphaCache` provides a reusable cache that accelerates repeated `compute_alpha!` calls
by warm-starting the electron density solver and reusing continuum buffers across calls.

```@docs
AlphaCache
```

## Miscellaneous Utilities

```@docs
round_to_power
elav
searchsortednearest
```
