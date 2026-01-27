# <img src="docs/src/assets/logo.png" height="48">  FormationTemps.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://palumbom.github.io/FormationTemps.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://palumbom.github.io/FormationTemps.jl/dev/)
[![Build Status](https://github.com/palumbom/FormationTemps.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/palumbom/FormationTemps.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2512.09861-b31b1b.svg)](https://arxiv.org/abs/2512.09861)

FormationTemps.jl wraps [Korg.jl](https://github.com/ajwheeler/Korg.jl) to produce model spectra and formation temperatures given fundamental stellar parameters and a linelist as input. We encourage users to read [the paper](https://ui.adsabs.harvard.edu/abs/2025arXiv251209861P/abstract) that presents FormationTemps.jl. The scripts used to generate the plots and other quantitative results presented therein can be found in the [`scripts/` directory](https://github.com/palumbom/FormationTemps.jl/tree/main/scripts) of the GitHub repo. 

## Installation

FormationTemps.jl is written entirely in Julia and requires Julia v1.12 or greater. Installation instructions for Julia are available from [julialang.org](https://julialang.org/downloads/).

To install FormationTemps.jl:

```bash
git clone git@github.com:palumbom/FormationTemps.jl.git
cd FormationTemps.jl
julia
```

Then from the Julia REPL:

```julia
using Pkg
Pkg.add(path=".")
using FormationTemps
```

### Basic Julia Example

To compute a basic formation temperature spectrum:

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
ξ = 850.0       # microturbulent broadening

# create StellarProps composite type to hold everything 
star_props = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, 
                          vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

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
![formation_temps](./docs/src/static/temp_example_jl.png)

More detail on the above example can be found in the [Basic Tutorial](https://michaelpalumbo.me/FormationTemps.jl/stable/tutorial/) and the [high-level API documentation](https://michaelpalumbo.me/FormationTemps.jl/stable/internals/).

### Calling FormationTemps.jl from Python

> [!WARNING] 
> Calling FormationTemps.jl from Python is currently somewhat fragile and a work in progress. 

FormationTemps.jl can be called from Python. The instructions can be found in the [Python Tutorial](https://michaelpalumbo.me/FormationTemps.jl/stable/pycall/). 

## Caveats

> [!CAUTION] 
> Users should be aware of the technical and "philosophical" discussion on formation temperatures in Sections 4.2 and 4.3 of [the paper](https://arxiv.org/abs/2512.09861) presenting FormationTemps.jl. 

In brief:

* Formation temperatures are *modeled* and not measured quantities
* The definition/concept of a formation temperature can belie some realities of radiative transfer (see the contribution function comparison in the [relevant tutorial](https://michaelpalumbo.me/FormationTemps.jl/stable/cont_func/#Formation-temperatures-can-lie-to-you!))
* Korg.jl only assumes LTE, and the MARCS model atmospheres used by default do not have chromospheres
* The model atmospheres are 1D, and do not handle the effects of convection (limb shift, line asymmetry, etc.) or magnetism

## Citation
[![DOI](https://zenodo.org/badge/1034682708.svg)](https://doi.org/10.5281/zenodo.18343827)
[![arXiv](https://img.shields.io/badge/arXiv-2512.09861-b31b1b.svg)](https://arxiv.org/abs/2512.09861)

If you use FormationTemps.jl in your research, please cite the relevant [software release](https://zenodo.org/records/18343828) and [paper](https://ui.adsabs.harvard.edu/abs/2025arXiv251209861P/abstract). The [`cffconvert` tool](https://github.com/citation-file-format/cffconvert) can be used to generate a bibtex entry from the included [CITATION.cff](https://github.com/palumbom/FormationTemps.jl/blob/main/CITATION.cff) (or just use the "cite this repository" button on the GitHub sidebar).

## Author & Contact 
[![GitHub followers](https://img.shields.io/github/followers/palumbom?label=Follow&style=social)](https://github.com/palumbom)

This repo is maintained by [Michael Palumbo](https://michaelpalumbo.me). You may may contact him via his email - [mpalumbo@flatironinstitute.org](mailto:mpalumbo@flatironinstitute.org)