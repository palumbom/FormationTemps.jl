# FormationTemps.jl
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://michaelpalumbo.me/FormationTemps.jl/dev/)
[![Build Status](https://github.com/palumbom/FormationTemps.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/palumbom/FormationTemps.jl/actions/workflows/CI.yml?query=branch%3Amain)

FormationTemperatures.jl wraps [Korg.jl](https://github.com/ajwheeler/Korg.jl) to produce model spectra and formation temperatures given fundamental stellar parameters and a linelist as input.

## Installation

GRASS is written entirely in Julia and requires Julia v1.12 or greater. Installation instructions for Julia are available from [julialang.org](https://julialang.org/downloads/).

To install FormationTemps.jl:

```bash
cd DIRECTORY
git clone git@github.com:palumbom/FormationTemps.jl.git
```

then from the Julia REPL:

```julia
using Pkg
Pkg.add(path="DIRECTORY")
using FormationTemps
```

### Calling FormationTemps.jl from Python

TBD

## Citation
[![arXiv](https://img.shields.io/badge/arXiv-2512.09861-b31b1b.svg)](https://arxiv.org/abs/2512.09861)

If you use FormationTemps.jl in your research, please cite the relevant [software release]() and [paper](https://ui.adsabs.harvard.edu/abs/2025arXiv251209861P/abstract).

## Author & Contact 
[![GitHub followers](https://img.shields.io/github/followers/palumbom?label=Follow&style=social)](https://github.com/palumbom)

This repo is maintained by [Michael Palumbo](https://michaelpalumbo.me). You may may contact him via his email - [mpalumbo@flatironinstitute.org](mailto:mpalumbo@flatironinstitute.org)