# Basic Tutorial

In the simplest use case, a model flux spectrum and associated formation temperatures can be calculated given stellar parameters as input. For further details on specifying linelists, abundances, and other details of the model atmosphere, please see the [Korg.jl documentation](https://ajwheeler.github.io/Korg.jl/stable/generated/tutorials/Basics/).

```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "simple.jl"), String)
Markdown.parse("```julia\n" * code * "\n```")
```
![formation_temps](static/temp_example_jl.png)

The high-level convenience function ```calc_formation_temp``` provides a few optional arguments discussed below. 

## Convolution vs. Integration

```convolve=false```

At low spectral resolving power, convolutions can be used to approximate the effects of macroturbulent and rotational broadening. As shown in Section 2.1.4 of the [paper presenting FormationTemps.jl](https://ui.adsabs.harvard.edu/abs/2025arXiv251209861P/abstract), this approximation can fail at higher resolution. By default, ```calc_formation_temp``` performs an explicit disk integration to evaluate model spectra and formation temperatures. Though more accurate, this approach is slower. To use the convolution approximation, ```convolve=true``` can be supplied to```calc_formation_temp```.

## GPU Usage

```use_gpu=true```

By default, ```calc_formation_temp``` will use an NVIDIA GPU to accelerate the computation of spectra, if one is present and configured. The [CUDA.jl documentation](https://cuda.juliagpu.org/stable/) provides installation instructions, though it is fortunately fairly autonomous. 

Several steps of the spectral synthesis are parallelized, including:
* calculation of the stellar grid (if ```convolve=false```)
* the convolution step in the microturbulent broadening
* the solving of the radiative transport equation:
  * solving for the monochromatic optical depth
  * calculating the monochromatic intensity/flux contribution functions
  * evaluation of the monochromatic intensity/flux from the contribution function
