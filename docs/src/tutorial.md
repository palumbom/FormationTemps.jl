# Basic Tutorial

In the simplest use case, a model flux spectrum and associated formation temperatures can be calculated given stellar parameters as input. For further details on specifying linelists, abundances, and other details of the model atmosphere, please see the [Korg.jl documentation](https://ajwheeler.github.io/Korg.jl/stable/generated/tutorials/Basics/).

```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "simple.jl"), String)
Markdown.parse("```julia\n" * code * "\n```")
```
![formation_temps](static/temp_example_jl.png)

The high-level convenience function ```calc_formation_temp``` provides a few optional arguments discussed below.

## StellarProps

`v_micro` (microturbulent velocity ξ) accepts either a scalar or a vector of length `Natm` for per-layer microturbulence. All other velocity parameters (`vsini`, `v_macro`) are scalars. `vsini` is the projected equatorial rotational velocity.

Rotation geometry is set by `istar` (inclination in degrees; `90` = equator-on) and the differential-rotation coefficients `α₂`, `α₄` in the normalized rate law `Ω(ϕ)/Ω_eq = 1 - α₂·sin²ϕ - α₄·sin⁴ϕ` (default `0`, i.e. solid-body). For rigid rotation the broadening depends only on `vsini`, so `istar` has no effect; with differential rotation (`α ≠ 0`) `istar` selects which latitude bands are visible and therefore matters. See [Integration Methods](@ref).

## Convolution vs. Integration

At low spectral resolving power, convolutions can be used to approximate the effects of macroturbulent and rotational broadening. As shown in Section 2.1.4 of the [paper presenting FormationTemps.jl](https://ui.adsabs.harvard.edu/abs/2025arXiv251209861P/abstract), this approximation can fail at higher resolution. By default, ```calc_formation_temp``` performs an explicit disk integration (```method=:disk```) to evaluate model spectra and formation temperatures. Though more accurate, this approach is slower. To use the convolution approximation, pass ```method=:hirano``` (equivalently the legacy ```convolve=true```) to ```calc_formation_temp```. A faster middle ground, ```method=:quadrature```, keeps most of the disk-integration accuracy; see [Integration Methods](@ref). A plot comparing the convolution and integration fluxes and temperatures is shown below for a solar-like model star. As shown in the [paper](https://ui.adsabs.harvard.edu/abs/2025arXiv251209861P/abstract), the error incurred by the convolution approximation grows with $v \sin i$.

![convolution_vs_integration](static/convolution_vs_integration.png)

## Running in parallel

Disk integration (`method=:disk`, the default) is the most computationally intensive mode and benefits from parallelization on both CPU and GPU.

### CPU multithreading

The CPU disk integration path distributes tiles across Julia threads. Launch Julia with multiple threads to benefit:

```bash
julia -t auto           # use all available cores
julia -t 8              # use 8 threads
```

No code changes are required — `calc_formation_temp` detects the available threads automatically.

!!! warning "Python interop"
    CPU multithreading is not compatible with calling FormationTemps.jl from Python via juliacall/PythonCall. Set `JULIA_NUM_THREADS=1` when calling from Python. Use the GPU path for parallelism in that case.

### GPU acceleration

By default, ```calc_formation_temp``` will use an NVIDIA GPU to accelerate the computation of spectra, if one is present and configured. The [CUDA.jl documentation](https://cuda.juliagpu.org/stable/) provides installation instructions. Pass `use_gpu=false` to force the CPU path. Pass `gpu_precision=Float32` to run GPU computations at single precision, which roughly halves GPU memory and can improve throughput on consumer GPUs. Absorption coefficients are always computed at Float64 (a Korg requirement) and converted before GPU upload.

For details on what gets accelerated, memory layout, benchmarks, and CPU/GPU numerical differences, see the [Parallelization](parallelization.md) guide.
