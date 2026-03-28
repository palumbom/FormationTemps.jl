# Parallelization

```@meta
CurrentModule = FormationTemps
```

FormationTemps.jl parallelizes the disk integration pipeline in two complementary ways: **CPU multithreading** across stellar surface tiles, and **GPU acceleration** via [CUDA.jl](https://cuda.juliagpu.org/stable/). Both target the same bottleneck — the per-tile radiative transfer loop that dominates wall-clock time when `convolve=false`.

## CPU Multithreading

### What gets parallelized

When `use_gpu=false` and `convolve=false`, `calc_formation_temp` iterates over every visible tile on the stellar surface (typically thousands for `Nϕ = 128`). Each tile independently computes microturbulent broadening, optical depth, the intensity contribution function, and macroturbulent broadening. These tiles are distributed across Julia threads using `Threads.@threads` with `:static` scheduling.

Each thread receives its own [`CPUTileWorkspace`](@ref) containing pre-allocated working arrays (`τs`, contribution functions, and per-thread accumulators). After the loop, the per-thread accumulators are reduced to produce the final result.

### Setup

Start Julia with multiple threads:

```bash
julia -t auto           # use all available cores
julia -t 8              # use 8 threads
```

Or set the environment variable before launching Julia:

```bash
export JULIA_NUM_THREADS=8
```

You can verify the thread count from within Julia:

```julia
Threads.nthreads()
```

### FFTW considerations

FFTW plan creation is not thread-safe. Before entering the threaded tile loop, `calc_formation_temp` pre-warms the FFTW plan cache on the main thread and sets `FFTW.set_num_threads(1)` to disable FFTW's internal threading (which would compete with tile-level parallelism). The previous FFTW thread count is restored after the loop.

### Limitations

!!! warning "Python interop"
    CPU multithreading is **not compatible** with calling FormationTemps.jl from Python via juliacall/PythonCall. Julia's multi-threaded garbage collector can conflict with PythonCall's runtime bridge, causing hard crashes (SIGBUS/segfault). When calling from Python, `JULIA_NUM_THREADS` must be set to `1`. See the [Python Tutorial](@ref "Python Tutorial") for details.

    If you need both Python interop and parallelism, use the GPU path (`use_gpu=True`) or run the computation in pure Julia and load the results in Python.

## GPU Acceleration

GPU acceleration is enabled by default when a compatible NVIDIA device is detected (`CUDA.functional() == true`); pass `use_gpu=false` to force the CPU path.

### What gets accelerated

The disk integration pipeline repeats the full radiative transfer calculation for every visible tile on the stellar surface — typically thousands of tiles for `Nϕ = 128`. Each tile requires:

1. **Microturbulent broadening** — FFT-based Gaussian convolution of the absorption coefficient matrix along the wavelength axis, per atmosphere layer.
2. **Optical depth integration** — cumulative integration of absorption coefficients through the atmosphere to build `τ(λ)` at each layer.
3. **Contribution function** — Planck-weighted intensity contribution per layer, combined with `dτ`.
4. **Macroturbulent broadening** — radial-tangential convolution of the contribution function.

On the GPU, steps 1--3 are fused into a single kernel dispatch ([`calc_intensity_quantities`](@ref)), and step 4 uses a separate FFT-based convolution kernel. Pre-allocated memory structs ([`GPUMemory`](@ref), [`ConvolutionMemory`](@ref)) eliminate per-tile allocations.

#### Additional GPU optimizations

- **Signal caching**: [`ConvolutionMemory`](@ref) tracks whether the input signal FFT has already been computed (`signal_cached` flag). During disk integration, the absorption coefficients are identical across tiles — only the Doppler shift kernel changes — so the signal FFT is computed once and reused.
- **Macroturbulence kernel caching**: the radial-tangential kernel depends only on `μ` (the cosine of the viewing angle), not on the tile's rotational velocity. Since many tiles share the same `μ`, kernels are precomputed for each unique `μ` value and looked up during the tile loop.
- **FFT-friendly padding**: convolution buffers are padded to lengths with small prime factors (2, 3, 5, 7) for efficient FFTs via CUFFT.
- **Buffer reuse**: per-tile contribution function results are written into pre-allocated GPU buffers via `copyto!`, avoiding repeated `CuArray` allocations during the tile loop.

### Setup

GPU support requires an NVIDIA GPU and a working CUDA toolkit. CUDA.jl handles driver detection and toolkit installation automatically in most cases:

```julia
using CUDA
CUDA.functional()  # should return true
CUDA.versioninfo()  # check device and toolkit details
```

If `CUDA.functional()` returns `false`, consult the [CUDA.jl troubleshooting guide](https://cuda.juliagpu.org/stable/installation/troubleshooting/). FormationTemps.jl will fall back to the CPU path transparently.

### Memory layout

The GPU path pre-allocates all working arrays at the start of a `calc_formation_temp` call:

| Struct | Contents | Lifetime |
|---|---|---|
| [`AtmosphereGPU`](@ref) | Temperature, number density, electron density, optical depth scale as `CuArrays` | One per call |
| [`GPUMemory`](@ref) | `αs`, `τs`, `cfunc`, `flux`, and Bezier work arrays (`tau_ds`, `tau_alphaC`) | One per call |
| [`ConvolutionMemory`](@ref) | Padded FFT buffers, CUFFT plans, cached signal/kernel transforms | One per convolution type (microturbulence, macroturbulence, continuum) |

This means GPU memory usage is determined at allocation time and remains constant throughout the tile loop. For a typical problem (`Natm ≈ 60`, `Nλ ≈ 600`, `Npad = 512`), total GPU memory usage is on the order of tens of MB.

## Benchmarks

The [`benchmarks/`](https://github.com/palumbom/FormationTemps.jl/tree/main/benchmarks) directory contains scripts that measure performance:

- [`benchmark_disk_integration.jl`](https://github.com/palumbom/FormationTemps.jl/blob/main/benchmarks/benchmark_disk_integration.jl) — per-tile step breakdown and end-to-end `calc_formation_temp` timing (CPU vs. GPU).
- [`benchmark_convolutions.jl`](https://github.com/palumbom/FormationTemps.jl/blob/main/benchmarks/benchmark_convolutions.jl) — individual convolution kernel comparisons.
- [`benchmark_threading.jl`](https://github.com/palumbom/FormationTemps.jl/blob/main/benchmarks/benchmark_threading.jl) — CPU thread-scaling for disk integration (spawns separate Julia processes for each thread count).

The benchmark results shown below were obtained on the following hardware:

| Component | Details |
|---|---|
| CPU | Intel Xeon w5-3435X (16 cores / 32 threads, 3.1 GHz) |
| GPU | NVIDIA RTX 6000 Ada Generation (48 GB VRAM) |

All scripts write CSV files to [`benchmarks/data/`](https://github.com/palumbom/FormationTemps.jl/tree/main/benchmarks/data) and PNG figures to [`docs/src/static/`](https://github.com/palumbom/FormationTemps.jl/tree/main/docs/src/static). Run them with:

```bash
julia --project=. benchmarks/benchmark_disk_integration.jl
julia --project=. benchmarks/benchmark_convolutions.jl
julia --project=. benchmarks/benchmark_threading.jl [max_threads]
```

### Per-tile breakdown and end-to-end performance

The left panel shows the normalized per-tile timing breakdown. On the GPU, the microturbulence, optical depth, and contribution function steps are fused into a single kernel dispatch (dashed dividers show estimated sub-step boundaries). The right panel shows wall-clock time for a complete `calc_formation_temp` call with disk integration, including atmosphere interpolation, absorption coefficient computation, and all tile iterations.

![per-tile and end-to-end benchmark](static/benchmark_pertile.png)

### Convolution kernels

Individual broadening kernel comparisons on a two-line Fe I test spectrum (`Nλ ≈ 400`, `Npad = 2400`).

![convolution benchmark](static/benchmark_convolutions.png)

## CPU vs. GPU numerical differences

The CPU and GPU paths use slightly different algorithms in a few places, leading to small numerical differences. In brief:

- **Microturbulence**: the GPU applies an analytical Fourier-domain Gaussian; the CPU samples a real-space kernel. Flux differences are ~4×10⁻⁴ at typical parameters.
- **RT macroturbulence kernels**: GPU `erfc` vs. CPU `erfc` differ at ~10⁻⁴ relative to peak.
- **Rotation kernels**: the CPU uses unpadded circular FFT; the GPU uses padded linear convolution. This produces edge effects in the first and last few pixels of the spectrum. Interior pixels agree to machine precision.
