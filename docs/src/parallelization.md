# Parallelization

```@meta
CurrentModule = FormationTemps
```

FormationTemps.jl parallelizes the disk integration pipeline in two complementary ways: CPU multithreading across stellar surface tiles, and GPU acceleration via [CUDA.jl](https://cuda.juliagpu.org/stable/). Both target the same bottleneck — the per-tile radiative transfer loop that dominates wall-clock time when `convolve=false`.

## CPU Multithreading

When `use_gpu=false` and `convolve=false`, `calc_formation_temp` distributes tiles across Julia threads. Start Julia with multiple threads to benefit:

```bash
julia -t auto           # use all available cores
julia -t 8              # use 8 threads
```

The environment variable `JULIA_NUM_THREADS` can also be set before launching Julia.

!!! warning "Python interoperability"
    CPU multithreading is not compatible with calling FormationTemps.jl from Python via juliacall/PythonCall. Julia's multi-threaded garbage collector can conflict with PythonCall's runtime bridge, causing hard crashes (SIGBUS/segfault). When calling from Python, `JULIA_NUM_THREADS` must be set to `1`. See the [Python Tutorial](@ref "Python Tutorial") for details.

    If you need both parallelism and Python, use the GPU path (`use_gpu=True`) or run the computation in Julia and load the results in Python.

## GPU Acceleration

GPU acceleration is enabled by default when a compatible NVIDIA device is detected (`CUDA.functional() == true`); pass `use_gpu=false` to force the CPU path.

### Setup

GPU support requires an NVIDIA GPU and a working CUDA toolkit. CUDA.jl handles driver detection and toolkit installation automatically in most cases:

```julia
using CUDA
CUDA.functional()   # should return true
CUDA.versioninfo()  # check device and toolkit details
```

If `CUDA.functional()` returns `false`, consult the [CUDA.jl troubleshooting guide](https://cuda.juliagpu.org/stable/installation/troubleshooting/). FormationTemps.jl will fall back to the CPU path transparently.

### GPU precision

Pass `gpu_precision=Float32` to `calc_formation_temp` to run GPU computations at single precision:

```julia
result = calc_formation_temp(star, linelist; gpu_precision=Float32)
```

Absorption coefficients are always computed at Float64 (a Korg requirement) and converted to the target precision before GPU upload. Float32 roughly halves GPU memory usage and can improve throughput on consumer GPUs with higher FP32 than FP64 performance. The default is `Float64`.

### What gets accelerated

The disk integration pipeline repeats the full radiative transfer calculation for every visible tile on the stellar surface — typically thousands of tiles for `Nϕ = 128`. Each tile requires microturbulent broadening (FFT-based Gaussian convolution of the absorption coefficient matrix along the wavelength axis, per atmosphere layer), optical depth integration through the atmosphere to build `τ(λ)` at each layer, computation of the intensity contribution function per layer, and radial-tangential macroturbulence convolution of the contribution function. On the GPU, tiles are processed in batches and the first three steps use batched CUDA kernels. The macroturbulence convolution uses a separate FFT-based kernel per tile. All GPU memory is pre-allocated at the start of the call and reused throughout.

## Benchmarks

The results below were obtained on an Intel Xeon w5-3435X (16 cores / 32 threads) with an NVIDIA RTX 6000 Ada (48 GB). All benchmarks use a two-line Fe I test spectrum near 6300 Å with `Δλ = 0.005` Å, solar stellar parameters (`vsini = 2100` m/s, `ζ_RT = 3500` m/s, `ξ = 850` m/s), and `Nϕ = 128`. To reproduce, run `julia --project=. benchmarks/run_all.jl` — see the [`benchmarks/`](https://github.com/palumbom/FormationTemps.jl/tree/main/benchmarks) directory for individual scripts.

### Per-tile breakdown

Per-tile timing breakdown for CPU (single tile) and GPU (per tile from a batch of 8). GPU times include both total and continuum absorption paths running on dual CUDA streams.

![per-tile benchmark](static/benchmark_pertile.png)

### CPU thread scaling

Speedup and wall-clock time as a function of the number of Julia threads for a full `calc_formation_temp` call with disk integration.

![threading benchmark](static/benchmark_threading.png)

### Performance vs. wavelength grid size

End-to-end wall-clock time as a function of `Nλ` (varied by changing `Δλ` over a fixed wavelength window) for single-threaded CPU, multi-threaded CPU, and GPU.

![Nlambda benchmark](static/benchmark_nlambda.png)

### Convolution kernels

Individual broadening kernel timings for each convolution type (CPU vs GPU Float64 vs GPU Float32).

![convolution benchmark](static/benchmark_convolutions.png)

## CPU vs. GPU numerical differences

The CPU and GPU paths use slightly different algorithms in a few places, leading to small numerical differences:

- Microturbulence: the GPU applies an analytical Fourier-domain Gaussian; the CPU samples a real-space kernel. Flux differences are ~4×10⁻⁴ at typical parameters.
- All broadening kernels (iso_rad_tan, rad_tan, rad_tan_two, gray_rot, Hirano): CPU and GPU both use padded linear convolution with edge replication. Agreement is at floating-point precision.

### Float32 vs. Float64 accuracy

The figures below compare flux and formation temperature spectra for two Fe I lines near 6300 Å, computed at CPU Float64, GPU Float64, and GPU Float32 for a solar-like star. The top row overlays the three spectra (visually indistinguishable); the bottom row shows residuals relative to the CPU Float64 reference on a symmetric log scale. GPU Float64 residuals are small, dominated by the algorithmic differences described above. GPU Float32 residuals are orders of magnitude larger but still modest: flux residuals at the ~10⁻⁴ level and formation temperature differences of ~1 K.

The dominant source of Float32 precision loss is the R2C/C2R FFT roundtrip in the per-tile microturbulent convolution: absorption coefficients span ~5 orders of magnitude across the wavelength grid, and Float32 FFT arithmetic distributes absolute rounding error proportional to the largest (line-core) values across all wavelengths. The tile accumulation kernels use Kahan compensated summation to prevent additional O(N) rounding from the ~10⁴ tile sum, and the Gaussian kernel construction avoids catastrophic cancellation in the Doppler-shifted center wavelength. See `debug/diagnose_f32_residuals.jl` for a stage-by-stage precision breakdown.

![GPU precision: convolution path](static/gpu_precision_convolve.png)

![GPU precision: disk integration](static/gpu_precision_diskint.png)

These plots can be regenerated with `julia --project=. benchmarks/gpu_precision_comparison.jl`.
