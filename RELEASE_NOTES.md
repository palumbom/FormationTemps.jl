## FormationTemps.jl v3.2.0

This release makes the microturbulence convolution correct, on the CPU and the GPU, for layers whose Doppler shift or width carries the kernel past the edge of the wavelength window. Every other layer gives the same result as in v3.1.0, bit for bit.

### Results that change from v3.1

- **Doppler convolution is now correct for any shift and width, on the CPU and the GPU.** Each layer's Doppler kernel is built on the wavelength samples of the window itself. A kernel that reached past the end of the window was therefore cut off and renormalized, which piled the line up at the window edge, and a kernel shifted entirely off the window gave zero opacity. Those layers are now recomputed by direct convolution with the spectrum extended by its edge values, which is what the FFT convolution already does for every other layer. The CPU and GPU recompute the same layers and agree on them to rounding.
  - **Affected layers:** those where |v_los| + 3.7·v_mic exceeds the window's half-width in velocity. That is 16 km/s for a ±0.33 Å window at 6173 Å, and about 240 km/s for a 10 Å window.
  - **Unaffected layers** keep their v3.1.0 results bit for bit, so runs whose velocities stay below that limit give identical results.
  - **Size of the fix:** in a ±0.33 Å Fe I 6173 Å test on the GPU with every layer shifted past 16 km/s, the disk-center intensity was off by 10–70% of the continuum, or the line vanished. It now agrees with a direct-convolution reference to 3e-7 of the continuum at Float32.
- **A Doppler kernel that underflows now gives the window's edge opacity** instead of zero, on the CPU and the GPU. The warning is kept and still suggests checking that `v_los` is in m/s.
- **A GPU Doppler kernel whose samples are all subnormal no longer produces NaN.** Such a kernel has its center far outside the window, just short of underflowing to zero. It overflowed when normalized and turned the intensity of its column to NaN.

### Breaking changes

- **`set_signal_ft!(smem, b, signal_ft, αs)` now also takes the component's unconvolved opacities** (`Natm × Nλ`, on the device or the host), which the stacked render needs to recompute out-of-range layers. Pass the same opacities the component's serial render uses. The three-argument form throws an `ArgumentError`.

**Full Changelog**: https://github.com/palumbom/FormationTemps.jl/compare/v3.1.0...v3.2.0

---

## FormationTemps.jl v3.1.0

This release fixes several accuracy bugs in the radiative transfer and the broadening kernels, makes single-precision GPU runs agree with double precision, and adds a batched GPU render for several atmosphere components at once. Most of the fixes change numerical results slightly. FormationTemps.jl now requires CUDA.jl 6.5.1 or later.

### Results that change from v3.0

- **The source function is now interpolated linearly in τ within each atmosphere interval.** v3.0 interpolated the temperature and evaluated the Planck function at the quadrature nodes, which biased the intensity low by about 0.1% in the optical. The continuum now agrees with Korg's to 4e-5 (it was 8e-4 below). Formation temperatures move by up to about 0.4 K and normalized flux by up to 2e-4. Every method is affected.
- **Radial–tangential macroturbulence is normalized per component.** The radial and tangential Gaussians now each carry half the kernel's weight, even when one is narrower than a pixel (near disk center and limb). With ζ ≠ 0, `:disk` and `:quadrature` move by up to 1–2 K in formation temperature at Δλ = 0.01 Å, and by less than 0.05 K at 0.002 Å. As a side effect, `:quadrature` converges faster in Nμ for slowly rotating stars. `:hirano` is unaffected.
- **Bézier τ integration is fixed in FormationTemps; Korg is not.** The quadratic Bézier interpolant placed each control point on the wrong side of its node's tangent, which made the error in τ twice the trapezoid rule's in the interior of the atmosphere and first order at the top and bottom layers. Corrected, the scheme is third to fourth order. The sign error comes from Korg's `fritsch_butland_C`, which the CPU path called and the GPU kernels reproduced. It is still present in Korg through v1.3.1 and on `main`, reported as [Korg.jl#600](https://github.com/ajwheeler/Korg.jl/issues/600). FormationTemps now uses its own implementation on CPU and GPU, so the fix does not depend on the Korg version, and the two paths agree to rounding (the CPU path also differed before at local extrema of the opacity, where Korg does not set the slope to zero). The Bézier path is used only for atmospheres that do not supply τ_ref; MARCS models from Korg use the anchored scheme and are unaffected.
- **`convolve_instrument_gauss` no longer shifts the spectrum by one pixel** for inputs with an even number of samples (600 m/s at 0.01 Å).
- **`gpu_precision=Float32` now agrees with Float64.** Doppler and macroturbulence kernels are placed using the double-precision wavelength grid, not its single-precision rounding, which displaced every kernel by several m/s. Float32 now agrees with Float64 to about 3e-6 in flux and 0.005 K in formation temperature for `:disk` and `:quadrature` (previously several kelvin), and to a few hundredths of a kelvin for `:hirano` (previously 0.15 K). The v3.0 advice to keep Float64 with `:quadrature` no longer applies.
- **A more accurate E₂ in the `:hirano` flux path.** This mostly matters at Float32.

### New features

- **Batched multi-component GPU render**, for programs that synthesize many atmosphere columns repeatedly (MCMC samplers, for example): `StackedRenderMemory`, `set_signal_ft!` and `render_stacked!` render several components in one set of GPU launches. Each component's result is bit-identical to rendering it alone, and the render is about 1.7× faster for 3 components and 2.4× for 12 (Float32, RTX 6000 Ada).
- **The single-tile GPU render allocates no device memory**, so a caller can capture it into a CUDA graph and replay it.
- **Broadening kernels that underflow** (for example when a Doppler shift falls outside the wavelength window) now give zero instead of NaN, with a warning naming the kernel. This applies to every macroturbulence, rotation, instrumental and Hirano kernel; v3.0 did this only for microturbulence.

### Breaking changes

- Requires **CUDA.jl 6.5.1** or later; the 5.x series is no longer supported.
- `rt_macro_kernel` expects a uniform velocity grid with a sample at zero velocity, which every internal caller provides.

### Other

- The GPU per-row kernel normalization no longer depends on how many rows are normalized together or on the CUDA.jl version, so a tile's result does not depend on the batch around it.
- Python: the setup script installs FormationTemps `^3` from the registry when no local clone is found (it requested `^1`), adds Korg as a direct dependency, and checks up front for a Python linked against OpenSSL 3.5 or later, which Julia 1.12 needs.
- The test suite has a fast default tier and a slow tier: `FT_TEST_SLOW=1`, or `Pkg.test(test_args=["slow"])`.
- Documentation of the integration methods, GPU precision and contribution functions updated, with regenerated figures.

**Full Changelog**: https://github.com/palumbom/FormationTemps.jl/compare/v3.0.0...v3.1.0

---

## FormationTemps.jl v3.0.0

_Released 2026-11-04._

This release adds a faster disk-integration method, differential rotation, hydrogen lines, and flags for formation temperatures that are contaminated by the top of the model atmosphere.

### Results that change from v2

- **Formation temperatures move warmer.** The 50% point of the cumulative contribution is now interpolated against layer temperatures. v2 interpolated against interval midpoints, which made its values about half a layer interval too cool.
- **Atmospheres use their native MARCS layer grid.** v2 resampled them onto a uniform log τ grid.
- **Hydrogen lines are included by default.** Korg computes Balmer and Brackett lines separately from the linelist, so v2 left them out. Pass `hydrogen_lines=false` to exclude them.
- **Rotation with `istar ≠ 90°` is fixed.** v2 computed the rotational velocity field incorrectly at those inclinations. `vsini` is the projected equatorial velocity, and for rigid rotation `istar` has no effect. Results at the default `istar = 90°` are unchanged.
- **`cont_func` now has the same absolute scale for every method.** It is each layer interval's contribution to the emergent surface flux, in `Korg.synthesize` flux units. This changes `cont_func` for `method=:hirano` / `convolve=true` (by a factor of 2π) and for `:disk` with `ρstar ≠ 1`. `flux` and `form_temps` are unaffected.
- **Fast rotators on fine wavelength grids:** convolution padding is now sized from the broadening velocities, so kernels can no longer wrap around the edges of the spectrum.

### New features

- **`method` keyword** for `calc_formation_temp`, on CPU and GPU:
  - `:disk` (default): explicit surface tiling, as before.
  - `:quadrature` (new): matches `:disk` to within ~2 K and runs 20–35× faster on CPU (9–34× on GPU). Keep `gpu_precision=Float64` with this method.
  - `:hirano`: analytic convolution, needs `u1` and `u2`.

  `convolve` is deprecated. `convolve=true` still works, with a warning, and maps to `method=:hirano`.
- **Differential rotation** via `StellarProps(...; α₂, α₄)`, where `Ω(ϕ)/Ω_eq = 1 − α₂ sin²ϕ − α₄ sin⁴ϕ`. `:disk` and `:quadrature` support it; `:hirano` handles rigid rotation only. With differential rotation on, `istar` matters.
- **Top-of-atmosphere contamination flags.** In strong saturated line cores the contribution can still be rising where the model atmosphere ends, so `form_temps` there is only a lower limit. Each result now carries `ceiling_ratio`, `boundary_mask(result)` returns the affected wavelengths, and `calc_formation_temp` warns when it finds any. The `r_thresh` keyword sets the threshold.
  ```julia
  good = .!boundary_mask(result)
  ```
- **`cfunc_per_dex(result)`** converts `cont_func` into a density per dex of optical depth. Use it to plot `cont_func` against depth or to compare layers. Raw `cont_func` jumps by a factor of 2 where the MARCS grid spacing changes.
- **`form_temps_from_cfunc`** is now exported, so you can compute formation temperatures from your own contribution functions.

### Breaking changes

- `FormTempResult` has two new fields, `ceiling_ratio` and `r_thresh`, placed before `atmosphere`. The five-argument constructor `FormTempResult(wavs, flux, form_temps, cont_func, atmosphere)` still works.
- Requires **Korg 1.2**.

### Other

- Julia compat now allows 1.12 and later (previously 1.12.x only). The Python interface is still pinned to Julia 1.12.
- Lower per-call overhead in the GPU convolution paths.



**Full Changelog**: https://github.com/palumbom/FormationTemps.jl/compare/v2.0.0...v3.0.0

---

## FormationTemps.jl v2.0.0

_Released 2026-06-02._

### Highlights

- **GPU single-precision mode**: pass `gpu_precision=Float32` to `calc_formation_temp` to roughly halve GPU memory and improve throughput on consumer GPUs. Absorption is always computed at Float64 (Korg requirement); conversion happens automatically before GPU upload. Formation temperature error is ~1–2 K.
- **CPU multithreading**: the disk integration path (`use_gpu=false, convolve=false`) now parallelizes tiles across Julia threads. Launch with `julia -t auto` to benefit.
- **Batched GPU disk integration**: tiles are processed in batches of up to 64 simultaneously, with shared signal FFTs and dual CUDA streams. Macro broadening accumulates in Fourier space across all tiles, reducing ~21K serial FFTs to ~165 batched FFTs.
- **Anchored tau integration**: new optical depth scheme (Wheeler et al. 2022, Appendix A.1) anchors quadrature to the MARCS reference tau grid, reducing sensitivity to atmosphere layer spacing. Activates automatically when the atmosphere model supplies `tau_ref`.
- **Chunked computation**: `calc_formation_temp_chunked` processes arbitrarily long wavelength ranges in fixed-width chunks with overlap stitching and wing padding. Supports a `callback` for streaming results to disk.

### New features

- **`AlphaCache`** (exported): caches electron density warm starts and abundance normalization for repeated `compute_alpha!` calls, improving performance when computing absorption for multiple wavelength windows.
- **`calc_formation_temp_chunked`** (exported): wavelength-windowed computation for large linelists. Configurable `chunk_width`, `overlap`, `wing_padding`, and optional `callback` for streaming.
- **`get_nd(atm)`**: accessor for atmosphere number densities (not exported; use `FT.get_nd`).
- **Hirano kernel on GPU**: rotation+macroturbulence kernel computed directly on device, eliminating a CPU→GPU transfer.
- **Progress bar**: disk integration shows a progress bar by default. Suppress with `showprogress=false`.

### API changes

#### `calc_formation_temp` new keyword arguments

| Keyword | Default | Description |
|---|---|---|
| `gpu_precision` | `Float64` | GPU float type (`Float32` for half-memory mode) |
| `showprogress` | `true` | Show/hide disk integration progress bar |
| `buffer` | `2.0` | Wavelength padding (Å) around linelist bounds |
| `line_buffer_Å` | `10.0` | Continuum interpolation range (Å) around linelist for `compute_alpha!` |

#### Removed functions

- `calc_tau_cpu!`, `calc_tau_trapezoid!`, `calc_tau_simpson!` — replaced by `calc_tau_anchored_cpu!` (used internally; not part of the public API).

#### Struct changes

- `FormTempResult` fields are now concrete `Vector{T}` / `Matrix{T}` instead of `AbstractArray`.
- `StellarProps.A_X` is now `Vector{T}` instead of `AbstractArray{T,1}`.
- `StellarProps.ξ` is now `Union{T, AbstractVector{T}}`; pass a vector to `v_micro` for per-layer microturbulence.
- `FormTempResult.cont_func` is now documented as C × Δτ (was labeled dC/dτ).
- `AtmosphereGPU` and `AtmosphereCPU`: fields `σ_v`/`μ_v` renamed to `v_mic`/`v_los`. New fields `vx`, `vy`, `vz` (per-layer velocity components, initialized to zero; intended for downstream MHD simulation data).

#### `calc_formation_temp_chunked` behavior change

- `stitch_mode` kwarg removed.
- When `callback=nothing`, now returns `Vector{FormTempResult}` (one entry per chunk) instead of a single stitched `FormTempResult`. Stitching is the caller's responsibility.

### Bug fixes

- Fixed bitwise-AND (`&`) vs short-circuit (`&&`) in `rad_tan_two.jl` zero-velocity guards.
- Fixed Float32 catastrophic cancellation in Gaussian Doppler kernel center calculation.
- Fixed Int32 overflow in batched tau kernel loop bounds on large wavelength grids.
- Fixed GPU memory budget underestimate for batched disk integration.
- Added Kahan compensated summation for tile accumulation to prevent O(N) rounding drift across ~10⁴ tiles.

### Performance

- **Fused τ+cfunc GPU kernel (anchored path)**: `calc_tau_cfunc_dt_fused!` computes τ in registers and evaluates the contribution function in a single pass, eliminating the `τs_batch` global memory round-trip on the anchored path.
- **Fused Bézier τ+cfunc+reduce kernel**: `calc_tau_cfunc_reduce_bezier!` fuses τ integration, contribution function evaluation, and intensity reduction into a single pass for the single-tile Bézier path. τ and cfunc are register-resident; only scalar intensity per wavelength is written to global memory, eliminating ~2.4 MB of intermediate storage.
- **GPU memory aliasing**: macro buffers aliased over existing bcmem allocations via `unsafe_wrap`, saving ~1.1 GB at B=64.
- **2D accumulation kernel**: accumulates cfunc_dt per layer-wavelength; flux derived post-loop via `sum(cfunc_acc, dims=1)`, eliminating separate flux accumulators.
- Zero-allocation GPU convolution pipeline with pre-allocated 1D FFT plans and `extract_valid!` CUDA kernel.
- Signal and macro kernel FFT caching eliminates redundant transforms across tiles.
- Multi-threaded `compute_alpha!` (absorption computation).
- `CPUTileWorkspace` pre-allocates per-thread buffers, eliminating GC contention in threaded tile loops.

---

## FormationTemps.jl v1.0.0

_Released 2026-01-22._

### What's New

* Initial versioned release of FormationTemps.jl, to accompany the revised manuscript to ApJ.
