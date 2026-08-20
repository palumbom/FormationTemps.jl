# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FormationTemps.jl is a Julia package that wraps [Korg.jl](https://github.com/ajwheeler/Korg.jl) to compute stellar spectral formation temperatures — the atmospheric temperature at which photons contributing to each wavelength bin are most likely to escape. Requires Julia 1.12+.

## Build and Test Commands

```julia
# From Julia REPL, run all tests
using Pkg; Pkg.test()

# Or from shell
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file. The test files are include-only: they inherit `using Test` (and
# variously Korg, make_plots, test_plotdir) from runtests.jl, and macros resolve at lowering
# time, so a `using Test` inside a file's own `let` block leaves @testset undefined. Running
# `julia --project=. test/<file>.jl` directly fails on all but the few files whose `using`s
# are at top level (test_sigma_floor_cache.jl, test_graph_capture.jl).
julia --project=. -e 'using Test; include("test/test_convenience.jl")'

# Instantiate dependencies (first time or after Project.toml changes)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

CI runs on Julia 1.12 / ubuntu / x64. The Aqua.jl test suite is included in the test directory but commented out in `runtests.jl`.

### Building Docs Locally

The user's `~/.julia/config/startup.jl` unconditionally runs `Pkg.activate(".")`, which overrides `--project=docs`. Use `--startup-file=no` to bypass it:

```bash
julia --startup-file=no --project=docs docs/make.jl
```

## Architecture

### Computation Pipeline (`src/convenience.jl`)

`calc_formation_temp(star::StellarProps, linelist; ...)` is the top-level API. Internally:

1. **Absorption** (`absorption.jl`): `compute_alpha!` wraps `Korg.line_absorption!` to fill `αs` (total) and `αs_cont` (continuum-only) arrays of shape `(Natm, Nλ)`.
2. **Microturbulence** (`microturbulence.jl`): FFT-based Gaussian convolution along the wavelength axis applied per atmosphere layer.
3. **Optical depth** (`tau.jl`): `calc_tau_anchored_cpu!` / GPU Bezier integration to build `τs` arrays of shape `(Natm, Nλ)`. On the single-tile GPU Bézier path (`calc_intensity_direct!`), τ is kept in registers and never written to global memory (see fused kernel below).
4. **Contribution functions** (`contribution.jl`): `calc_flux_cfunc_cpu!` / GPU kernels yield `cfunc` of shape `(Natm-1, Nλ)`. On the single-tile GPU Bézier path, cfunc is also register-resident (fused with τ and intensity reduction in `calc_tau_cfunc_reduce_bezier!`).
5. **Broadening** (two modes):
   - `convolve=true`: analytical Hirano et al. (2011) rotation+macroturbulence kernel in Fourier space (fast, requires limb-darkening `u1`, `u2`).
   - `convolve=false` (default): numerical disk integration over a `Nϕ × 2Nϕ` stellar surface grid, looping over visible tiles.
6. **Formation temperature**: cumulative contribution function interpolated at 50% to yield `form_temps` per wavelength.

Returns `FormTempResult` with fields `wavs`, `flux`, `form_temps`, `cont_func`, `atmosphere`.

#### `cont_func` is a per-interval integral, not a density

`cont_func[k, j]` is `cfunc * Δτ_λ`: the contribution of the atmosphere interval between layers
`k` and `k+1`, such that `sum(cont_func, dims=1)` is the emergent flux. The layer width is
already inside each element, which splits its consumers in two:

- **Sums over depth take it as-is.** Weighted means, cumulative distributions, and the 50%
  crossing in `form_temps_from_cfunc` are sums in which the interval width cancels, so they are
  independent of the layer grid. Dividing by the width first drops that weighting.
- **Comparisons across depth must divide the width out first**, via `cfunc_per_dex(cfunc_dt,
  τ_ref)` → `dF/dlog₁₀τ_ref`. The atmosphere constructors use the native MARCS grid, which
  samples at Δlog τ_ref = 0.1 dex for log τ_ref ∈ [-3, +1] and 0.2 dex outside, so any
  across-layer comparison of raw `cont_func` carries a factor-of-two step at those two depths
  (T ≈ 4500 K and ≈ 8900 K for the Sun). This is why `ceiling_ratio` and `boundary_mask`
  require `τ_ref` — as a ratio of two intervals, a bare reduction over-flags by ~2×.

Plot `dF/dlog₁₀τ_ref`, not `dF/dτ_ref`: both remove the step, but the linear-τ density varies
by only ~1.5× between log τ_ref = -3 and the peak against ~1300× for the per-dex density, so it
flattens the real depth structure. `test/test_cfunc_measure.jl` pins both the conversion and the
grid-invariance of `ceiling_ratio`, and asserts that the bare reduction is *not* invariant so
the conversion cannot be dropped as redundant.

### GPU vs CPU

The module sets `GPU_DEFAULT = CUDA.functional()` at load time. Both `_calc_formation_temp_cpu` and `_calc_formation_temp_gpu` exist; the GPU path uses `AtmosphereGPU`, `GPUMemory`, `ConvolutionMemory`, and `BatchedMicroConvMem` structs that pre-allocate CuArrays and CUFFT plans. Pass `use_gpu=false` to force CPU. Pass `gpu_precision=Float32` to run GPU computations at single precision (absorption is always Float64 via Korg; conversion happens before GPU upload).

`ConvolutionMemory` caches FFT plans and pads to the next FFT-friendly length (factors of 2/3/5/7). Its `signal_cached` flag avoids redundant FFTs when absorption coefficients have not changed between disk integration tiles.

The GPU disk integration path (`convolve=false`) uses several optimizations:
- **Pre-uploaded tile parameters**: all μ, dA, and velocity arrays are uploaded to the GPU once before the tile loop. Batched kernels accept a `tile_offset` parameter to index into these arrays, eliminating per-batch H2D transfers.
- **Batched Fourier-domain macro accumulation**: instead of per-tile forward FFT → multiply → inverse FFT → accumulate, the macro path does batched forward FFT → per-tile spectral multiply-accumulate into a Fourier-space accumulator → single inverse FFT after the tile loop. This reduces ~21K serial FFTs to ~165 batched FFTs.
- **Batched macro kernel precomputation**: `compute_rt_macro_dft_layout_2d!` evaluates RT macro kernels for all unique μ values in a single 2D CUDA kernel launch, writing directly to DFT layout (zero-lag at index L). One batched R2C FFT produces `kernel_cache_flat`, a `(N_unique, nfreq)` CuArray used by the multiply-accumulate kernel.
- **Conditional GC**: `GC.gc(); CUDA.reclaim()` before batch sizing is skipped when `CUDA.free_memory()` already allows the maximum batch size (B=64).
- **Fused Bézier τ+cfunc+reduce kernel** (`calc_tau_cfunc_reduce_bezier!`): for the single-tile path (`calc_intensity_direct!`), the Bézier τ integration, Gauss-Legendre cfunc evaluation, and intensity reduction are fused into a single 1D kernel (one thread per wavelength, sequential z-loop over layers). τ and cfunc are accumulated entirely in registers; only scalar intensity per wavelength is written to global memory. `mem.τs` and `mem.cfunc` are NOT populated on this path.

### Key Structures (`src/structures/`)

| Struct | Purpose |
|---|---|
| `StellarProps` | Input: Teff, logg, Fe_H, vsini, ζ (macro), ξ (micro; scalar `T` or per-layer `AbstractVector{T}`), ρstar, istar. Parametric: `StellarProps{T, V}` where `V <: Union{T, AbstractVector{T}}` |
| `AtmosphereCPU` / `AtmosphereGPU` | Wrap Korg MARCS atmosphere; GPU variant holds `CuArray` fields including `nd_gpu` (number densities) |
| `GPUMemory` | Pre-allocated GPU working arrays (αs, τs, cfunc, cfunc_dt, flux); Bezier work arrays (tau_ds, tau_alphaC); anchored-τ constants (log_τ_ref, ifactor_base, use_anchored); `v_los_zeros` for stationary-frame flux path |
| `ConvolutionMemory` | Pre-allocated buffers + CUFFT plans for single-tile Doppler convolution; 1D FFT infrastructure for broadening kernels (kr_1d, kernel_row_ft_1d, plan_fwd_1d); `signal_cached` flag skips redundant FFTs; `xs_cpu` caches the CPU-side wavelength grid (set once in `_init_micro_params!`) to avoid redundant GPU→CPU syncs; `σ_floor` caches the grid-invariant Gaussian width floor, written with `xs_cpu` in `_init_micro_params!`; `n_kernel_builds` paces the underflow readback; used by `convolve=true` (Hirano) path and standalone GPU broadening functions |
| `MacroConvolutionMemory` | Similar layout to `ConvolutionMemory` but for macroturbulence kernels, plus the Hirano C2C infrastructure (`kc_1d`, `plan_bwd_1d`) and the `out_gpu` output buffer, none of which exist on `ConvolutionMemory` (`test_convmem_types.jl` asserts that). Carries `σ_floor` and `n_kernel_builds` for the same reasons. `xs_gpu` is reused as scratch by `hirano.jl` (stores frequency-domain σ values), so `xs_cpu` is the authoritative wavelength grid after init; used by `convolve=true` path and by `precompute_rt_macro_kernel_ft` (serial per-μ kernel FFT). In the `convolve=false` (disk integration) path, only `.L` and `.pad_left` geometry fields are used — the actual macro convolution uses inline batched buffers |
| `BatchedMicroConvMem` | Pre-allocated GPU buffers + cuFFT plans for batched Doppler convolution of B tiles simultaneously; shared signal FFT (Natm rows) + per-tile batched buffers (B*Natm rows); `signal_cached` flag skips redundant signal FFTs across batches; `xs_cpu` caches wavelength grid to avoid per-call GPU→CPU copies; `σ_floor` as above. **No** `n_kernel_builds`: the batched underflow readback is deliberately not paced |
| `AlphaCache` | Caches electron density guesses and abundance normalization across atmosphere columns to warm-start the chemical equilibrium solver in repeated `compute_alpha!` calls |
| `CPUTileWorkspace` | Per-thread pre-allocated working arrays for the CPU disk integration tile loop; includes radiative transfer buffers (τs, cfunc), convolution output buffers (αs_broad, macro_out), FFT work buffers (row_buf, kernel_ft, kernel_real), per-tile velocity buffer (v_los_buf), and in-place FFTW plans; eliminates per-tile heap allocations so threading scales without GC contention |
| `FormTempResult` | Output: wavs, flux, form_temps, cont_func, atmosphere |

### Broadening Kernels (`src/macroturbulence/`)

- `hirano.jl`: Hirano et al. (2011) combined rotation+macro kernel via Fourier transform (recommended for speed).
- `rad_tan.jl` / `iso_rad_tan.jl`: Radial-tangential macroturbulence kernels used in disk integration.
- `gray_rot.jl`: Gray rotation broadening kernel.
- `rad_tan_two.jl`, `iso_gaussian.jl`: Additional variants (iso_gaussian is excluded from module load).

#### GPU Convolution Output Slice Convention

The standalone GPU broadening functions (`convolve_rt_macro_gpu`, `convolve_gray_rotation_gpu`, etc.) and the `convolve=true` (Hirano) path pad the signal to FFT-friendly length `L`, convolve, then extract back to `Nλ` via `extract_valid!` into `cmem.out_gpu`. The extraction offset is `cmem.pad_left : cmem.pad_left + cmem.Nλ - 1`.

The `convolve=false` (disk integration) macro path does NOT use `cmem.out_gpu`. Instead, it accumulates in Fourier space across all tiles via `batched_macro_multiply_accumulate!`, then does a single post-loop IFFT + `extract_valid!` into `mac_out`/`mac_out_cont`.

#### CPU/GPU Algorithmic Differences Affecting Test Tolerances

- **Microturbulence**: CPU and GPU both build a real-space Gaussian kernel with wavelength-dependent σ(x) = x·v_mic/c, placed in DFT layout and R2C FFT'd. They agree to floating-point precision. GPU uses a two-tier dispatch: Tier 1 (uniform v_mic, common case) builds one base kernel, caches its FFT, and applies per-row phase rotation for v_los shifts; Tier 2 (varying v_mic) builds per-row kernels and batch-FFTs them. Physical constants in `constants.jl` are aligned with Korg's values (`c`, `kB`) to avoid Planck function bias between `Korg.blackbody` (CPU) and the inline GPU Planck formula. The Gaussian kernel computes the Doppler-shifted center as `Δx = (xj - λ0) - (v_los/c)*λ0` rather than `xj - λc` to avoid catastrophic cancellation when both `xj` and `λc` are ~10³ Å in Float32.

- **v_mic sourcing**: `StellarProps.ξ` (scalar or per-layer vector) is copied into `atm.v_mic` at the top of `_calc_formation_temp_cpu`/`_calc_formation_temp_gpu`. All internal code sources v_mic from the atmosphere struct, so external packages that populate `atm.v_mic` directly see their values used. The GPU batched kernel uses modular indexing (`(row-1) % Natm + 1`) to wrap the Natm-length v_mic across batched tiles without tiling allocation.

- **σ_floor is cached, and the cache is keyed by object identity**: `_init_micro_params!` is the sole writer of both `cmem.xs_cpu` and `cmem.σ_floor`, and they must be written together — anything that assigns `xs_cpu` on its own path leaves a stale floor that `_sigma_floor_cached` will return, because `xs_h === cmem.xs_cpu` still holds. The three single-tile kernel-build sites go through `_sigma_floor_cached(cmem, xs_h)` rather than reading the field, because the host-`xs` convolution overloads pass a fresh `collect` and must keep recomputing from the caller's grid; only the two batched sites, which read `bcmem.xs_cpu` unconditionally, take the field directly. σ_floor reaches the kernel only through `σ(x) = max(x·v_mic/c, σ_floor)`, so it is invisible at the `v_mic` values every other GPU test uses (`v_mic ≥ 600 m/s` puts `σ_phys` an order of magnitude above the floor); `test_sigma_floor_cache.jl` pins it at `v_mic = 0`, the only regime where the floor is the active branch. Note the floor's two terms trade places with precision: at Float64 on the production grid the quarter-pixel term wins, at Float32 `eps(T)·mean(xs)` does.

- **v_los additive model**: Disk integration computes per-tile v_los as `atm.v_los + rotation_velocity`. Default `atm.v_los` is zeros (backward-compatible). The flux path (`calc_flux_quantities`, `calc_flux_cfunc!`) uses a pre-allocated `mem.v_los_zeros` buffer instead of destructively zeroing `atm.v_los`.

- **All broadening kernels (iso_rad_tan, rad_tan, rad_tan_two, gray_rot, Hirano)**: CPU and GPU both use padded linear convolution with edge replication. CPU uses R2C FFTs (FFTW); GPU uses R2C FFTs (CUFFT). CPU/GPU agreement is at floating-point precision (~1e-8). (Previously the CPU used circular FFT, causing ~1e-4 edge divergence that was misattributed to CUDA vs Julia `erfc` differences — CUDA's Float64 `erfc` is accurate to ≤5 ULP.)

#### Float32 Precision Characteristics

The dominant Float32 precision loss is in the per-tile R2C/C2R FFT roundtrip during microturbulent convolution: absorption coefficients span ~5 orders of magnitude (continuum vs line core), and Float32 FFT arithmetic distributes absolute error proportional to the largest values. This produces ~2e-4 relative error in the convolved αs per tile, cascading to ~1e-3 in τ and ~1e-2 in cfunc_dt. After disk integration over ~10⁴ tiles (with partial cancellation from opposing Doppler signs), the net formation temperature error is ~1–2 K (~0.03% of ~5000 K). Other stages (τ integration, Planck evaluation, cumsum/interpolation) contribute negligibly at Float32.

Bit-identity of the emergent flux is scoped to a fixed CUDA.jl. `row_sums = sum(conv_gpu, dims=2)` inherits CUDA.jl's `mapreduce` partitioning, `Manifest.toml` is gitignored, and `Project.toml` pins `CUDA = "5.9.6"` as a floor rather than an exact version — so a fresh clone or a `Pkg.up` can move the last ULP of the flux with no source change. The enforceable form of the RT-exactness policy is therefore "no change within a comparison set" (one chain, one suite of runs being compared), not "no change ever". Re-baseline at a deliberate boundary, never mid-array.

The real-space tile accumulation kernels (`accumulate_batch_kernel!`, `accumulate_tile_kernel!`) use Kahan compensated summation with per-element compensation arrays to prevent O(N) rounding bias across the ~10⁴ tile sum. The Fourier-domain macro accumulation path (`batched_macro_multiply_accumulate_kernel!`) uses naive summation — acceptable for Float64 (~10⁴ × ε ≈ 10⁻¹²) but may need Kahan for Float32 if precision degrades. Run `debug/diagnose_f32_residuals.jl` for a stage-by-stage precision breakdown.

#### Short-Circuit Aliasing

All standalone GPU broadening functions that can short-circuit (ζ=0 or vsini=0) must do so **before** `copyto!(cmem.ys_gpu, ...)` and must return `CuArray(ys)` (a fresh allocation), not `cmem.ys_gpu` or `cmem.out_gpu`. Returning a shared buffer aliases it and causes the second call (continuum convolution) to overwrite the first call's result. The normal (non-short-circuit) return path uses `cmem.out_gpu`, which is safe because `extract_valid!` writes a fresh result into it on every call. This applies to the `convolve=true` (Hirano) path and standalone broadening benchmarks. The `convolve=false` disk integration path handles ζ=0 via a separate branch (`accumulate_batch!` instead of Fourier accumulation) and does not use `cmem.out_gpu`.

#### Micro-convolution results alias workspace scratch

`convolve_wavelength_axis_gpu` (every overload) and `convolve_wavelength_axis_batched!` return a `@view` into `cmem.conv_gpu` / `bcmem.conv_gpu` — the same buffer the per-row kernel build uses as scratch. CUDA.jl's `view` over a contiguous column range yields a `CuArray`, not a `SubArray`, so the return type carries no indication that the memory is not owned; `parent()` does not recover the workspace either. Check the device pointer if you need to confirm.

The result is valid only until the next convolution on that memory object, and for the vector-`v_los` overloads it dies *before* the replacement exists: `_build_per_row_kernels!` opens with `fill!(cmem.conv_gpu, zero(T))`, so a held result reads all zeros from partway through the next call. The scalar-`v_los` overloads build into `cmem.kr_1d` instead and only clobber `conv_gpu` at the inverse FFT.

`copy` the result if it must outlive the next convolution. The callers in `contribution.jl` consume it immediately (τ integration) and need no copy; the `:quadrature` drivers hoist the convolution out of the μ loop and so must copy. Do not substitute "nothing else uses this `cmem`" for the copy — `convenience.jl` also overlays the batched macro padding buffer on `bcmem.conv_gpu` through `unsafe_wrap`, so the memory has more writers than the call graph suggests.

Contrast the macro convolutions (`convolve_rt_macro_gpu` and friends), which extract into the dedicated `cmem.out_gpu`: that buffer survives a kernel build, but is still overwritten by the next macro call on the same `cmem` (see the slice convention above).

#### DFT Layout Conventions

Two DFT layout conventions coexist. The microturbulence kernels (`kernel_to_dft_layout_2d_*_gpu!`) place zero-lag at index **1** (`idx = d >= 0 ? d + 1 : L + d + 1`). The macro kernel precomputation (`compute_rt_macro_dft_layout_2d!`) places zero-lag at index **L** (`idx = d > 0 ? d : L + d`), matching the serial `precompute_rt_macro_kernel_ft` path (roll + ifftshift produces zero-lag at L for even-length arrays). Each convention is internally consistent within its own FFT pipeline.

#### Kernel normalization underflow guard

All FFT convolution kernels in `src/microturbulence.jl` are normalized via `./= sum(...)`. When Doppler `v_los` shifts the kernel center more than ~half a wavelength window outside the grid (≳ 38 km/s at λ ≈ 6000 Å with `σ_floor = 0.25·Δλ`), every Gaussian sample underflows to zero and the divide produces NaN. That NaN propagates through αs → τ → cfunc and yields all-NaN intensity for the affected atmosphere layer; in the scalar-v_los case it wipes the entire column. The guard `./= ifelse(iszero(s), one(T), s)` (used at all 9 microturbulence sites) falls back to a zero kernel — αs_conv row → 0, the discrete-convolution limit "shifted out of window contributes nothing." For all non-degenerate rows the divide is bit-identical to the prior implementation. A `@warn ... maxlog=3` surfaces the underflow so upstream callers can detect e.g. cm/s vs m/s unit bugs rather than seeing silent NaN. Counting the zero-sum rows needs a blocking `Int(CUDA.sum(iszero.(row_sums)))`, so on the two single-tile paths (`_build_per_row_kernels!`) that count is **paced** by `_maybe_report_underflow!`: every build for the first `_UNDERFLOW_EAGER_BUILDS` on a given memory object, then one in every `_UNDERFLOW_CHECK_STRIDE`, tracked by `cmem.n_kernel_builds`. Underflow is a configuration error, not a transient, so sampling the diagnostic is enough; the *guard* is unconditional. The readback must stay inside that branch — computing it unconditionally and gating only the `@warn` restores the full per-render cost and passes every assertion in `test_underflow_guard.jl`, because the warning stays paced; `test/test_graph_capture.jl` is the only gate that catches it. The check is additionally skipped while `CUDA.is_capturing()`: a readback inside a CUDA graph capture invalidates it, and `CUDA.@captured` re-executes its body under capture on every invocation, so an unguarded due build would break capture periodically and force a from-scratch graph re-instantiation. `_build_per_row_kernels!` is the only capture blocker in the GPU render path, so that guard is what keeps the path capturable at all. Two consequences to expect: once a caller captures the render into a graph the paced check stops firing inside it, so detection reduces to the builds before capture; and the "fires on the first renders or never" premise holds for a fixed-configuration run, not for a long-lived cmem driven by per-call data velocities (a cube column sweep reusing one cmem across columns can first underflow at column 5000, and the guard turns that into zeros rather than the NaNs such a sweep's own checks look for) — such a caller should zero `cmem.n_kernel_builds` before each call to force a check. The counter is per-struct, so each parallel-tempering chain gets its own eager window; that is why a cold chain can warn while a hot one stays quiet. The batched path (`convolve_wavelength_axis_batched!`) is deliberately **not** paced — it runs ~165 times per disk integration rather than per render, and per-tile `v_los` varies, so a late batch can be the first to underflow; `BatchedMicroConvMem` therefore has no counter field and the test asserts its absence. The 1D scalar path (`_build_kernel_ft_1d!`) also reads back every call: its `normval` is already a host scalar because it is used as a scalar divisor. The 15 macro/instrumental/convenience normalization sites (`rad_tan.jl`, `iso_rad_tan.jl`, `gray_rot.jl`, `rad_tan_two.jl`, `hirano.jl`, `instrumental.jl`, `convenience.jl`) carry `# TODO(zero-sum-guard)` markers; they short-circuit for ζ=0 / vsini=0 upstream so underflow shouldn't trigger in practice, but the same guard should be added if/when one is touched.

### Velocity and Unit Conventions

- All velocities are in **m/s** throughout (vsini, v_macro, v_micro inputs and stored fields).
- Wavelengths in **Angstrom** on the public API; internally Korg uses cm.
- Linelists from Korg use vacuum wavelengths (`l.wl` in cm); convert with `Korg.vacuum_to_air` if needed (see `test_convenience.jl`).
- Constants in `src/constants.jl`; `c_ms = 2.99782458e8` is the primary speed-of-light constant used in Doppler shifts.
- Disk-integration rotation (`calc_stellar_grid[_cpu]`, `disk_calculations.jl`): the per-tile line-of-sight velocity is `z_rot·c = -vsini·f(ϕ)·(x_sky/ρstar) = -vsini·f(ϕ)·cosϕ·sinθ`, where `x_sky` is the sky-plane coordinate perpendicular to the projected spin axis and `f(ϕ) = 1 - α₂·sin²ϕ - α₄·sin⁴ϕ` is the normalized differential-rotation rate factor (`diff_rot_factor` in `geometry.jl`). `vsini` is the **projected equatorial** rotational velocity (`f(0)=1`). For rigid rotation (`α₂=α₄=0`) the field is independent of `istar` — inclination is degenerate with `vsini` for a featureless sphere, so `istar` is a no-op. With differential rotation (`α≠0`) `istar` becomes physically meaningful: it selects which latitude bands are visible/weighted, and those rotate at different rates. Coefficients `α₂`, `α₄` live on `StellarProps` (default 0). (A prior implementation normalized the rotation vector to constant `vsini` and re-projected through the inclination matrix, dropping the `cosϕ` latitude factor and adding a spurious `sin i`; both errors vanished only at the default `istar=90°`.)

### Empirical Fits (`src/turb_fits.jl`)

`vmac_fit(Teff, logg)` uses Doyle et al. (2014); `vmac_fit(Teff)` uses Bruntt et al. (2010). `vmic_fit(Teff)` uses Bruntt et al. (2010). These are applied automatically when `v_macro=NaN` or `v_micro=NaN` in `StellarProps`.

## Data Directory

`src/config.jl` sets `FT.datdir` to `<repo>/data/` at load time (auto-created if absent). The solar linelist is at `FT.datdir * "Sun_VALD.lin"` and is required by the test suite.

## Python Interop

`pyproject.toml` defines Python dependencies managed by `uv`. The Python interface uses `juliacall`/`juliapkg`. The CI workflow sets `PYTHON=python` so that PyCall finds the correct interpreter; replicate this when testing locally if PyCall is used.

## Scripts

`scripts/` contains standalone analysis and plotting scripts used to generate figures for the paper (arXiv:2512.09861). They are not part of the package API and import `FormationTemps` directly. Most plotting scripts use Matplotlib via `PythonPlot` (note: the recent commits switched from `PyPlot` to `PythonPlot`).

## Plotting

The project matplotlib style (`fig.mplstyle`) enables `text.usetex: true`, so **all text passed to matplotlib** (titles, labels, annotations) is rendered through LaTeX. This means:

- **Never use raw Unicode Greek letters** (λ, μ, τ, ϕ, etc.) in plot strings. Use LaTeX math mode instead: `$N_\\lambda$`, `$\\mu$`, `$\\tau$`, `$N_\\phi$`.
- **Never use Unicode symbols** like `×` in plot strings. Use `$\\times$` instead. LaTeX commands like `\\times` must always be wrapped in `$...$` math delimiters.
- **Underscores in plain text** must be escaped: `calc\\_formation\\_temp`, not `calc_formation_temp`.
- Julia variable names with Unicode (e.g., `Nλ`, `αs`) are fine in code, but must be converted to LaTeX when used in any matplotlib string argument (`set_title`, `set_xlabel`, `ax.text`, `label=`, etc.).
