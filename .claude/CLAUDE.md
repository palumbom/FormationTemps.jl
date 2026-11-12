# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FormationTemps.jl is a Julia package that wraps [Korg.jl](https://github.com/ajwheeler/Korg.jl) to compute stellar spectral formation temperatures — the atmospheric temperature at which photons contributing to each wavelength bin are most likely to escape. Requires Julia 1.12+.

## Build and Test Commands

```julia
# From Julia REPL, run the fast tier (the default)
using Pkg; Pkg.test()

# Or from shell: the fast tier directly, or in Pkg's sandbox
julia --project=. test/runtests.jl
julia --project=. -e 'using Pkg; Pkg.test()'

# Fast and slow tiers. test/test_tiers.jl defines run_slow_tests(): FT_TEST_SLOW=1 or "slow" in ARGS.
# The slow tier adds the convergence checks, the GPU T-range and Float32 smoke runs, and the finer
# tilings; the fast tier runs the same comparisons on coarser tilings with wider tolerances.
FT_TEST_SLOW=1 julia --project=. test/runtests.jl
julia --project=. -e 'using Pkg; Pkg.test(test_args=["slow"])'

# Run a single test file. The test files are include-only: they inherit `using Test` (and
# variously Korg, make_plots, test_plotdir) from runtests.jl, and macros resolve at lowering
# time, so a `using Test` inside a file's own `let` block leaves @testset undefined. Running
# `julia --project=. test/<file>.jl` directly fails on all but the few files whose `using`s
# are at top level (test_sigma_floor_cache.jl, test_graph_capture.jl).
julia --project=. -e 'using Test; include("test/test_convenience.jl")'

# Instantiate dependencies (first time or after Project.toml changes)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Requires Julia 1.12 or later (`julia = "^1.12"`) and CUDA.jl 6.5.1; the 5.x series is no longer in the compat,
and the FFT module is `CUDA.cuFFT` (on 6.x `CUDA.CUFFT` is only a deprecated alias for it; 5.x has no `cuFFT`).
On the development workstation `Manifest.toml` is resolved under the default `julia` (1.13), and
`julia +1.12 --project=.` reads its own gitignored `Manifest-v1.12.toml`.

**The fast tier is slower under Julia 1.13 because of one compile** (measured 2026-11-11, Korg 1.2.1):
`Korg.interpolate_marcs` compiles in about 30 s on 1.12.7 and about 230 s on 1.13.3, once per process;
execution time and results are the same. Every `calc_formation_temp(star, …)` path pays it, so the fast tier
takes about 12 min under 1.13.3 and about 4.5 min as `julia +1.12 --project=. test/runtests.jl`.

CI runs on Julia 1.12 and the latest 1.x release (`'1'`) / ubuntu / x64. The runners have no GPU, so CI covers only the CPU tests; the GPU files in `runtests.jl` run only in local passes, which are the pre-release gate. The Aqua.jl test suite is included in the test directory but commented out in `runtests.jl`.

### Building Docs Locally

The user's `~/.julia/config/startup.jl` unconditionally runs `Pkg.activate(".")`, which overrides `--project=docs`. Use `--startup-file=no` to bypass it:

```bash
julia --startup-file=no --project=docs docs/make.jl
```

## Architecture

### Computation Pipeline (`src/convenience.jl`)

`calc_formation_temp(star::StellarProps, linelist; ...)` is the top-level API. Internally:

1. **Absorption** (`absorption.jl`): `compute_alpha!` wraps `Korg.line_absorption!` to fill `αs` (total) and `αs_cont` (continuum-only) arrays of shape `(Natm, Nλ)`. Hydrogen lines, which Korg builds from its own Stark/MHD physics rather than the linelist, are added by `_add_hydrogen_line_absorption!` (`hydrogen_lines=true` by default) to `αs` only, never `αs_cont`, and with `ξ = 0` because microturbulence is applied downstream.
2. **Microturbulence** (`microturbulence.jl`): FFT-based Gaussian convolution along the wavelength axis applied per atmosphere layer.
3. **Optical depth** (`tau.jl`): `calc_tau_anchored_cpu!` / GPU Bezier integration to build `τs` arrays of shape `(Natm, Nλ)`. On the single-tile GPU Bézier path (`calc_intensity_direct!`), τ is kept in registers and never written to global memory (see fused kernel below).
4. **Contribution functions** (`contribution.jl`): `calc_flux_cfunc_cpu!` / GPU kernels yield `cfunc` of shape `(Natm-1, Nλ)`. On the single-tile GPU Bézier path, cfunc is also register-resident (fused with τ and intensity reduction in `calc_tau_cfunc_reduce_bezier!`). Each interval is integrated by two-point Gauss–Legendre in τ, with the Planck source evaluated at the two layers and interpolated **linearly in τ** to the nodes: Korg's linear-source scheme up to the quadrature error. Do not interpolate T to the nodes and evaluate B there instead: T(τ) is concave and B(T) steeply convex in the optical, so that scheme biases the intensity low by about 1e-3 at 5000 Å on the native MARCS grid (its continuum sat 8e-4 below Korg's; the B-linear one sits 4e-5 below). All eleven sites (CPU, GPU single-tile, batched, fused, Bézier helper `_accumulate_cfunc`, stacked) use the same expressions, which keeps the paths bit-identical to each other.
5. **Disk integration**, selected by `method` (three modes):
   - `:disk` (default): numerical disk integration over a `Nϕ × 2Nϕ` stellar surface grid, looping over visible tiles.
   - `:quadrature` (`quadrature.jl`): ring-by-ring Gauss–Legendre μ-quadrature (`Nμ` nodes, `N_az` azimuth samples for the per-ring Doppler kernel), dispatched to `_calc_formation_temp_quadrature_cpu`/`_gpu`. Solves transfer once per μ node rather than once per tile.
   - `:hirano`: analytical Hirano et al. (2011) rotation+macroturbulence kernel in Fourier space (requires limb-darkening `u1`, `u2`; rigid rotation only).

   `convolve::Bool` is a deprecated public alias (`true` ⇒ `:hirano`, `false` ⇒ `:disk`). `_calc_formation_temp_cpu`/`_gpu` still take `convolve::Bool` internally, so "the `convolve=true` path" elsewhere in this file means `method=:hirano` and "the `convolve=false` path" means `method=:disk`.
6. **Formation temperature** (`form_temps_from_cfunc` in `interpolations.jl`): node-anchored cumulative contribution (`F = 0` at `Ts[1]`, interval `k` assigned to the deep node `Ts[k+1]`) interpolated at 50% against node temperatures. Interpolating against interval centers (`elav(Ts)`) biases the result half an interval cool. Also warns where `ceiling_ratio > r_thresh` (contribution still peaking at the top of the atmosphere).

Returns `FormTempResult` with fields `wavs`, `flux`, `form_temps`, `cont_func`, `ceiling_ratio`, `r_thresh`, `atmosphere`.

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

The absolute scale is the emergent surface flux `F = 2π∫Iμdμ` in `Korg.synthesize`'s flux
units, identical for every `method`. The methods' internal sums are not: the `:disk` tile sum
over projected area is `ρstar²·F`, while the `:quadrature` weights `w_k·μ_k` and the `:hirano`
E₂ integral give `F/2π`. Each assembly site (`_calc_formation_temp_cpu`/`_gpu`, the two
quadrature drivers) multiplies `cont_func` by its factor *after* `flux` and `form_temps` are
computed from the unscaled array, so those stay bit-identical, and `:disk` at `ρstar = 1`
multiplies by exactly 1. Every other consumer is a ratio and cannot see the scale, so
`test/test_cont_func_scale.jl`, which compares the continuum level against Korg, is its only gate.

### GPU vs CPU

The module sets `GPU_DEFAULT = CUDA.functional()` at load time. Both `_calc_formation_temp_cpu` and `_calc_formation_temp_gpu` exist; the GPU path uses `AtmosphereGPU`, `GPUMemory`, `ConvolutionMemory`, and `BatchedMicroConvMem` structs that pre-allocate CuArrays and cuFFT plans. Pass `use_gpu=false` to force CPU. Pass `gpu_precision=Float32` to run GPU computations at single precision (absorption is always Float64 via Korg; conversion happens before GPU upload).

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
| `StellarProps` | Input: Teff, logg, Fe_H, vsini, ζ (macro), ξ (micro; scalar `T` or per-layer `AbstractVector{T}`), ρstar, istar, α₂, α₄ (differential rotation). Parametric: `StellarProps{T, V}` where `V <: Union{T, AbstractVector{T}}` |
| `AtmosphereCPU` / `AtmosphereGPU` | Wrap Korg MARCS atmosphere; GPU variant holds `CuArray` fields including `nd_gpu` (number densities) |
| `GPUMemory` | Pre-allocated GPU working arrays (αs, τs, cfunc, cfunc_dt, flux); Bezier work arrays (tau_ds, tau_alphaC); anchored-τ constants (log_τ_ref, ifactor_base, use_anchored); `v_los_zeros` for stationary-frame flux path |
| `ConvolutionMemory` | Pre-allocated buffers + cuFFT plans for single-tile Doppler convolution; 1D FFT infrastructure for broadening kernels (kr_1d, kernel_row_ft_1d, plan_fwd_1d); `signal_cached` flag skips redundant FFTs; `xs_cpu` caches the CPU-side wavelength grid (set once in `_init_micro_params!`) to avoid redundant GPU→CPU syncs; `σ_floor` caches the grid-invariant Gaussian width floor and `dx_gpu` the kernel offsets `xs[j] - xs[i0]`, both written with `xs_cpu` in `_init_micro_params!`; `n_kernel_builds` paces the underflow readback; used by `convolve=true` (Hirano) path and standalone GPU broadening functions |
| `MacroConvolutionMemory` | Similar layout to `ConvolutionMemory` but for macroturbulence kernels, plus the Hirano C2C infrastructure (`kc_1d`, `plan_bwd_1d`) and the `out_gpu` output buffer, none of which exist on `ConvolutionMemory` (`test_convmem_types.jl` asserts that). Carries `σ_floor`, `dx_gpu` and `n_kernel_builds` for the same reasons; `dx_gpu` is also rewritten by every macro convolution (see "Kernel offsets come from the Float64 grid"). `xs_gpu` is reused as scratch by `hirano.jl` (stores frequency-domain σ values), so `xs_cpu` is the authoritative wavelength grid after init; used by `convolve=true` path and by `precompute_rt_macro_kernel_ft` (serial per-μ kernel FFT). In the `convolve=false` (disk integration) path, only `.L` and `.pad_left` geometry fields are used — the actual macro convolution uses inline batched buffers |
| `BatchedMicroConvMem` | Pre-allocated GPU buffers + cuFFT plans for batched Doppler convolution of B tiles simultaneously; shared signal FFT (Natm rows) + per-tile batched buffers (B*Natm rows); `signal_cached` flag skips redundant signal FFTs across batches; `xs_cpu` caches wavelength grid to avoid per-call GPU→CPU copies; `σ_floor` and `dx_gpu` as above. **No** `n_kernel_builds`: the batched underflow readback is deliberately not paced |
| `AlphaCache` | Caches electron density guesses and abundance normalization across atmosphere columns to warm-start the chemical equilibrium solver in repeated `compute_alpha!` calls |
| `CPUTileWorkspace` | Per-thread pre-allocated working arrays for the CPU disk integration tile loop; includes radiative transfer buffers (τs, cfunc), convolution output buffers (αs_broad, macro_out), FFT work buffers (row_buf, kernel_ft, kernel_real), per-tile velocity buffer (v_los_buf), and in-place FFTW plans; eliminates per-tile heap allocations so threading scales without GC contention |
| `FormTempResult` | Output: wavs, flux, form_temps, cont_func, ceiling_ratio, r_thresh, atmosphere. The five-argument constructor derives `ceiling_ratio` from `cont_func` and `atmosphere.τs`; internal paths use it so the two cannot disagree |

### Broadening Kernels (`src/macroturbulence/`)

- `hirano.jl`: Hirano et al. (2011) combined rotation+macro kernel via Fourier transform (recommended for speed).
- `rad_tan.jl` / `iso_rad_tan.jl`: Radial-tangential macroturbulence kernels used in disk integration.
  In `rad_tan.jl` and `rad_tan_two.jl` the radial and the tangential Gaussian are each normalized by the sum of their own samples (`_sampled_gauss_sum`, an analytic sum that the GPU kernels evaluate per sample), on the CPU and the GPU alike, so each carries the weight ½ even when it is narrower than a pixel (μ → 1, μ → 0). Normalizing only their sum hands the narrow term part of the other's weight. The joint `./= sum` that follows removes only rounding and window truncation. The analytic sum assumes a uniform velocity grid with a sample at zero velocity, which every internal caller provides.
- `gray_rot.jl`: Gray rotation broadening kernel.
- `rad_tan_two.jl`, `iso_gaussian.jl`: Additional variants (iso_gaussian is excluded from module load).

#### GPU Convolution Output Slice Convention

The standalone GPU broadening functions (`convolve_rt_macro_gpu`, `convolve_gray_rotation_gpu`, etc.) and the `convolve=true` (Hirano) path pad the signal to FFT-friendly length `L`, convolve, then extract back to `Nλ` via `extract_valid!` into `cmem.out_gpu`. The extraction offset is `cmem.pad_left : cmem.pad_left + cmem.Nλ - 1`.

The `convolve=false` (disk integration) macro path does NOT use `cmem.out_gpu`. Instead, it accumulates in Fourier space across all tiles via `batched_macro_multiply_accumulate!`, then does a single post-loop IFFT + `extract_valid!` into `mac_out`/`mac_out_cont`.

#### CPU/GPU Algorithmic Differences Affecting Test Tolerances

- **Microturbulence**: CPU and GPU both build a real-space Gaussian kernel with wavelength-dependent σ(x) = x·v_mic/c, placed in DFT layout and R2C FFT'd. They agree to floating-point precision. On the GPU a scalar `v_los` and `v_mic` build one kernel (`_build_kernel_ft_1d!`); a vector `v_los` builds one real-space kernel per row, with the shift in its center, and batch-FFTs them (scalar or per-row `v_mic`). No path applies the shift as a Fourier-domain phase rotation. Physical constants in `constants.jl` are aligned with Korg's values (`c`, `kB`) to avoid Planck function bias between `Korg.blackbody` (CPU) and the inline GPU Planck formula. The GPU Gaussian kernel computes its argument as `Δx = dx[j] - (v_los/c)*λ0`, with `dx[j] = xs[j] - xs[i0]` precomputed on the host (`dx_gpu`), rather than `xj - λc`, which cancels catastrophically when both `xj` and `λc` are ~10³ Å in Float32.

- **v_mic sourcing**: `StellarProps.ξ` (scalar or per-layer vector) is copied into `atm.v_mic` at the top of `_calc_formation_temp_cpu`/`_calc_formation_temp_gpu`. All internal code sources v_mic from the atmosphere struct, so external packages that populate `atm.v_mic` directly see their values used. The GPU batched kernel uses modular indexing (`(row-1) % Natm + 1`) to wrap the Natm-length v_mic across batched tiles without tiling allocation.

- **σ_floor is cached, and the cache is keyed by object identity**: `_init_micro_params!` is the sole writer of `cmem.xs_cpu`, `cmem.σ_floor` and `cmem.dx_gpu`, and they must be written together — anything that assigns `xs_cpu` on its own path leaves a stale floor that `_sigma_floor_cached` will return, because `xs_h === cmem.xs_cpu` still holds. The three single-tile kernel-build sites go through `_sigma_floor_cached(cmem, xs_h)` rather than reading the field, because the host-`xs` convolution overloads pass a fresh `collect` and must keep recomputing from the caller's grid; only the two batched sites, which read `bcmem.xs_cpu` unconditionally, take the field directly. σ_floor reaches the kernel only through `σ(x) = max(x·v_mic/c, σ_floor)`, so it is invisible at the `v_mic` values every other GPU test uses (`v_mic ≥ 600 m/s` puts `σ_phys` an order of magnitude above the floor); `test_sigma_floor_cache.jl` pins it at `v_mic = 0`, the only regime where the floor is the active branch. Note the floor's two terms trade places with precision: at Float64 on the production grid the quarter-pixel term wins, at Float32 `eps(T)·mean(xs)` does.

- **A Doppler kernel narrower than about a pixel snaps the shift, and that is kept on purpose.** The point-sampled Gaussian puts its weight on the nearest samples, so when `v_mic·λ/c` is below about a pixel its first moment is not the requested shift (at `v_mic = 0` the error reaches a good fraction of a pixel). An exact-shift kernel (the Gaussian integrated over the linear-interpolation hat, which at `v_mic = 0` is linear interpolation) was implemented and measured on 2026-11-12, and rejected: in `:disk` it smooths the *opacity* by up to a quarter pixel² before the nonlinear transfer, and on the default 0.01 Å grid, where thermal lines at `ξ = 0` are a couple of pixels wide, that moved line cores by tens of kelvin and took `:disk` about 8× further from a converged 0.001 Å run than the snapping does (whose per-tile shift errors partly average out over the disk). At `ξ = 400 m/s` the two were equivalent. Exact sub-pixel shifts at `v_mic ≈ 0` therefore need a grid on which the thermal width spans many pixels. Applying the rotation shift to the intensity after transfer, as `:quadrature` does, would avoid both errors.

- **Kernel offsets come from the Float64 grid**: the opacity is computed on the Float64 grid and only its values are converted to `gpu_precision`, so the Doppler kernel has to be placed on that grid too. `_init_micro_params!(cmem, xs_h, xs_exact)` takes the offsets `dx_gpu` from `xs_exact`; `_calc_formation_temp_gpu` and `_calc_formation_temp_quadrature_gpu` call it with `λs_korg` right after allocating their memories, and `StackedRenderMemory` latches `dx` from its `cmems`. A memory left to the lazy init on its first convolution gets `xs_exact = xs_h`, the offsets of the rounded grid: the same bits at Float64, but at Float32 each kernel is displaced by the rounding of the samples under it, a m/s-level velocity shift that depends on the window. A caller that builds its own Float32 `ConvolutionMemory` must therefore make that call itself before its first render. `_ring_doppler_kernel` takes its velocity step from the Float64 grid for the same reason. The macro GPU kernels are placed the same way but do not go through `_init_micro_params!`: `convolve_rt_macro_gpu` (both forms), `convolve_iso_rt_macro_gpu`, `convolve_gray_rotation_gpu` and `precompute_rt_macro_kernel_ft` take the Float64 grid as the keyword `xs_exact` and upload the offsets to `cmem.dx_gpu` on every call, as they upload `xs_gpu`; without the keyword they difference the grid they were given. The `:disk` batched precompute passes `bcmem.dx_gpu` and the center wavelength to the eight-argument `compute_rt_macro_dft_layout_2d!`; the seven-argument form differences a device grid and is kept for callers that hold only that. `_kernel_offsets` is the one place the offsets are computed. `:hirano` builds its kernel in Fourier space and needs none of this.

- **v_los additive model**: Disk integration computes per-tile v_los as `atm.v_los + rotation_velocity`. Default `atm.v_los` is zeros (backward-compatible). The flux path (`calc_flux_quantities`, `calc_flux_cfunc!`) uses a pre-allocated `mem.v_los_zeros` buffer instead of destructively zeroing `atm.v_los`.

- **All broadening kernels (iso_rad_tan, rad_tan, rad_tan_two, gray_rot, Hirano)**: CPU and GPU both use padded linear convolution with edge replication. CPU uses R2C FFTs (FFTW); GPU uses R2C FFTs (cuFFT). CPU/GPU agreement is at floating-point precision (~1e-8). (Previously the CPU used circular FFT, causing ~1e-4 edge divergence that was misattributed to CUDA vs Julia `erfc` differences — CUDA's Float64 `erfc` is accurate to ≤5 ULP.)

#### Float32 Precision Characteristics

The dominant Float32 precision loss is in the per-tile R2C/C2R FFT roundtrip during microturbulent convolution: absorption coefficients span ~5 orders of magnitude (continuum vs line core), and Float32 FFT arithmetic distributes absolute error proportional to the largest values. This produces ~2e-4 relative error in the convolved αs per tile, cascading to ~1e-3 in τ and ~1e-2 in cfunc_dt. After disk integration over ~10⁴ tiles (with partial cancellation from opposing Doppler signs), the net formation temperature error is ~1–2 K (~0.03% of ~5000 K). Other stages (τ integration, Planck evaluation, cumsum/interpolation) contribute negligibly at Float32.

Bit-identity of the emergent flux is scoped to a fixed environment. The cuFFT library, the GPU compiler and the device all sit under it, `Manifest.toml` is gitignored, and `Project.toml` gives CUDA.jl a floor rather than an exact version — so a fresh clone or a `Pkg.up` can move the last ULP of the flux with no source change. The single-tile Float32 render was bit-identical between CUDA.jl 5.11.1 and 6.5.1 on one RTX 6000 Ada (2026-11-11, one state), but nothing enforces that. The enforceable form of the RT-exactness policy is therefore "no change within a comparison set" (one chain, one suite of runs being compared), not "no change ever". Re-baseline at a deliberate boundary, never mid-array.

No library reduction sits in the render path. The per-row Doppler kernels are normalized by `_normalize_rows!`, whose summation order depends only on the padded length and `_ROW_NORM_LANES`; a row's kernel is therefore the same bits whether it is built alone, in a batch of tiles, or under another CUDA.jl. `sum(A, dims=2)` must not come back: CUDA.jl's reductions pick their grouping from the array's shape (AcceleratedKernels' launch shapes from GPUArrays 12 on, `mapreduce`'s block count before), which made a tile's convolved opacities depend on the batch around it and moved the flux between CUDA.jl versions.

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

#### cuFFT pairs rows at odd padded lengths

For an odd padded length `L` the batched real plans (`plan_fwd`, `plan_bwd`, R2C and C2R along
the wavelength axis) transform the rows in consecutive pairs (2k−1, 2k), and each row's output
carries round-off-level leakage from its partner; the last row of an odd row count is unpaired.
An even `L` transforms every row independently. Measured 2026-11-11 by zeroing one input row
and listing the output rows that change, Float32 and Float64, CUDA.jl 5.11.1 and 6.5.1: paired
at `L = 1875` (what Nλ = 1321 pads to), independent at 784, 960, 1024 and 2744.
`next_fft_friendly_len` does not avoid odd lengths. What follows from it:

- **A tile is independent of its batch only if it has an even number of rows, or `L` is even.**
  With an even `Natm` the pairs fall inside each tile and a batched tile is bit-identical to the
  same tile convolved alone; with an odd `Natm` at an odd `L` they straddle tile boundaries and
  it is not, whatever the normalization.
- **A row whose kernel underflowed is exactly zero only when `L` is even or its partner is zero
  too.** Otherwise it holds ε times its partner.
- **`batched_spectral_multiply!` zeroes the spectrum rows past `BNatm`**, launching over every
  row, so with an odd `BNatm` at an odd `L` the last active row is paired with zeros rather than
  a stale spectrum. Launching it over `BNatm` rows only would reopen that.
- **The `:disk` macro padding is not zeroed the same way.** `pad_signal!` fills all `B·Natm1`
  rows of `mac_pad` from `cfdt_batch`, whose rows past `Bcur·Natm1` hold the previous batch's
  contribution functions, and `Natm1` is odd for MARCS (55). In a final partial batch with an
  odd `Bcur` at an odd `L_mac`, the last active row therefore carries round-off leakage from
  finite data of its own magnitude; the pairs straddle tiles in every batch anyway.

#### Kernel normalization underflow guard

All FFT convolution kernels in `src/microturbulence.jl` are normalized to unit sum in real space: the CPU sites and the 1D scalar GPU site via `./= sum(...)`, the four GPU per-row sites (`_build_per_row_kernels!`, `convolve_wavelength_axis_batched!`) via `_normalize_rows!`. When Doppler `v_los` shifts the kernel center more than ~half a wavelength window outside the grid (≳ 38 km/s at λ ≈ 6000 Å with `σ_floor = 0.25·Δλ`), every Gaussian sample underflows to zero and the divide produces NaN. That NaN propagates through αs → τ → cfunc and yields all-NaN intensity for the affected atmosphere layer; in the scalar-v_los case it wipes the entire column. The guard — `./= ifelse(iszero(s), one(T), s)` at the five `sum` sites, the same test on the row sum inside `normalize_rows_kernel!` — falls back to a zero kernel — αs_conv row → 0, the discrete-convolution limit "shifted out of window contributes nothing." (At an odd padded length the convolved row is ε times its cuFFT pair partner rather than exactly zero; see "cuFFT pairs rows at odd padded lengths".) A `@warn ... maxlog=3` surfaces the underflow so upstream callers can detect e.g. cm/s vs m/s unit bugs rather than seeing silent NaN. Counting the zero rows (`_count_zero_rows`: a library `maximum` over the normalized kernels, which allocates, then a blocking read) is the cost the pacing avoids, so on the two single-tile paths (`_build_per_row_kernels!`) that count is **paced** by `_maybe_report_underflow!`: every build for the first `_UNDERFLOW_EAGER_BUILDS` on a given memory object, then one in every `_UNDERFLOW_CHECK_STRIDE`, tracked by `cmem.n_kernel_builds`. Underflow is a configuration error, not a transient, so sampling the diagnostic is enough; the *guard* is unconditional. The readback must stay inside that branch — computing it unconditionally and gating only the `@warn` restores the full per-render cost and passes every assertion in `test_underflow_guard.jl`, because the warning stays paced; `test/test_graph_capture.jl` is the only gate that catches it. The check is additionally skipped while `CUDA.is_capturing()`: a readback inside a CUDA graph capture invalidates it, and `CUDA.@captured` re-executes its body under capture on every invocation, so an unguarded due build would break capture periodically and force a from-scratch graph re-instantiation. `_build_per_row_kernels!` is the only capture blocker in the GPU render path, so that guard is what keeps the path capturable at all. Two consequences to expect: once a caller captures the render into a graph the paced check stops firing inside it, so detection reduces to the builds before capture; and the "fires on the first renders or never" premise holds for a fixed-configuration run, not for a long-lived cmem driven by per-call data velocities (a cube column sweep reusing one cmem across columns can first underflow at column 5000, and the guard turns that into zeros rather than the NaNs such a sweep's own checks look for) — such a caller should zero `cmem.n_kernel_builds` before each call to force a check. The counter is per-struct, so each parallel-tempering chain gets its own eager window; that is why a cold chain can warn while a hot one stays quiet. The batched path (`convolve_wavelength_axis_batched!`) is deliberately **not** paced — it runs ~165 times per disk integration rather than per render, and per-tile `v_los` varies, so a late batch can be the first to underflow; `BatchedMicroConvMem` therefore has no counter field and the test asserts its absence. The 1D scalar path (`_build_kernel_ft_1d!`) also reads back every call: its `normval` is already a host scalar because it is used as a scalar divisor. The macro, instrumental and Hirano kernels (`rad_tan.jl`, `iso_rad_tan.jl`, `gray_rot.jl`, `rad_tan_two.jl`, `hirano.jl`, `instrumental.jl`) normalize through `_normalize_kernel!`, which applies the same guard with an unpaced warning naming the kernel; the batched macro precompute in `convenience.jl` guards its per-row sums inline. Those kernels are unshifted and centered in the window, so their sums do not vanish in practice and the guard is defensive; normalize any new kernel the same way.

#### CUDA graph capture vs repeat replay

Consumers that drive the GPU render in a tight loop (an MCMC sampler, e.g. Anemoi) may want to
capture it into a CUDA graph and replay it. Two separate properties gate that, and they fail
for different reasons:

- **Capture** requires no blocking device→host operation in the region. `_build_per_row_kernels!`
  is the only place in the render that has one; its underflow readback is paced *and* skipped
  under `CUDA.is_capturing()`, which is what makes the render capturable at all. See the
  underflow-guard section above — the readback must stay inside that branch.
- **Repeat replay** additionally requires no *allocation* in the region. A stream-ordered
  allocation becomes a graph node that cannot be replayed, so the graph is single-use: the
  first launch succeeds and the next fails with `ERROR_INVALID_VALUE`.

The render path allocates no device memory on a build whose underflow check is not due: the
per-row kernels are normalized by `_normalize_rows!`, a hand-written kernel, and the τ
integration, the cfunc kernel and `reduce_intensity!` allocate nothing. A due build does allocate
(`_count_zero_rows`), and is skipped under capture. Checked 2026-11-11 on CUDA.jl 6.5.1:
`CUDA.@allocated` over three renders returns 0, and one captured graph of three renders,
replayed five times, reproduced the direct intensities bit for bit and followed a change of
`v_los`. Replay ran 1.13× faster than direct launches on the single-tile render: the render is
bound by device-side kernel latency, not by host launch overhead.

Rules for the normalization:

- **Keep it a hand-written kernel.** `sum(A, dims=2)` allocates, `sum!(buf, A)` still allocates
  reduction scratch inside the library, and either one's summation grouping follows the array's
  shape (see "Float32 Precision Characteristics"). Verify zero bytes with `CUDA.@allocated`,
  not by reading the call.
- **`_ROW_NORM_LANES` is part of the arithmetic; `_ROW_NORM_ROWS` is not.** The lanes per row
  fix the order of the additions, so changing them moves the last bit of every per-row kernel.
  The rows per block only trade coalescing against occupancy and can be retuned freely.
- **The layout is measured.** One block per row is uncoalesced striding and costs about twice
  the device time of the library reduction; 32 rows per block starves Float64, whose arithmetic
  is throughput-bound on consumer-class devices. Any retuning has to beat `sum(A, dims=2)`
  followed by the broadcast divide in *device* time at both precisions, since the single-tile
  vector-`v_los` path serves every caller (`contribution.jl`, `convenience.jl`'s
  `convolve=true` path, the `:quadrature` drivers) and the batched path the disk integration.
  Time it with the calls queued behind a long-running kernel and bracketed by device events; a
  host-side timing of a synchronized call mostly measures launch overhead.

Capture freezes control flow as well as launches. `calc_intensity_cfunc!` branches on
`cmem.signal_cached`, so a graph captured with the cache warm has the `copyto!(mem.αs, αs_init)`
permanently elided and cannot serve the cold state — a capturing consumer needs one graph per
cache state. The paced underflow diagnostic also stops firing inside a graph, so detection
reduces to the renders taken before capture.

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
