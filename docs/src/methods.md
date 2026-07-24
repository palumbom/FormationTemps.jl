# Integration Methods

```@meta
CurrentModule = FormationTemps
```

`calc_formation_temp` can integrate over the stellar disk in three ways, selected with
the `method` keyword:

| `method` | description | accuracy | speed |
|----------|-------------|----------|-------|
| `:disk` (default) | explicit disk integration over `Nϕ` latitude bins; self-consistent limb darkening | reference | slowest |
| `:quadrature` | Gauss–Legendre μ-quadrature; reproduces the `:disk` physics | ~1 K of `:disk` | fast (CPU & GPU) |
| `:hirano` | analytic rotation + macroturbulence convolution (needs `u1`, `u2`) | approximate | fastest |

```julia
calc_formation_temp(star, linelist; method=:disk)                      # reference
calc_formation_temp(star, linelist; method=:quadrature)                # fast, near-reference
calc_formation_temp(star, linelist; method=:hirano, u1=0.43, u2=0.31)  # fastest, approximate
```

All three run on the GPU with `use_gpu=true`.

!!! note "`convolve` is a deprecated alias"
    `convolve=false` maps to `method=:disk` and `convolve=true` to `method=:hirano`.
    New code should use `method`.

## Choosing a method

- **`:disk`** is the default and the reference. Use it for the definitive answer, or at
  very low `vsini`, where `:quadrature`'s pixel-grid Doppler kernel is least accurate
  (worst-pixel ~2 K near `vsini ≈ 2 km/s`; mean ≪ 0.1 K). That low-`vsini` floor comes from
  representing a only-a-few-pixels-wide Doppler kernel on the wavelength grid; it does not
  improve with `Nμ`.

- **`:quadrature`** is the one to reach for when `:disk` is too slow: it reproduces the
  `:disk` result — including inclination and differential rotation (set via
  [`StellarProps`](@ref)) — to within ~1 K, while running roughly 5–10× faster on CPU and
  up to ~100× on GPU. Use Float64; at `gpu_precision=Float32` it is noticeably less
  accurate. The reformulation itself is exact rather than approximate: radiative transfer
  is wavelength-local, so a Doppler shift of the input opacity shifts the emergent
  intensity identically, and rotation can therefore be applied as a per-ring convolution
  after the transfer solve. That is what lets RT be solved once per μ node instead of once
  per surface tile. What remains is the pixel-grid discretization of the ring kernel.

- **`:hirano`** is fastest but assumes a parametric limb-darkening law, so its error
  grows with `vsini` — a physical model difference, not a numerical one:

![formation-temperature error vs vsini](static/quadrature_vsini.png)

Wall-clock time for each method (CPU and GPU) as a function of the wavelength-grid size:

![speed vs Nλ](static/quadrature_scaling.png)

Rotation is set by `vsini` (the projected equatorial velocity), `istar`, and the
differential-rotation coefficients `α₂`, `α₄` — see [`StellarProps`](@ref). `:hirano`
supports rigid rotation only.

For rigid rotation `istar` has no physical effect (the projected velocity field carries no
inclination dependence); it matters only once `α₂`/`α₄` are nonzero, because inclination
then selects which latitude bands — rotating at different rates — are visible. See
[`StellarProps`](@ref) for the one numerical caveat at finite `Nϕ`.

## Accuracy notes

- **`Nμ` is the most effective accuracy knob** for slow rotators, and defaults to 32. Cost
  scales linearly with it. Halving it to 16 costs roughly an order of magnitude in
  worst-pixel agreement with `method=:disk`; doubling it to 64 buys comparatively little.
  `test_quadrature.jl` asserts this convergence.

- **Formation temperatures in deep line cores are lower limits.** Where more than half the
  flux contribution comes from the topmost layer interval, the 50% crossing is set by the
  top of the MARCS grid rather than by the line's actual formation depth.
  `form_temps_from_cfunc` counts and warns about affected wavelengths. Balmer cores are the
  common case, and hydrogen lines are on by default.

- **Convolution padding is sized from the kernel support.** The rotational, macroturbulent
  and microturbulent kernels are applied as padded linear convolutions; the padding is
  derived from `vsini`, `ζ` and `ξ` together with `Δλ`, so a fast rotator on a finely
  sampled grid does not silently wrap. If the ring Doppler kernel is wider than the
  synthesis window itself, `:quadrature` warns that the broadening is truncated — widen the
  window with `minλ`/`maxλ`.
