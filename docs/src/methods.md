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

- **`:quadrature`** is the recommended default: it reproduces the `:disk` result —
  including inclination and differential rotation (set via [`StellarProps`](@ref)) — to
  within ~1 K, while running roughly 5–10× faster on CPU and up to ~100× on GPU.
  Use Float64 (`:quadrature` at `gpu_precision=Float32` is noticeably less accurate).

- **`:disk`** is the reference. Use it for the definitive answer, or at very low
  `vsini`, where `:quadrature`'s pixel-grid Doppler kernel is least accurate
  (worst-pixel ~1–2 K near `vsini ≈ 2 km/s`; mean ≪ 0.1 K).

- **`:hirano`** is fastest but assumes a parametric limb-darkening law, so its error
  grows with `vsini` — a physical model difference, not a numerical one:

![formation-temperature error vs vsini](static/quadrature_vsini.png)

Wall-clock time for each method (CPU and GPU) as a function of the wavelength-grid size:

![speed vs Nλ](static/quadrature_scaling.png)

Rotation is set by `vsini` (the projected equatorial velocity), `istar`, and the
differential-rotation coefficients `α₂`, `α₄` — see [`StellarProps`](@ref). `:hirano`
supports rigid rotation only.
```
