# Integration Methods

```@meta
CurrentModule = FormationTemps
```

`calc_formation_temp` can integrate the emergent spectrum over the stellar disk in three
ways, selected with the `method` keyword:

| `method` | what it does | accuracy | speed |
|----------|--------------|----------|-------|
| `:disk` (default) | explicit numerical disk integration over `Nϕ` latitude bins | reference (self-consistent limb darkening) | slowest |
| `:quadrature` | ring-by-ring Gauss–Legendre μ-quadrature | ~1 K of `:disk` | fast (CPU & GPU) |
| `:hirano` | analytic Hirano et al. (2011) rotation + macroturbulence convolution | approximate (parametric limb darkening) | fastest |

```julia
star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=5000.0)

calc_formation_temp(star, linelist; method=:disk)        # reference
calc_formation_temp(star, linelist; method=:quadrature)  # fast, near-reference
calc_formation_temp(star, linelist; method=:hirano, u1=0.43, u2=0.31)  # fastest, approximate
```

!!! note "`convolve` is a deprecated alias"
    The old boolean keyword still works — `convolve=false` maps to `method=:disk` and
    `convolve=true` to `method=:hirano` — but new code should use `method`.

Every method runs on the GPU with `use_gpu=true`.

## `:disk` — explicit disk integration (reference)

Tiles the visible hemisphere into `Nϕ` latitude bands (`≈4Nϕ²/π` tiles) and, for each
tile, applies microturbulence + the rotational Doppler shift, integrates the optical
depth along the slanted ray, evaluates the contribution function, applies
macroturbulence, and accumulates with the projected-area weight. Limb darkening emerges
self-consistently from the radiative transfer (no parametric law). This is the most
faithful method and the reference against which the others are measured; it is also the
most expensive (cost grows with the tile count `∝ Nϕ²`).

## `:quadrature` — ring-by-ring μ-quadrature

The expensive radiative transfer depends on a tile only through its projection cosine
`μ`, and rotation enters only as a Doppler shift. `:quadrature` exploits this: it solves
the depth-resolved contribution function once at each of `Nμ` Gauss–Legendre μ nodes,
then integrates over the disk analytically — each μ "ring" is convolved with the
azimuthal line-of-sight-velocity distribution built from `N_az` samples. It reproduces
the `:disk` physics — including inclination and differential rotation — but replaces
`~10⁴` radiative-transfer solves with `~Nμ`.

Accuracy converges with the node counts `Nμ` and `N_az` (defaults `16` and `256`):

![quadrature convergence](static/quadrature_convergence.png)

At the defaults the formation temperature agrees with `:disk` to `~1 K` (mean `≪0.1 K`)
across inclinations and differential-rotation laws, while running roughly 5–10× faster
on CPU and up to ~100× faster on GPU (the speedup grows with the wavelength-grid size):

![quadrature speed scaling](static/quadrature_scaling.png)

!!! warning "Accuracy floor at small vsini"
    The per-ring Doppler kernel lives on the wavelength pixel grid, so a rotation
    profile only a few pixels wide (small `vsini`) is resolved only to ~pixel accuracy.
    The worst-pixel formation-temperature difference vs `:disk` is largest (~1–2 K) near
    `vsini ~ 2 km/s` and *shrinks* toward larger `vsini`; the mean stays `≪0.1 K`. Raise
    `Nμ`/`N_az` to reduce it. Use `:disk` if you need the reference answer at low `vsini`.
    Note also that `:quadrature` at `gpu_precision=Float32` is materially less accurate
    than at Float64 — prefer Float64.

![formation-temperature error vs vsini](static/quadrature_vsini.png)

The figure also shows `:hirano` for comparison: its larger, roughly `vsini`-independent
offset is a *physical model difference* (parametric limb darkening + a shift-invariant
kernel), not a numerical error.

## `:hirano` — analytic convolution

Computes the non-rotating, disk-integrated (`E₂(τ)`) flux contribution function once,
then convolves it with the analytic Hirano et al. (2011) rotation + radial-tangential
macroturbulence kernel, using the quadratic limb-darkening coefficients `u1`, `u2`. This
is the fastest method, but it assumes a parametric limb-darkening law and a
shift-invariant broadening kernel, so it does not capture center-to-limb variation of
the line profile. See [Broadening & Convolutions](@ref "Broadening & Convolutions") for
the kernels it uses.

## Rotation, inclination & differential rotation

All disk-based methods (`:disk`, `:quadrature`) use the projected rotational velocity
`vsini` (the observable equatorial value) and the [`StellarProps`](@ref) fields:

- `istar` — inclination (degrees; `90` = equator-on). For **rigid** rotation the
  broadening depends only on `vsini`, so `istar` has no effect; it becomes meaningful
  once differential rotation is enabled (it sets which latitude bands are visible).
- `α₂`, `α₄` — differential-rotation coefficients in the normalized rate law
  `Ω(ϕ)/Ω_eq = 1 - α₂·sin²ϕ - α₄·sin⁴ϕ`. Default `0` (solid body); positive values make
  the equator rotate faster than the poles (solar-like).

The per-tile line-of-sight velocity is `-vsini·f(ϕ)·(x_sky/R)`, where `x_sky` is the
sky-plane coordinate perpendicular to the projected spin axis and `f(ϕ)` is the rate law
above. `:hirano` supports rigid rotation only.

## Choosing a method

- Need the reference answer, or low `vsini` at highest fidelity → `:disk`.
- Want most of `:disk`'s fidelity at a fraction of the cost, including inclination /
  differential rotation → `:quadrature` (Float64).
- Want a fast approximate profile and can accept parametric limb darkening → `:hirano`.
```
