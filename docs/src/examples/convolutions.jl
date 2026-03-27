using Korg
using PythonPlot; plt = PythonPlot.pyplot
using FormationTemps; FT = FormationTemps
plt.style.use(joinpath(FT.moddir, "fig.mplstyle"))

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]

# set stellar parameters
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 5000.0   # m/s
ζ_RT = 3400.0    # radial-tangential macroturbulent broadening (m/s)
ξ = 850.0        # microturbulent broadening (m/s)

# compute a base (unbroadened) spectrum via disk integration
star_props_base = FT.StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H,
                                  vsini=0.0, v_macro=0.0, v_micro=ξ)
result_base = FT.calc_formation_temp(star_props_base, linelist; Δλ=0.01)
wavs = result_base.wavs
flux_base = result_base.flux

# BREAK1

# --- compare disk integration vs. convolution approximation ---

# disk integration (default, convolve=false)
star_props = FT.StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H,
                             vsini=vsini, v_macro=ζ_RT, v_micro=ξ)
result_int = FT.calc_formation_temp(star_props, linelist; Δλ=0.01, convolve=false)

# convolution approximation (convolve=true uses Hirano et al. 2011 kernel;
# u1 and u2 are linear and quadratic limb-darkening coefficients)
u1 = 0.43
u2 = 0.31
result_conv = FT.calc_formation_temp(star_props, linelist; Δλ=0.01, convolve=true, u1=u1, u2=u2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 7.2), sharex=true)
ax1.plot(wavs, result_int.flux,  label="{\\rm integration}", lw=2.0, c="k")
ax1.plot(wavs, result_conv.flux, label="{\\rm convolution}", lw=2.0, ls="--")
ax1.set_ylabel("{\\rm Normalized Flux}")
ax1.legend()

ax2.plot(wavs, result_int.flux .- result_conv.flux, c="k", lw=0.8)
ax2.axhline(0, ls=":", c="gray")
ax2.set_xlabel("{\\rm Vacuum Wavelength [\\AA]}")
ax2.set_ylabel("{\\rm Integration} \$-\$ {\\rm Convolution}")
ax2.set_xlim(5410, 5415)

fig.tight_layout()
fname = joinpath(FT.moddir, "docs", "src", "static", "convolution_vs_integration.png")
fig.savefig(fname, bbox_inches="tight")
plt.show()
plt.clf(); plt.close()

# BREAK2

# --- compare the broadening kernels directly ---

# build a velocity grid for kernel visualization
Δλ = wavs[2] - wavs[1]
λ0 = wavs[length(wavs) ÷ 2 + 1]
Δv = FT.c_ms * Δλ / λ0
vs = collect((-length(wavs)÷2 : length(wavs)÷2 - 1)) .* Δv

# evaluate the three macroturbulence kernels at disk center (μ=1)
k_gray_rot  = FT.gray_rot_kernel(vs, vsini, u1)
k_iso = FT.gray_iso_rt_macro_kernel(vs, ζ_RT)
k_rt = FT.rt_macro_kernel(vs, ζ_RT, 1.0)

fig, ax1 = plt.subplots(figsize=(9.6, 4.8))
ax1.plot(vs ./ 1e3, k_gray_rot, lw=2.0, label="{\\rm Gray rotation (vsini = $(round(Int, vsini/1e3)) km/s)}")
ax1.plot(vs ./ 1e3, k_iso, lw=2.0, label="{\\rm Isotropic RT macro (}\$\\zeta\$ {\\rm = $(round(Int, ζ_RT/1e3)) km/s)}")
ax1.plot(vs ./ 1e3, k_rt, lw=2.0, label="{\\rm Anisotropic RT macro at }\$\\mu=1\${\\rm  (}\$\\zeta\$ {\\rm = $(round(Int, ζ_RT/1e3)) km/s)}")
ax1.set_xlabel("{\\rm Velocity [km/s]}")
ax1.set_ylabel("{\\rm Kernel Amplitude}")
ax1.set_xlim(-1.5*max(vsini, ζ_RT)/1e3, 1.5*max(vsini, ζ_RT)/1e3)
ax1.legend(fontsize=12)
fig.tight_layout()
fname = joinpath(FT.moddir, "docs", "src", "static", "broadening_kernels.png")
fig.savefig(fname, bbox_inches="tight")
plt.show()
plt.clf(); plt.close()

# BREAK3

# --- apply the convolution functions to the base spectrum ---

flux_gray_rot = FT.convolve_gray_rotation(wavs, flux_base, vsini, u1)
flux_iso = FT.convolve_iso_rt_macro(wavs, flux_base, ζ_RT)
flux_hirano = FT.convolve_hirano_rotmacro(wavs, flux_base, vsini, ζ_RT, u1, u2)

fig, ax1 = plt.subplots(figsize=(9.6, 4.8))
ax1.plot(wavs, flux_base, label="{\\rm Unbroadened}", lw=0.8)
ax1.plot(wavs, flux_gray_rot, label="{\\rm Gray rotation only}", lw=2.0, ls="--")
ax1.plot(wavs, flux_iso, label="{\\rm Isotropic RT macro only}", lw=2.0, ls="-.")
ax1.plot(wavs, flux_hirano, label="{\\rm Hirano rotation + macro}", lw=2.0, ls=":")
ax1.set_xlim(5410, 5415)
ax1.set_xlabel("{\\rm Vacuum Wavelength [\\AA]}")
ax1.set_ylabel("{\\rm Normalized Flux}")
ax1.legend()
fig.tight_layout()
fname = joinpath(FT.moddir, "docs", "src", "static", "broadened_spectra.png")
fig.savefig(fname, bbox_inches="tight")
plt.show()
plt.clf(); plt.close()

# BREAK4
