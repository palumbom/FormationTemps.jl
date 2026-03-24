using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, NPZ, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
import PythonPlot; plt = PythonPlot
mpl = plt.matplotlib
plt.ioff()

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")

# manually create the linelist
wl = 6.17333e-5
log_gf = -2.880
species = Korg.Species("Fe I")
E_lower = 2.223
factor = 1.0
gamma_rad   = factor * exp10(8.31)
gamma_stark = factor * exp10(-6.16)
gamma_vdw   = log10(factor * exp10(-7.69))
linelist = [Korg.Line(wl, log_gf, species, E_lower, gamma_rad, gamma_stark, gamma_vdw)]

# wavelength grid
λ0 = linelist[1].wl * 1e8
λs_korg  = collect(range(λ0 - 0.25, λ0 + 0.25, step=0.001))
vels_korg = FT.c_ms .* (λs_korg .- λ0) ./ λ0

# abundances and atmosphere
A_X = Korg.asplund_2009_solar_abundances

# _resample_log_tau gives a uniform log-τ grid; pass the same resampled atmosphere to
# both Korg and FormationTemps so the comparison is on identical layer structures.
atm_korg      = Korg.interpolate_marcs(5777, 4.44, A_X)
atm_resampled = FT._resample_log_tau(atm_korg)
atm_gpu       = FT.AtmosphereGPU(atm_korg)

# atmosphere dimensions (from the resampled atmosphere — matches atm_gpu)
Natm   = atm_gpu.Natm
Ts_gpu = atm_gpu.Ts_gpu

# velocity broadening
val = 2400.0
σ_v = CuArray{Float64}(fill(val, Natm))
μ_v = CUDA.zeros(Float64, Natm)

# disk angles
μ_vals = 0.2:0.05:1.0
μ_idx  = 1

# canonical Korg synthesis — anchored τ, same resampled atmosphere as FormationTemps
sol = synthesize(atm_resampled, linelist, A_X, λs_korg;
                 vmic=val/1e3, tau_scheme="anchored",
                 mu_values=μ_vals, hydrogen_lines=false)
αs_korg = sol.alpha

# absorption coefficients
αs      = zeros(Natm, length(λs_korg))
αs_cont = zeros(Natm, length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# convolve with microturbulence
Nλ   = length(λs_korg)
Npad = 100
cmem        = FT.ConvolutionMemory(Nλ, Natm, Npad)
αs_gpu      = FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs,      μ_v, σ_v)
cmem        = FT.ConvolutionMemory(Nλ, Natm, Npad)
αs_cont_gpu = FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs_cont, μ_v, σ_v)

# anchored τ precomputed arrays
α_ref            = αs_cont[:, end]
log_τ_ref_gpu    = CuArray(log.(atm_gpu.τs))
ifactor_base_gpu = CuArray(atm_gpu.τs ./ α_ref)

# output arrays
λs_gpu              = CuArray(λs_korg)
τs_gpu              = CUDA.zeros(Float64, Natm, Nλ)
cfunc_int_gpu       = CUDA.zeros(Float64, Natm - 1, Nλ)
cfunc_flux_gpu      = CUDA.zeros(Float64, Natm - 1, Nλ)
cfunc_flux_cont_gpu = CUDA.zeros(Float64, Natm - 1, Nλ)

ts = (32, 16)
bs = (cld(Nλ, ts[1]), cld(Natm, ts[2]))

# intensity cfunc at μ = μ_vals[μ_idx]
FT.calc_tau_anchored_gpu!(μ_vals[μ_idx], log_τ_ref_gpu, ifactor_base_gpu, αs_gpu, τs_gpu)
@cuda threads=ts blocks=bs FT.calc_intensity_cfunc!(μ_vals[μ_idx], Ts_gpu, λs_gpu, τs_gpu, cfunc_int_gpu)
CUDA.synchronize()
intensity_gpu = Array(sum(cfunc_int_gpu .* diff(τs_gpu, dims=1), dims=1))'

# flux cfunc
τs_gpu .= 0.0
FT.calc_tau_anchored_gpu!(1.0, log_τ_ref_gpu, ifactor_base_gpu, αs_gpu, τs_gpu)
@cuda threads=ts blocks=bs FT.calc_flux_cfunc!(Ts_gpu, λs_gpu, τs_gpu, cfunc_flux_gpu)
CUDA.synchronize()
flux_gpu = 2π .* Array(CUDA.sum(cfunc_flux_gpu .* diff(τs_gpu, dims=1), dims=1))'

# continuum flux cfunc
τs_gpu .= 0.0
FT.calc_tau_anchored_gpu!(1.0, log_τ_ref_gpu, ifactor_base_gpu, αs_cont_gpu, τs_gpu)
@cuda threads=ts blocks=bs FT.calc_flux_cfunc!(Ts_gpu, λs_gpu, τs_gpu, cfunc_flux_cont_gpu)
CUDA.synchronize()
flux_cont_gpu = 2π .* Array(CUDA.sum(cfunc_flux_cont_gpu .* diff(τs_gpu, dims=1), dims=1))'

# plot intensity
grid = plt.matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(grid[0])
ax2 = plt.subplot(grid[1])
ax1.plot(λs_korg, sol.intensity[μ_idx, :], c="k", label="Korg")
ax1.plot(λs_korg, intensity_gpu, ls="--", label="mine")
ax2.scatter(λs_korg, 100 .* (sol.intensity[μ_idx, :] .- intensity_gpu) ./ sol.intensity[μ_idx, :], c="k", s=5)
ax2.set_xlabel("Wavelength (Å)")
ax1.set_ylabel("Emergent Intensity (idk units lol)")
ax2.set_ylabel("Percent Error")
ax1.set_xticklabels([])
ax1.set_title("mu = " * string(sol.mu_grid[μ_idx][1]))
ax1.legend()
ax1.set_xlim(λ0 - 0.25, λ0 + 0.25)
ax2.set_xlim(λ0 - 0.25, λ0 + 0.25)
plt.show()

# plot normalized flux
grid = plt.matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(grid[0])
ax2 = plt.subplot(grid[1])
ax1.plot(λs_korg, sol.flux ./ sol.cntm, c="k", label="Korg")
ax1.plot(λs_korg, flux_gpu ./ flux_cont_gpu, ls="--", label="mine")
# ax1.plot(λs_korg, sol.cntm, c="k", label="Korg cont")
# ax1.plot(λs_korg, flux_cont_gpu, ls="--", label="my cont")
ax2.scatter(λs_korg, 100 .* (sol.flux .- flux_gpu) ./ sol.flux, c="k", s=5)
ax2.set_xlabel("Wavelength (Å)")
ax1.set_ylabel("Emergent Flux (idk units lol)")
ax2.set_ylabel("Percent Error")
ax1.set_xticklabels([])
ax1.legend()
ax1.set_xlim(λ0 - 0.25, λ0 + 0.25)
ax2.set_xlim(λ0 - 0.25, λ0 + 0.25)
plt.savefig("compare_korg.pdf")
plt.show()
