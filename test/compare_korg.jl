let
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Printf
using Statistics

# manually create a single Fe I 6173 line
wl          = 6.17333e-5
log_gf      = -2.880
species     = Korg.Species("Fe I")
E_lower     = 2.223
gamma_rad   = exp10(8.31)
gamma_stark = exp10(-6.16)
gamma_vdw   = log10(exp10(-7.69))
linelist    = [Korg.Line(wl, log_gf, species, E_lower, gamma_rad, gamma_stark, gamma_vdw)]

λ0      = linelist[1].wl * FT.CM_TO_ANGSTROM
λs_korg = collect(range(λ0 - 0.25, λ0 + 0.25, step=0.001))

A_X = Korg.asplund_2009_solar_abundances

# Resample the atmosphere to a uniform log-τ grid and give BOTH sides the same grid.
# Before 2.1 the Atmosphere constructors resampled internally, so `AtmosphereGPU(atm_korg)`
# happened to land on the same grid Korg was handed here. They no longer do (the τ
# integrators consume the native per-interval spacing directly), so the resampled model has
# to be passed in explicitly — otherwise this comparison silently pits FT on the native
# non-uniform grid against Korg on a uniform one.
atm_korg      = Korg.interpolate_marcs(5777, 4.44, A_X)
atm_resampled = FT._resample_log_tau(atm_korg)
atm_gpu       = FT.AtmosphereGPU(atm_resampled)

Natm   = atm_gpu.Natm
Ts_gpu = atm_gpu.Ts_gpu

# microturbulence velocity
val  = 2400.0
v_mic  = CuArray{Float64}(fill(val, Natm))
v_los  = CUDA.zeros(Float64, Natm)

# disk angles (only flux comparison below)
v_losals = 0.2:0.05:1.0

# canonical Korg synthesis on the resampled atmosphere
sol = Korg.synthesize(atm_resampled, linelist, A_X, λs_korg;
                      vmic=val / 1e3, tau_scheme="anchored",
                      mu_values=v_losals, hydrogen_lines=false)

# absorption coefficients via FT
αs      = zeros(Natm, length(λs_korg))
αs_cont = zeros(Natm, length(λs_korg))
α_ref   = zeros(Natm)
# hydrogen_lines=false to match the Korg call above. Inert at this wavelength (Fe I 6173 is
# 389 Å from Hα, outside the 150 Å per-line window) but stated explicitly so the comparison
# stays apples-to-apples if the window ever moves nearer a Balmer line.
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  α_ref_out=α_ref, hydrogen_lines=false)

# convolve absorption with microturbulence
Nλ   = length(λs_korg)
Npad = 100
cmem        = FT.ConvolutionMemory(Nλ, Natm, Npad)
αs_gpu      = FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs,      v_los, v_mic)
cmem        = FT.ConvolutionMemory(Nλ, Natm, Npad)
αs_cont_gpu = FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs_cont, v_los, v_mic)

# anchored-τ reference arrays
log_τ_ref_gpu    = CuArray(log.(atm_gpu.τs))
ifactor_base_gpu = CuArray(atm_gpu.τs ./ α_ref)

λs_gpu              = CuArray(λs_korg)
τs_gpu              = CUDA.zeros(Float64, Natm, Nλ)
cfunc_flux_gpu      = CUDA.zeros(Float64, Natm - 1, Nλ)
cfunc_flux_cont_gpu = CUDA.zeros(Float64, Natm - 1, Nλ)

ts = (32, 16)
bs = (cld(Nλ, ts[1]), cld(Natm, ts[2]))

# flux contribution function
FT.calc_tau_anchored_gpu!(1.0, log_τ_ref_gpu, ifactor_base_gpu, αs_gpu, τs_gpu)
@cuda threads=ts blocks=bs FT.calc_flux_cfunc!(Ts_gpu, λs_gpu, τs_gpu, cfunc_flux_gpu)
CUDA.synchronize()
flux_gpu = 2π .* Array(CUDA.sum(cfunc_flux_gpu .* diff(τs_gpu, dims=1), dims=1))'

# continuum flux contribution function
τs_gpu .= 0.0
FT.calc_tau_anchored_gpu!(1.0, log_τ_ref_gpu, ifactor_base_gpu, αs_cont_gpu, τs_gpu)
@cuda threads=ts blocks=bs FT.calc_flux_cfunc!(Ts_gpu, λs_gpu, τs_gpu, cfunc_flux_cont_gpu)
CUDA.synchronize()
flux_cont_gpu = 2π .* Array(CUDA.sum(cfunc_flux_cont_gpu .* diff(τs_gpu, dims=1), dims=1))'

# percent errors
flux_norm_korg = sol.flux ./ sol.cntm
flux_norm_gpu  = flux_gpu ./ flux_cont_gpu
pct_err_flux   = 100.0 .* (sol.flux .- flux_gpu) ./ sol.flux
pct_err_norm   = 100.0 .* (flux_norm_korg .- flux_norm_gpu) ./ flux_norm_korg

@testset "FormationTemps vs Korg flux comparison" begin
    @test all(α_ref .> 0)
    @test maximum(abs.(pct_err_flux[flux_norm_korg .> 0.99])) < 0.5
    @test maximum(abs.(pct_err_norm)) < 2.0
end

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, height_ratios=[2, 1])
    ax1.plot(λs_korg, flux_norm_korg, c="k", label="{\\rm Korg}")
    ax1.plot(λs_korg, flux_norm_gpu,  ls="--",  label="{\\rm FormationTemps}")
    ax2.scatter(λs_korg, pct_err_norm, c="k", s=5)
    ax2.set_xlabel("{\\rm Wavelength [\\AA]}")
    ax1.set_ylabel("{\\rm Normalized flux}")
    ax2.set_ylabel("{\\rm Percent error}")
    ax1.set_xticklabels([])
    ax1.legend()
    ax1.set_xlim(λ0 - 0.25, λ0 + 0.25)
    ax2.set_xlim(λ0 - 0.25, λ0 + 0.25)
    fig.tight_layout()
    plt.savefig(joinpath(test_plotdir, "compare_korg.pdf"), bbox_inches="tight")
    plt.close()
end

end
