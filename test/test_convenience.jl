use_gpu = FT.GPU_DEFAULT

# Use the same linelist slice as scripts/generate_solar.jl
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16025]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

# wavelength grid
Δλ = 0.01
wls = [l.wl * 1e8 for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)

# set parameters for synthesis
convolve = true
u1 = 0.43
u2 = 0.31
Nϕ = 16

# set stellar params
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 0.0
ζ_RT = 0.0
ξ = 850.0
star_stat = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# get stationary convenience flux
result_stationary_convenience = calc_formation_temp(star_stat, linelist, use_gpu=false,
                                                    u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                                    ne_warn_thresh=Inf)
if use_gpu
    result_stationary_convenience_gpu = calc_formation_temp(star_stat, linelist, use_gpu=true,
                                                            u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                                            ne_warn_thresh=Inf)
end

# get model atmosphere
atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(Teff, logg, star_stat.A_X))
zs = atm_cpu.zs
Ts = atm_cpu.Ts

# get the absorption coefficients
T = Float64
Natm = length(zs)
α_ref = zeros(T, Natm)
Nλ = length(λs_korg)
αs = zeros(T, Natm, Nλ)
αs_cont = zeros(T, Natm, Nλ)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_cpu, star_stat.A_X;
                  α_ref_out=α_ref, vmic_ref_cms=star_stat.ξ * 100.0, ne_warn_thresh=Inf)

# set microturbulent broadening
σ_v = fill(ξ, Natm)
μ_v = zeros(T, Natm)

# get the absorption coefficients
αs_broad = FT.convolve_wavelength_axis(λs_korg, αs, μ_v, σ_v)
αs_cont_broad = FT.convolve_wavelength_axis(λs_korg, αs_cont, μ_v, σ_v)

# get the taus
τs = zeros(T, Natm, Nλ)
τs_cont = zeros(T, Natm, Nλ)
FT.calc_tau_anchored_cpu!(one(T), atm_cpu.τs, α_ref, αs_broad, τs)
FT.calc_tau_anchored_cpu!(one(T), atm_cpu.τs, α_ref, αs_cont_broad, τs_cont)

# get the contribution function
cfunc_flux = zeros(T, Natm - 1, Nλ)
cfunc_flux_cont = zeros(T, Natm - 1, Nλ)
FT.calc_flux_cfunc_cpu!(cfunc_flux, Ts, λs_korg, τs)
FT.calc_flux_cfunc_cpu!(cfunc_flux_cont, Ts, λs_korg, τs_cont)

cfunc_dt_flux = cfunc_flux .* diff(τs, dims=1)
cfunc_dt_flux_cont = cfunc_flux_cont .* diff(τs_cont, dims=1)

# get the flux
flux_norm = vec(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))

@testset "Testing dimensions" begin
    @test size(αs) == size(αs_broad) == size(αs_cont) == size(αs_cont_broad)
    @test size(result_stationary_convenience.cont_func) == (length(zs)-1, length(λs_korg))
    @test size(cfunc_dt_flux_cont) == size(result_stationary_convenience.cont_func)
end

@testset "Testing stationary flux" begin
    @test maximum(flux_norm) .<= (one(T) .+ eps(Float32))
    @test maximum(result_stationary_convenience.flux) .<= (one(T) .+ eps(Float32))
    @test all(isapprox.(result_stationary_convenience.flux, flux_norm))
end

# now do non-stationary flux
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0
star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# get stationary convenience flux
result_convenience = calc_formation_temp(star, linelist, use_gpu=false,
                                         u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                         ne_warn_thresh=Inf)
if use_gpu
    result_convenience_gpu = calc_formation_temp(star, linelist, use_gpu=true,
                                                 u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                                 ne_warn_thresh=Inf)
end

# convolve manually
cfunc_dt_flux = FT.convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux, star.vsini, star.ζ, u1, u2)
cfunc_dt_flux_cont = FT.convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux_cont, star.vsini, star.ζ, u1, u2)
flux_rotating = vec(Array(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1)))

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
    plt.ioff()
    fig, axes = plt.subplots(2, 1, sharex=true, figsize=(10, 6))
    axes[0].plot(collect(λs_korg), result_stationary_convenience.flux, label="{\\rm stationary}")
    axes[0].plot(collect(λs_korg), result_convenience.flux, label="{\\rm vsini=$(vsini), zeta=$(ζ_RT)}")
    axes[0].set_ylabel("{\\rm Normalized flux}")
    axes[0].legend()
    axes[1].plot(collect(λs_korg), result_stationary_convenience.form_temps, label="{\\rm stationary}")
    axes[1].plot(collect(λs_korg), result_convenience.form_temps, label="{\\rm vsini=$(vsini), zeta=$(ζ_RT)}")
    axes[1].set_ylabel("{\\rm Formation temp [K]}")
    axes[1].set_xlabel("{\\rm Wavelength [\\AA]}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(joinpath(test_plotdir, "test_convenience.pdf"), bbox_inches="tight")
    plt.close()
end


@testset "Testing convolved convenience flux" begin
    @test maximum(flux_rotating) .<= (one(T) .+ eps(Float32))
    @test maximum(result_convenience.flux) .<= (one(T) .+ eps(Float32))
    @test all(isapprox.(result_convenience.flux, flux_rotating))
end

if use_gpu
    @testset "Testing GPU convolved convenience flux" begin
        # GPU uses an analytical Fourier-domain Gaussian for microturbulence while CPU uses a
        # sampled real-space kernel; at ξ ≈ 850 m/s (σ ≈ 1.8 pixels) the difference is ~4e-4.
        @test maximum(abs.(result_stationary_convenience.flux .- result_stationary_convenience_gpu.flux)) < 1e-3
        @test maximum(abs.(result_convenience.flux .- result_convenience_gpu.flux)) < 1e-3
    end
end
