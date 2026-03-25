using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Printf

Δλ = 0.005
u1 = 0.43
u2 = 0.31

# match generate_solar.jl inputs
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

star_props = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

result_gpu = FT.calc_formation_temp(star_props, linelist; use_gpu=true,  Δλ=Δλ, convolve=true, u1=u1, u2=u2, ne_warn_thresh=Inf)
result_cpu = FT.calc_formation_temp(star_props, linelist; use_gpu=false, Δλ=Δλ, convolve=true, u1=u1, u2=u2, ne_warn_thresh=Inf)

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=true)
    ax1.plot(result_cpu.wavs, result_gpu.form_temps .- result_cpu.form_temps)
    ax1.set_ylabel("GPU − CPU formation temp [K]")
    ax2.plot(result_cpu.wavs, result_gpu.flux .- result_cpu.flux)
    ax2.set_ylabel("GPU − CPU flux")
    ax2.set_xlabel("Wavelength [Å]")
    fig.savefig(joinpath(test_plotdir, "compare_cpu_gpu_convenience.pdf"), bbox_inches="tight")
    plt.close()
end


@testset "CPU/GPU convenience function agreement (convolve=true)" begin
    @test length(result_gpu.wavs) == length(result_cpu.wavs)
    @test maximum(abs.(result_gpu.wavs .- result_cpu.wavs)) == 0.0
    @test size(result_gpu.cont_func) == size(result_cpu.cont_func)
    # GPU uses analytical Fourier-domain Gaussian for microturbulence; CPU uses sampled
    # real-space kernel.  At ξ ≈ 850 m/s (σ ≈ 1.8 px) the max difference is ~4e-4.
    @test maximum(abs.(result_gpu.flux .- result_cpu.flux))       < 1e-3
    @test mean(abs.(result_gpu.flux .- result_cpu.flux))          < 1e-4
    # Hirano CPU uses circular FFT; GPU uses padded linear convolution. They differ at
    # the spectrum edges (first/last ~3ζ_RT/c * λ0/Δλ pixels). Compare only the interior.
    λ0_val = mean(result_cpu.wavs)
    edge_px = ceil(Int, 3 * ζ_RT / FT.c_ms * λ0_val / Δλ) + 10
    interior = (edge_px+1):(length(result_cpu.wavs) - edge_px)
    @test maximum(abs.(result_gpu.form_temps[interior] .- result_cpu.form_temps[interior])) < 1.0
    @test mean(abs.(result_gpu.form_temps[interior] .- result_cpu.form_temps[interior]))    < 0.5
    # formation temperatures should be within the atmospheric temperature range
    atm_cpu = result_cpu.atmosphere
    T_min = minimum(FT.get_Ts(atm_cpu))
    T_max = maximum(FT.get_Ts(atm_cpu))
    @test all(result_cpu.form_temps .>= T_min)
    @test all(result_cpu.form_temps .<= T_max)
end
