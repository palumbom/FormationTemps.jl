using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Test

# short linelist for speed
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0
Δλ = 0.01
Nϕ = 16

@testset "Threaded CPU vs GPU disk integration" begin
    star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini,
                        v_macro=ζ_RT, v_micro=ξ)

    result_cpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                      use_gpu=false, convolve=false,
                                      showprogress=false, ne_warn_thresh=Inf)
    result_gpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                      use_gpu=true, convolve=false,
                                      showprogress=false, ne_warn_thresh=Inf)

    @test length(result_gpu.wavs) == length(result_cpu.wavs)

    # flux agreement (CPU microturbulence uses sampled kernel; GPU uses analytical
    # Fourier-domain Gaussian — systematic difference ~4e-4 at ξ ≈ 850 m/s)
    @test maximum(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-3
    @test mean(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-4

    # formation temperatures: exclude edges where CPU circular FFT and GPU padded
    # linear convolution diverge
    λ0_val = mean(result_cpu.wavs)
    edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0_val)) + 10
    interior = (edge_px+1):(length(result_cpu.wavs) - edge_px)
    if length(interior) > 10
        @test maximum(abs.(result_gpu.form_temps[interior] .- result_cpu.form_temps[interior])) < 5.0
    end

    # contribution function shapes should agree
    @test size(result_gpu.cont_func) == size(result_cpu.cont_func)

    # sanity: formation temps within atmosphere range
    atm = result_cpu.atmosphere
    T_min = minimum(FT.get_Ts(atm))
    T_max = maximum(FT.get_Ts(atm))
    @test all(result_cpu.form_temps .>= T_min)
    @test all(result_cpu.form_temps .<= T_max)
    @test all(result_gpu.form_temps .>= T_min)
    @test all(result_gpu.form_temps .<= T_max)
end
