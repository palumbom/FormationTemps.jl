let
# Tests that the dual-stream GPU path (total + continuum on separate CUDA streams)
# produces results consistent with the CPU reference, for both ζ=0 and ζ≠0.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Test

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
Fe_H = 0.0
ξ = 850.0
Δλ = 0.01
Nϕ = 16

@testset "Dual-stream GPU agreement" begin
    @testset "ζ = 0 (no macro convolution)" begin
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=2100.0,
                            v_macro=0.0, v_micro=ξ)

        result_cpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                          use_gpu=false, convolve=false,
                                          showprogress=false, ne_warn_thresh=Inf)
        result_gpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                          use_gpu=true, convolve=false,
                                          showprogress=false, ne_warn_thresh=Inf)

        @test maximum(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-3
        @test mean(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-4

        λ0_val = mean(result_cpu.wavs)
        edge_px = ceil(Int, 2100.0 * 3 / (FT.c_ms * Δλ / λ0_val)) + 10
        interior = (edge_px+1):(length(result_cpu.wavs) - edge_px)
        if length(interior) > 10
            @test maximum(abs.(result_gpu.form_temps[interior] .- result_cpu.form_temps[interior])) < 5.0
        end
    end

    @testset "ζ ≠ 0 (with macro convolution on both streams)" begin
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=2100.0,
                            v_macro=3400.0, v_micro=ξ)

        result_cpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                          use_gpu=false, convolve=false,
                                          showprogress=false, ne_warn_thresh=Inf)
        result_gpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                          use_gpu=true, convolve=false,
                                          showprogress=false, ne_warn_thresh=Inf)

        @test maximum(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-3
        @test mean(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-4

        λ0_val = mean(result_cpu.wavs)
        edge_px = ceil(Int, max(2100.0, 3400.0) * 3 / (FT.c_ms * Δλ / λ0_val)) + 10
        interior = (edge_px+1):(length(result_cpu.wavs) - edge_px)
        if length(interior) > 10
            @test maximum(abs.(result_gpu.form_temps[interior] .- result_cpu.form_temps[interior])) < 5.0
        end
    end

    @testset "GPU determinism (two runs produce identical results)" begin
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=2100.0,
                            v_macro=3400.0, v_micro=ξ)

        result1 = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                       use_gpu=true, convolve=false,
                                       showprogress=false, ne_warn_thresh=Inf)
        result2 = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=Nϕ,
                                       use_gpu=true, convolve=false,
                                       showprogress=false, ne_warn_thresh=Inf)

        @test result1.flux == result2.flux
        @test result1.form_temps == result2.form_temps
        @test result1.cont_func == result2.cont_func
    end
end

end
