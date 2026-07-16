let
# Validates the GPU quadrature path (method=:quadrature, use_gpu=true) against the CPU
# quadrature (same algorithm, cross-device) and against the explicit tiling reference.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Test

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff, logg, Fe_H = 5777.0, 4.44, 0.0
ζ_RT, ξ = 3400.0, 850.0
Δλ = 0.01
Nϕ = 64

# GPU quad, CPU quad, CPU tiling (ground truth) for one star
function run_trio(; vsini, istar, α₂=0.0, α₄=0.0)
    star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT,
                        v_micro=ξ, istar=istar, α₂=α₂, α₄=α₄)
    rq_gpu = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=true, method=:quadrature,
                                 gpu_precision=Float64, ne_warn_thresh=Inf)
    rq_cpu = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:quadrature,
                                 ne_warn_thresh=Inf)
    rt = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:disk, Nϕ=Nϕ,
                             showprogress=false, ne_warn_thresh=Inf)
    λ0 = mean(rt.wavs)
    edge = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
    n = length(rt.wavs)
    interior = (edge + 1):(n - edge)
    return rq_gpu, rq_cpu, rt, interior
end

@testset "GPU quadrature" begin

    @testset "GPU quad ≈ CPU quad (cross-device, Float64)" begin
        for (vsini, istar, α₂, α₄) in ((0.0, 90.0, 0.0, 0.0),
                                       (15000.0, 90.0, 0.0, 0.0),
                                       (15000.0, 30.0, 0.2, 0.1))
            rq_gpu, rq_cpu, _, interior = run_trio(vsini=vsini, istar=istar, α₂=α₂, α₄=α₄)
            @test length(rq_gpu.wavs) == length(rq_cpu.wavs)
            @test maximum(abs.(rq_gpu.flux .- rq_cpu.flux)) < 1e-8
            @test maximum(abs.(rq_gpu.form_temps[interior] .- rq_cpu.form_temps[interior])) < 1e-2
        end
    end

    @testset "GPU quad ≈ tiling (within quadrature tolerance)" begin
        # non-rotating
        rq_gpu, _, rt, interior = run_trio(vsini=0.0, istar=90.0)
        @test maximum(abs.(rq_gpu.flux .- rt.flux)) < 1e-3
        @test maximum(abs.(rq_gpu.form_temps[interior] .- rt.form_temps[interior])) < 2.0
        # rigid
        rq_gpu, _, rt, interior = run_trio(vsini=15000.0, istar=90.0)
        @test maximum(abs.(rq_gpu.flux .- rt.flux)) < 1e-3
        @test maximum(abs.(rq_gpu.form_temps[interior] .- rt.form_temps[interior])) < 3.0
        # differential + inclined
        rq_gpu, _, rt, interior = run_trio(vsini=15000.0, istar=30.0, α₂=0.2, α₄=0.1)
        @test maximum(abs.(rq_gpu.flux .- rt.flux)) < 1e-3
        @test maximum(abs.(rq_gpu.form_temps[interior] .- rt.form_temps[interior])) < 3.0
    end

    @testset "formation temps within atmosphere T range" begin
        rq_gpu, _, _, _ = run_trio(vsini=15000.0, istar=90.0)
        atm = rq_gpu.atmosphere
        @test all(rq_gpu.form_temps .>= minimum(FT.get_Ts(atm)))
        @test all(rq_gpu.form_temps .<= maximum(FT.get_Ts(atm)))
    end
end

end
