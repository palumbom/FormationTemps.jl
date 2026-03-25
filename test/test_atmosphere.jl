marcs_atm = Korg.interpolate_marcs(5777.0, 4.44, Korg.asplund_2020_solar_abundances)
atm_cpu = FT.AtmosphereCPU(marcs_atm)
if use_gpu
    atm_gpu = FT.AtmosphereGPU(marcs_atm)
end

τs_orig = Korg.get_tau_refs(marcs_atm)

@testset "Atmosphere structure and resampling" begin
    τs = get_τs(atm_cpu)
    zs = get_zs(atm_cpu)
    Ts = get_Ts(atm_cpu)

    # Natm field is consistent with array lengths; layer count preserved
    @test atm_cpu.Natm == length(τs) == length(zs) == length(Ts)
    @test atm_cpu.Natm == length(τs_orig)

    # physical positivity
    @test all(τs .> 0)
    @test all(Ts .> 0)

    # physical ordering: τ increases with depth, z decreases with depth, T increases with depth
    @test issorted(τs)
    @test issorted(reverse(zs))
    @test issorted(Ts)

    # τ range endpoints match the original atmosphere (resampling preserves bounds)
    @test isapprox(first(τs), first(τs_orig))
    @test isapprox(last(τs), last(τs_orig))

    # log-τ spacing is uniform after resampling (max/min step ratio near 1)
    log_τ = log.(τs)
    step_ratio = maximum(diff(log_τ)) / minimum(diff(log_τ))
    @test step_ratio < 1.05

    # reference wavelength is passed through unchanged
    @test atm_cpu.reference_wavelength == marcs_atm.reference_wavelength
end

if use_gpu
    @testset "GPU atmosphere matches CPU atmosphere" begin
        # CPU and GPU both resample from the same source, so fields must be identical
        @test get_τs(atm_cpu) == get_τs(atm_gpu)
        @test get_zs(atm_cpu) == get_zs(atm_gpu)
        @test get_Ts(atm_cpu) == get_Ts(atm_gpu)
        @test atm_gpu.Natm    == atm_cpu.Natm
        @test atm_gpu.reference_wavelength == atm_cpu.reference_wavelength
    end
end
