# atmosphere
marcs_atm = Korg.interpolate_marcs(5777.0, 4.44, Korg.asplund_2020_solar_abundances)
atm_cpu = FT.AtmosphereCPU(marcs_atm)
if use_gpu  
    atm_gpu = FT.AtmosphereGPU(marcs_atm)
end

@testset "Testing atmosphere fields" begin
    @test issorted(reverse(get_zs(atm_cpu)))
    @test all(Korg.get_tau_refs(marcs_atm) .== get_τs(atm_cpu))
    @test all(Korg.get_zs(marcs_atm) .== get_zs(atm_cpu))
    @test all(Korg.get_temps(marcs_atm) .== get_Ts(atm_cpu))
end

if use_gpu
    @testset "Testing GPU atmosphere fields" begin
        @test all(Korg.get_tau_refs(marcs_atm) .== get_τs(atm_gpu))
        @test all(Korg.get_zs(marcs_atm) .== get_zs(atm_gpu))
        @test all(Korg.get_temps(marcs_atm) .== get_Ts(atm_gpu))

        @test all(get_τs(atm_cpu) .== get_τs(atm_gpu))
        @test all(get_zs(atm_cpu) .== get_zs(atm_gpu))
        @test all(get_Ts(atm_cpu) .== get_Ts(atm_gpu))
    end
end