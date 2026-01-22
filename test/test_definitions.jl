@testset "Testing atmosphere type hierarchy" begin
    @test isdefined(FormationTemps, :Atmosphere)
    @test isdefined(FormationTemps, :AtmosphereGPU)
    @test isdefined(FormationTemps, :AtmosphereCPU)
    @test FT.AtmosphereCPU <: FT.Atmosphere
    @test FT.AtmosphereGPU <: FT.Atmosphere
end