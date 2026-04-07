@testset "Testing atmosphere type hierarchy" begin
    @test isdefined(FormationTemps, :Atmosphere)
    @test isdefined(FormationTemps, :AtmosphereGPU)
    @test isdefined(FormationTemps, :AtmosphereCPU)
    @test FT.AtmosphereCPU <: FT.Atmosphere
    @test FT.AtmosphereGPU <: FT.Atmosphere
end

@testset "Public types" begin
    @test isdefined(FormationTemps, :StellarProps)
    @test isdefined(FormationTemps, :FormTempResult)
    @test isdefined(FormationTemps, :ConvolutionMemory)
    @test isdefined(FormationTemps, :GPUMemory)
    @test isdefined(FormationTemps, :AlphaCache)
end

@testset "Public functions" begin
    @test isdefined(FormationTemps, :calc_formation_temp)
    @test isdefined(FormationTemps, :compute_alpha!)
    @test isdefined(FormationTemps, :calc_tau_anchored_cpu!)
    @test isdefined(FormationTemps, :calc_flux_cfunc_cpu!)
end

@testset "Constants" begin
    @test isapprox(FT.c_ms, 2.99792458e8)
    @test isdefined(FormationTemps, :GPU_DEFAULT)
    @test FT.GPU_DEFAULT isa Bool
end

@testset "StellarProps constructor" begin
    # NaN v_macro and v_micro trigger empirical fits (non-NaN after construction)
    star = FT.StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0)
    @test star.Teff == 5777.0
    @test star.logg == 4.44
    @test star.Fe_H == 0.0
    @test !isnan(star.ζ)
    @test !isnan(star.ξ)
    @test star.ζ > 0
    @test star.ξ > 0

    # explicit velocities are stored verbatim
    star2 = FT.StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=1500.0, v_macro=2000.0, v_micro=800.0)
    @test star2.vsini == 1500.0
    @test star2.ζ == 2000.0
    @test star2.ξ == 800.0

    # vsini=0 default
    @test star.vsini == 0.0

    # vector v_micro
    v_mic_vec = collect(range(600.0, 1200.0, length=56))
    star_vec = FT.StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, v_micro=v_mic_vec)
    @test star_vec.ξ == v_mic_vec
    @test star_vec.ξ isa Vector{Float64}

    # NaN default still triggers vmic_fit (scalar only)
    star_fit = FT.StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0)
    @test star_fit.ξ isa Float64
    @test !isnan(star_fit.ξ)
end
