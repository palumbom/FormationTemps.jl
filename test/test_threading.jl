let
    # short linelist for fast tests
    linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
    linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

    Δλ = 0.01
    Teff = 5777.0
    logg = 4.44
    Fe_H = 0.0
    vsini = 2100.0
    ζ_RT = 3400.0
    ξ = 850.0

    @testset "CPU threaded disk integration" begin
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini,
                            v_macro=ζ_RT, v_micro=ξ)

        @testset "Repeatability" begin
            r1 = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=8,
                                      use_gpu=false, convolve=false,
                                      showprogress=false, ne_warn_thresh=Inf)
            r2 = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=8,
                                      use_gpu=false, convolve=false,
                                      showprogress=false, ne_warn_thresh=Inf)

            @test r1.flux ≈ r2.flux
            @test r1.form_temps ≈ r2.form_temps
        end

        @testset "Flux bounded by unity" begin
            result = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=8,
                                          use_gpu=false, convolve=false,
                                          showprogress=false, ne_warn_thresh=Inf)
            @test maximum(result.flux) <= 1.0 + eps(Float32)
        end

        @testset "Formation temps within atmosphere range" begin
            result = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=8,
                                          use_gpu=false, convolve=false,
                                          showprogress=false, ne_warn_thresh=Inf)
            atm = result.atmosphere
            T_min = minimum(FT.get_Ts(atm))
            T_max = maximum(FT.get_Ts(atm))
            @test all(result.form_temps .>= T_min)
            @test all(result.form_temps .<= T_max)
        end

        @testset "Disk integration with zero broadening" begin
            # vsini=0, ζ=0: disk integration with no broadening should still produce
            # physically valid results
            star_stat = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=0.0,
                                      v_macro=0.0, v_micro=ξ)

            result = calc_formation_temp(star_stat, linelist; Δλ=Δλ, Nϕ=8,
                                          use_gpu=false, convolve=false,
                                          showprogress=false, ne_warn_thresh=Inf)

            @test maximum(result.flux) <= 1.0 + eps(Float32)
            atm = result.atmosphere
            T_min = minimum(FT.get_Ts(atm))
            T_max = maximum(FT.get_Ts(atm))
            @test all(result.form_temps .>= T_min)
            @test all(result.form_temps .<= T_max)
        end
    end
end
