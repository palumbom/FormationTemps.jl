let
    marcs_atm = Korg.interpolate_marcs(5777.0, 4.44, Korg.asplund_2020_solar_abundances)
    atm_cpu = FT.AtmosphereCPU(marcs_atm)
    if use_gpu
        atm_gpu = FT.AtmosphereGPU(marcs_atm)
    end

    τs_orig = Korg.get_tau_refs(marcs_atm)
    zs_orig = Korg.get_zs(marcs_atm)
    Ts_orig = Korg.get_temps(marcs_atm)

    @testset "Atmosphere structure (native grid)" begin
        τs = get_τs(atm_cpu)
        zs = get_zs(atm_cpu)
        Ts = get_Ts(atm_cpu)

        # Natm field is consistent with array lengths; native layer count preserved
        @test atm_cpu.Natm == length(τs) == length(zs) == length(Ts)
        @test atm_cpu.Natm == length(τs_orig)

        # physical positivity
        @test all(τs .> 0)
        @test all(Ts .> 0)

        # physical ordering: τ increases with depth, z decreases with depth, T increases with depth
        @test issorted(τs)
        @test issorted(reverse(zs))
        @test issorted(Ts)

        # the native MARCS grid is used as-is: fields match the source atmosphere exactly
        @test τs == τs_orig
        @test zs == zs_orig
        @test Ts == Ts_orig

        # regression guard: the native solar grid really is non-uniform in log τ. The τ
        # integrators are spacing-aware, as Korg's are, so the grid is not resampled.
        log_τ = log.(τs)
        @test maximum(diff(log_τ)) / minimum(diff(log_τ)) > 1.05

        # reference wavelength is passed through unchanged
        @test atm_cpu.reference_wavelength == marcs_atm.reference_wavelength
    end

    @testset "_resample_log_tau utility (opt-in uniform resampling)" begin
        # The constructor no longer resamples, but _resample_log_tau remains available as an
        # explicit tool (e.g. convergence studies, flux_vs_intensity.jl). Called directly it
        # produces a uniform log-τ grid and preserves the τ bounds.
        resampled = FT._resample_log_tau(marcs_atm)
        τs_r = Korg.get_tau_refs(resampled)
        log_τr = log.(τs_r)
        @test maximum(diff(log_τr)) / minimum(diff(log_τr)) < 1.05     # uniform
        @test isapprox(first(τs_r), first(τs_orig))                    # bounds preserved
        @test isapprox(last(τs_r), last(τs_orig))
        @test length(τs_r) == length(τs_orig)                          # default preserves count

        # explicit upsampling changes the layer count
        up = FT._resample_log_tau(marcs_atm; n_layers = 2 * length(τs_orig))
        @test length(Korg.get_tau_refs(up)) == 2 * length(τs_orig)
    end

    if use_gpu
        @testset "GPU atmosphere matches CPU atmosphere" begin
            # CPU and GPU both use the same native grid, so fields must be identical
            @test get_τs(atm_cpu) == get_τs(atm_gpu)
            @test get_zs(atm_cpu) == get_zs(atm_gpu)
            @test get_Ts(atm_cpu) == get_Ts(atm_gpu)
            @test atm_gpu.Natm    == atm_cpu.Natm
            @test atm_gpu.reference_wavelength == atm_cpu.reference_wavelength
        end
    end
end
