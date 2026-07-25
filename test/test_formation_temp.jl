@testset "Formation temperature extraction (node-anchored CDF)" begin
    # Analytic 4-node case. Contribution 0.2/0.6/0.2 over intervals spanning
    # 4000–5000, 5000–6000, 6000–7000 K. The 50% point lies at the center of
    # interval 2 (which holds 60% of the flux, spanning 5000–6000 K) → 5500 K.
    # The previous center-paired convention returned 5000 K (half-interval cool).
    Ts    = [4000.0, 5000.0, 6000.0, 7000.0]
    cfunc = reshape([0.2, 0.6, 0.2], 3, 1)
    ft    = form_temps_from_cfunc(cfunc, Ts)
    @test length(ft) == 1
    @test isapprox(ft[1], 5500.0; atol=1e-9)

    # result stays within the node temperature range (never inside mid-interval range)
    @test all(ft .>= minimum(Ts))
    @test all(ft .<= maximum(Ts))

    # symmetric contribution whose median lands on a node returns that node's temperature
    Ts_sym = [3000.0, 4000.0, 5000.0, 6000.0, 7000.0]
    c_sym  = reshape([0.1, 0.4, 0.4, 0.1], 4, 1)     # cumulative hits 0.5 at node Ts_sym[3]
    @test isapprox(form_temps_from_cfunc(c_sym, Ts_sym)[1], 5000.0; atol=1e-9)

    # multiple wavelength columns handled independently
    @test form_temps_from_cfunc(hcat(cfunc, cfunc), Ts) ≈ [5500.0, 5500.0]

    # normalization is invariant to overall scale (per-column normalization)
    @test form_temps_from_cfunc(1e-8 .* cfunc, Ts) ≈ ft

    # dimension mismatch (cfunc rows must be length(Ts)-1) is caught
    @test_throws AssertionError form_temps_from_cfunc(reshape([0.5, 0.5], 2, 1), Ts)

    # Float32 support (GPU precision path passes host copies through the same helper)
    ft32 = form_temps_from_cfunc(Float32.(cfunc), Float32.(Ts))
    @test eltype(ft32) == Float32
    @test isapprox(ft32[1], 5500.0f0; atol=1e-2)
end

@testset "Formation temperature degenerate cases are reported, not silent" begin
    # Two failure modes have no meaningful median; both must yield NaN and warn. These are
    # diagnostics, which rot quietly if untested: a refactor that stopped emitting them
    # would otherwise go unnoticed.
    Ts = [4000.0, 5000.0, 6000.0, 7000.0]

    @testset "zero total contribution → NaN + warning" begin
        # reachable: the microturbulence underflow guard zeros a row when the
        # Doppler shift moves the kernel out of the window, which can zero a whole column
        empty = reshape([0.0, 0.0, 0.0], 3, 1)
        ft = @test_logs (:warn, r"no positive total") match_mode=:any begin
            form_temps_from_cfunc(empty, Ts)
        end
        @test length(ft) == 1
        @test isnan(ft[1])
    end

    @testset "negative total contribution → NaN, not the deepest temperature" begin
        # Float32 FFT roundoff on a near-zero column can make the total slightly negative.
        # That inverts the CDF, and linear_interp's `x >= last(xs)` branch would silently
        # return Ts[end] — a hot, confident, wrong answer. Guard against exactly that.
        neg = reshape([-1e-20, -1e-20, -1e-20], 3, 1)
        ft = @test_logs (:warn, r"no positive total") match_mode=:any begin
            form_temps_from_cfunc(neg, Ts)
        end
        @test isnan(ft[1])
        @test ft[1] != last(Ts)          # the specific wrong answer this guard prevents
    end

    @testset "a degenerate column does not contaminate its neighbours" begin
        good = [0.2, 0.6, 0.2]
        mixed = hcat(good, zeros(3), good)
        ft = @test_logs (:warn, r"no positive total") match_mode=:any begin
            form_temps_from_cfunc(mixed, Ts)
        end
        @test isapprox(ft[1], 5500.0; atol=1e-9)
        @test isnan(ft[2])
        @test isapprox(ft[3], 5500.0; atol=1e-9)
    end
end

@testset "Boundary contamination statistic and mask" begin
    # A formation temperature is only meaningful if the flux contribution has decayed before
    # the model atmosphere is truncated. `ceiling_ratio` measures that (top interval as a
    # fraction of the column peak) and `boundary_mask` thresholds it. Both are core outputs so
    # callers do not each reinvent the criterion.
    Ts = [4000.0, 5000.0, 6000.0, 7000.0]

    @testset "ceiling_ratio" begin
        @test ceiling_ratio(reshape([0.2, 0.6, 0.2], 3, 1)) ≈ [0.2 / 0.6]
        @test ceiling_ratio(reshape([0.9, 0.05, 0.05], 3, 1)) ≈ [1.0]      # peak IS the top
        @test ceiling_ratio(reshape([0.05, 0.9, 0.05], 3, 1)) ≈ [0.05 / 0.9]

        # columns are independent
        @test ceiling_ratio(hcat([0.9, 0.05, 0.05], [0.05, 0.9, 0.05])) ≈ [1.0, 0.05 / 0.9]

        # scale-invariant, so it is comparable across lines of different depth
        c = reshape([0.2, 0.6, 0.2], 3, 1)
        @test ceiling_ratio(1e-8 .* c) ≈ ceiling_ratio(c)

        # an all-zero column is 0, not NaN; those are reported separately as NaN form_temps
        @test ceiling_ratio(reshape(zeros(3), 3, 1)) == [0.0]

        @test eltype(ceiling_ratio(Float32.(c))) == Float32
    end

    @testset "boundary_mask thresholds it" begin
        just_over  = reshape([0.34, 1.0, 0.1], 3, 1)     # ratio 0.34
        just_under = reshape([0.32, 1.0, 0.1], 3, 1)     # ratio 0.32
        @test FT.BOUNDARY_R_THRESH == 0.33
        @test boundary_mask(just_over)  == [true]
        @test boundary_mask(just_under) == [false]
        # threshold is honoured in both directions
        @test boundary_mask(just_over;  r_thresh=0.5) == [false]
        @test boundary_mask(just_under; r_thresh=0.2) == [true]
    end

    @testset "the warning fires iff any(boundary_mask)" begin
        # the consolidation invariant: form_temps_from_cfunc must never flag a different set
        # of wavelengths than the mask a caller applies downstream
        for c in (reshape([0.34, 1.0, 0.1], 3, 1),      # flagged, just over
                  reshape([0.32, 1.0, 0.1], 3, 1),      # not flagged, just under
                  reshape([0.9, 0.05, 0.05], 3, 1),     # flagged, peak at ceiling
                  reshape([0.05, 0.9, 0.05], 3, 1))     # not flagged, well resolved
            logger = Test.TestLogger(respect_maxlog=false)
            Base.CoreLogging.with_logger(logger) do
                form_temps_from_cfunc(c, Ts)
            end
            warned = any(l -> occursin("peaking at the top", string(l.message)), logger.logs)
            @test warned == any(boundary_mask(c))
        end
    end

    @testset "the warning reports the same count as the mask" begin
        c = hcat([0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.34, 1.0, 0.1], [0.32, 1.0, 0.1])
        logger = Test.TestLogger(respect_maxlog=false)
        Base.CoreLogging.with_logger(logger) do
            form_temps_from_cfunc(c, Ts)
        end
        msg = only(filter(m -> occursin("peaking at the top", m),
                          [string(l.message) for l in logger.logs]))
        n_reported = parse(Int, match(r"(\d+) of \d+ wavelengths", msg).captures[1])
        @test n_reported == count(boundary_mask(c)) == 2
    end

    @testset "warn_boundary=false silences it without changing the value" begin
        pinned = reshape([0.9, 0.05, 0.05], 3, 1)
        ft = @test_logs (:warn, r"peaking at the top") match_mode=:any begin
            form_temps_from_cfunc(pinned, Ts)
        end
        # crossing at F=0.5 within a first interval holding 0.9 → 4000 + 1000*(0.5/0.9)
        @test isapprox(ft[1], 4000.0 + 1000.0 * (0.5 / 0.9); atol=1e-9)
        @test Ts[1] <= ft[1] <= Ts[2]

        quiet = @test_logs min_level=Base.CoreLogging.Warn begin
            form_temps_from_cfunc(pinned, Ts; warn_boundary=false)
        end
        @test quiet == ft
    end
end

@testset "FormTempResult carries a consistent ceiling_ratio" begin
    # the five-argument constructor derives the field, so it cannot disagree with cont_func
    Ts = [4000.0, 5000.0, 6000.0, 7000.0]
    cf = hcat([0.9, 0.05, 0.05], [0.05, 0.9, 0.05])
    atm = FT.AtmosphereCPU(Korg.interpolate_marcs(5777.0, 4.44, Korg.format_A_X(0.0)))
    res = FormTempResult([1.0, 2.0], [1.0, 1.0], [5000.0, 6000.0], cf, atm)
    @test res.ceiling_ratio == ceiling_ratio(cf)
    @test ceiling_ratio(res) == ceiling_ratio(cf)
    @test boundary_mask(res) == boundary_mask(cf)
    @test boundary_mask(res; r_thresh=0.9) == (ceiling_ratio(cf) .> 0.9)
end
