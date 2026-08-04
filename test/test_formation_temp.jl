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
    # Δlog₁₀τ_ref = 1 exactly, so the per-dex density equals the per-interval integrals and
    # these ratios are read directly off the numbers below. See test_cfunc_measure.jl for the
    # non-uniform case, which is where the measure matters.
    τ_unif = 10.0 .^ [-3.0, -2.0, -1.0, 0.0]

    @testset "ceiling_ratio" begin
        @test ceiling_ratio(reshape([0.2, 0.6, 0.2], 3, 1), τ_unif) ≈ [0.2 / 0.6]
        @test ceiling_ratio(reshape([0.9, 0.05, 0.05], 3, 1), τ_unif) ≈ [1.0]  # peak IS the top
        @test ceiling_ratio(reshape([0.05, 0.9, 0.05], 3, 1), τ_unif) ≈ [0.05 / 0.9]

        # columns are independent
        @test ceiling_ratio(hcat([0.9, 0.05, 0.05], [0.05, 0.9, 0.05]), τ_unif) ≈
            [1.0, 0.05 / 0.9]

        # scale-invariant, so it is comparable across lines of different depth
        c = reshape([0.2, 0.6, 0.2], 3, 1)
        @test ceiling_ratio(1e-8 .* c, τ_unif) ≈ ceiling_ratio(c, τ_unif)

        # an all-zero column is 0, not NaN; those are reported separately as NaN form_temps
        @test ceiling_ratio(reshape(zeros(3), 3, 1), τ_unif) == [0.0]

        @test eltype(ceiling_ratio(Float32.(c), Float32.(τ_unif))) == Float32
    end

    @testset "boundary_mask thresholds it" begin
        just_over  = reshape([0.34, 1.0, 0.1], 3, 1)     # ratio 0.34
        just_under = reshape([0.32, 1.0, 0.1], 3, 1)     # ratio 0.32
        @test FT.BOUNDARY_R_THRESH == 0.33
        @test boundary_mask(just_over, τ_unif)  == [true]
        @test boundary_mask(just_under, τ_unif) == [false]
        # threshold is honoured in both directions
        @test boundary_mask(just_over, τ_unif;  r_thresh=0.5) == [false]
        @test boundary_mask(just_under, τ_unif; r_thresh=0.2) == [true]
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
                form_temps_from_cfunc(c, Ts; τ_ref=τ_unif)
            end
            warned = any(l -> occursin("peaking at the top", string(l.message)), logger.logs)
            @test warned == any(boundary_mask(c, τ_unif))
        end
    end

    @testset "the warning reports the same count as the mask" begin
        c = hcat([0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.34, 1.0, 0.1], [0.32, 1.0, 0.1])
        logger = Test.TestLogger(respect_maxlog=false)
        Base.CoreLogging.with_logger(logger) do
            form_temps_from_cfunc(c, Ts; τ_ref=τ_unif)
        end
        msg = only(filter(m -> occursin("peaking at the top", m),
                          [string(l.message) for l in logger.logs]))
        n_reported = parse(Int, match(r"(\d+) of \d+ wavelengths", msg).captures[1])
        @test n_reported == count(boundary_mask(c, τ_unif)) == 2
    end

    @testset "warn_boundary=false silences it without changing the value" begin
        pinned = reshape([0.9, 0.05, 0.05], 3, 1)
        ft = @test_logs (:warn, r"peaking at the top") match_mode=:any begin
            form_temps_from_cfunc(pinned, Ts; τ_ref=τ_unif)
        end
        # crossing at F=0.5 within a first interval holding 0.9 → 4000 + 1000*(0.5/0.9)
        @test isapprox(ft[1], 4000.0 + 1000.0 * (0.5 / 0.9); atol=1e-9)
        @test Ts[1] <= ft[1] <= Ts[2]

        quiet = @test_logs min_level=Base.CoreLogging.Warn begin
            form_temps_from_cfunc(pinned, Ts; τ_ref=τ_unif, warn_boundary=false)
        end
        @test quiet == ft
    end
end

@testset "FormTempResult carries a consistent ceiling_ratio and threshold" begin
    # the five-argument constructor derives ceiling_ratio, so it cannot disagree with
    # cont_func, and records the threshold so boundary_mask reproduces the warned set
    cf = hcat([0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.4, 1.0, 0.1])
    # 4-layer stand-in whose τ_ref matches cf's row count, with Δlog₁₀τ_ref = 1 so the
    # asserted ratios are read straight off cf. A real MARCS atmosphere has 56 layers and
    # would not pair with a 3-row contribution function.
    n = size(cf, 1) + 1
    atm = FT.AtmosphereCPU(n, 10.0 .^ collect(range(-(n - 1.0), 0.0, length=n)),
                           zeros(n), [4e3, 5e3, 6e3, 7e3], zeros(n), zeros(n), 5e-5,
                           zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))
    mk(; kw...) = FormTempResult([1.0, 2.0, 3.0], ones(3), [5e3, 6e3, 7e3], cf, atm; kw...)

    res = mk()
    @test res.ceiling_ratio == ceiling_ratio(cf, atm.τs)
    @test ceiling_ratio(res) == ceiling_ratio(cf, atm.τs)
    @test res.r_thresh == FT.BOUNDARY_R_THRESH
    @test boundary_mask(res) == boundary_mask(cf, atm.τs)

    # the recorded threshold is what boundary_mask defaults to
    res_hi = mk(r_thresh=0.5)
    @test res_hi.r_thresh == 0.5
    @test boundary_mask(res_hi) == (ceiling_ratio(cf, atm.τs) .> 0.5)
    @test boundary_mask(res_hi) != boundary_mask(res)      # 0.4 column flips
    # and an explicit threshold still overrides it
    @test boundary_mask(res_hi; r_thresh=0.9) == (ceiling_ratio(cf, atm.τs) .> 0.9)
end

@testset "r_thresh reaches calc_formation_temp's warning and result" begin
    # the whole point of plumbing it: a non-default threshold must drive BOTH the warning
    # and the mask, so they cannot describe different wavelengths
    linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
    linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
    star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=0.0,
                        v_macro=0.0, v_micro=850.0)
    # not named `run`: this file is included into Main, and shadowing Base.run there would
    # leak into every test file included after it
    synth(; kw...) = calc_formation_temp(star, linelist; Δλ=0.05, use_gpu=false,
                                        method=:quadrature, Nμ=8, ne_warn_thresh=Inf, kw...)

    r_default = synth()
    @test r_default.r_thresh == FT.BOUNDARY_R_THRESH

    # a threshold of 0 flags everything with any top-layer contribution; 1.0 flags nothing,
    # since ceiling_ratio is at most 1 by construction
    r_all  = synth(r_thresh=0.0)
    r_none = synth(r_thresh=1.0)
    @test r_all.r_thresh == 0.0
    @test r_none.r_thresh == 1.0
    @test all(boundary_mask(r_none) .== false)
    @test count(boundary_mask(r_all)) >= count(boundary_mask(r_default))

    # ceiling_ratio itself is threshold-free, so all three agree on the statistic
    @test r_all.ceiling_ratio == r_default.ceiling_ratio == r_none.ceiling_ratio

    # the warning count matches the mask at the threshold actually requested
    logger = Test.TestLogger(respect_maxlog=false)
    r_cap = Base.CoreLogging.with_logger(logger) do
        synth(r_thresh=0.0)
    end
    msgs = filter(m -> occursin("peaking at the top", m),
                  [string(l.message) for l in logger.logs])
    @test !isempty(msgs)
    n_reported = parse(Int, match(r"(\d+) of \d+ wavelengths", msgs[1]).captures[1])
    @test n_reported == count(boundary_mask(r_cap))
end
