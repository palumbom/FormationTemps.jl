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
    # Two failure modes have no meaningful median and previously returned a plausible-looking
    # number. Both must now yield NaN and say so. These are diagnostics, which rot quietly if
    # untested: a refactor that stopped emitting them would otherwise go unnoticed.
    Ts = [4000.0, 5000.0, 6000.0, 7000.0]

    @testset "zero total contribution → NaN + warning" begin
        # reachable: the microturbulence underflow guard deliberately zeros a row when the
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

@testset "Formation temperature boundary pinning is flagged" begin
    # When over half the flux contribution comes from the topmost layer interval, the 50%
    # crossing is set by where the model atmosphere was truncated rather than by where the
    # line forms. Expected in deep line cores (Balmer especially, now on by default), and the
    # returned value must be read as a lower limit rather than a measurement.
    Ts = [4000.0, 5000.0, 6000.0, 7000.0]
    pinned = reshape([0.9, 0.05, 0.05], 3, 1)     # 90% of the flux in the top interval

    @testset "warns and lands in the first interval" begin
        ft = @test_logs (:warn, r"topmost layer interval") match_mode=:any begin
            form_temps_from_cfunc(pinned, Ts)
        end
        # crossing at F=0.5 within a first interval holding 0.9 → 4000 + 1000*(0.5/0.9)
        @test isapprox(ft[1], 4000.0 + 1000.0 * (0.5 / 0.9); atol=1e-9)
        @test Ts[1] <= ft[1] <= Ts[2]
    end

    @testset "warn_boundary=false silences it without changing the value" begin
        quiet = @test_logs min_level=Base.CoreLogging.Warn begin
            form_temps_from_cfunc(pinned, Ts; warn_boundary=false)
        end
        @test quiet == form_temps_from_cfunc(pinned, Ts; warn_boundary=false)
        @test isapprox(quiet[1], 4000.0 + 1000.0 * (0.5 / 0.9); atol=1e-9)
    end

    @testset "a well-resolved column does not warn" begin
        # guards against the counter firing on ordinary spectra, which would make the
        # warning useless noise
        ok = reshape([0.2, 0.6, 0.2], 3, 1)
        @test_logs min_level=Base.CoreLogging.Warn begin
            form_temps_from_cfunc(ok, Ts)
        end
        # exactly at the threshold: F reaches 0.5 at the first deep node, which is the
        # boundary of the pinned condition (Fnodes[2] >= 0.5)
        edge = reshape([0.5, 0.3, 0.2], 3, 1)
        @test_logs (:warn, r"topmost layer interval") match_mode=:any begin
            form_temps_from_cfunc(edge, Ts)
        end
    end
end
