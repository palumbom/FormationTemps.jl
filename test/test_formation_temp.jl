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
