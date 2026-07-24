let
# Regression tests for the ring Doppler kernel used by method=:quadrature: the
# bin-integrated line-of-sight velocity distribution of a constant-μ ring.
#
# Two properties are asserted:
#   1. Exact symmetry in velocity. The true distribution is symmetric because az → π-az
#      flips x_sky while leaving the latitude (hence f(ϕ)) fixed. An only-approximately
#      symmetric kernel imprints a spurious radial-velocity shift on every spectrum.
#   2. Area-exact bin weights rather than sampled estimates — analytic (arcsine CDF) for
#      solid-body rotation, arc-overlap deposition for differential rotation.
using FormationTemps; FT = FormationTemps
using Statistics
using Test

# 20 Å window at 6000 Å: wide enough (±500 km/s) that nothing below is truncated
λs = collect(range(5990.0, 6010.0, step=0.01))
Nλ = length(λs)
i0 = Nλ ÷ 2 + 1
Δv = FT.c_ms * (λs[2] - λs[1]) / λs[i0]

first_moment(K) = sum((eachindex(K) .- i0) .* K)      # in pixels
rv_offset(K) = first_moment(K) * Δv                   # in m/s

# independent reference: brute-force nearest-bin histogram at very high sampling.
# deliberately a different algorithm from both production paths.
function reference_kernel(μ, vsini, iₛ, α₂, α₄; N=2_000_000)
    r = sqrt(max(1 - μ^2, 0.0))
    K = zeros(Float64, Nλ)
    w = 1 / N
    for j in 0:(N-1)
        az = 2π * j / N
        sinϕ = r * sin(az) * cos(iₛ) + μ * sin(iₛ)
        v = -vsini * FT.diff_rot_factor(sinϕ, α₂, α₄) * r * cos(az)
        pn = round(Int, i0 + v / Δv)
        1 <= pn <= Nλ && (K[pn] += w)
    end
    s = sum(K)
    s > 0 && (K ./= s)
    return K
end

@testset "Ring Doppler kernel" begin

    @testset "normalization" begin
        for vsini in (2000.0, 15000.0), μ in (0.1, 0.5, 0.95)
            K = FT._ring_doppler_kernel(μ, vsini, 0.0, 0.0, 0.0, λs, 256)
            @test sum(K) ≈ 1.0 atol=1e-12
            @test all(≥(0.0), K)
        end
    end

    @testset "solid body: exactly symmetric (no spurious RV)" begin
        # asin is odd, so the analytic kernel cannot carry a first moment. Parity of the
        # (now unused) sample count must not matter.
        for vsini in (500.0, 2000.0, 2100.0, 5000.0, 15000.0, 50000.0)
            for N_az in (255, 256, 257, 1000, 1001)
                K = FT._ring_doppler_kernel(0.5, vsini, 0.0, 0.0, 0.0, λs, N_az)
                @test abs(rv_offset(K)) < 1e-9        # m/s
            end
        end
    end

    @testset "solid body: kernel is mirror-symmetric bin by bin" begin
        K = FT._ring_doppler_kernel(0.5, 15000.0, 0.0, 0.0, 0.0, λs, 256)
        half = min(i0 - 1, Nλ - i0)
        @test maximum(abs.(K[i0 .+ (1:half)] .- K[i0 .- (1:half)])) < 1e-15
    end

    @testset "solid body: inclination-independent" begin
        # with f ≡ 1 the projected field is -vsini·x_sky/ρ, carrying no istar dependence
        K90 = FT._ring_doppler_kernel(0.5, 15000.0, deg2rad(0.0), 0.0, 0.0, λs, 256)
        K30 = FT._ring_doppler_kernel(0.5, 15000.0, deg2rad(60.0), 0.0, 0.0, λs, 256)
        @test K90 == K30
    end

    @testset "solid body: matches an independent high-N reference" begin
        for vsini in (2000.0, 15000.0)
            K = FT._ring_doppler_kernel(0.5, vsini, 0.0, 0.0, 0.0, λs, 256)
            R = reference_kernel(0.5, vsini, 0.0, 0.0, 0.0)
            @test maximum(abs.(K .- R)) / maximum(R) < 5e-3    # reference's own noise floor
        end
    end

    @testset "solid body: analytic arcsine form" begin
        # spot-check against the closed form written out independently of the source
        vsini, μ = 15000.0, 0.5
        vmax = vsini * sqrt(1 - μ^2)
        G(u) = asin(clamp(u, -1.0, 1.0)) / π + 0.5
        K = FT._ring_doppler_kernel(μ, vsini, 0.0, 0.0, 0.0, λs, 256)
        for n in (i0 - 40, i0 - 7, i0, i0 + 7, i0 + 40)
            expect = G((n - i0 + 0.5) * Δv / vmax) - G((n - i0 - 0.5) * Δv / vmax)
            @test K[n] ≈ expect atol=1e-12
        end
    end

    @testset "differential rotation: symmetric for odd and even sample counts" begin
        iₛ = deg2rad(60.0)      # istar = 30 deg
        for vsini in (2000.0, 5000.0, 15000.0, 50000.0)
            for N_az in (255, 256, 257)
                K = FT._ring_doppler_kernel(0.5, vsini, iₛ, 0.2, 0.1, λs, N_az)
                @test sum(K) ≈ 1.0 atol=1e-12
                @test abs(rv_offset(K)) < 1e-3       # m/s; arc pairing is exact to roundoff
            end
        end
    end

    @testset "differential rotation: arc-overlap is area-exact" begin
        iₛ = deg2rad(60.0)
        for vsini in (2000.0, 15000.0)
            K = FT._ring_doppler_kernel(0.5, vsini, iₛ, 0.2, 0.1, λs, 256)
            R = reference_kernel(0.5, vsini, iₛ, 0.2, 0.1)
            @test maximum(abs.(K .- R)) / maximum(R) < 5e-3
        end
    end

    @testset "α=0 and α→0 agree" begin
        # the two code paths must meet in the limit
        K_rigid = FT._ring_doppler_kernel(0.5, 15000.0, 0.0, 0.0, 0.0, λs, 4096)
        K_tiny  = FT._ring_doppler_kernel(0.5, 15000.0, 0.0, 1e-12, 0.0, λs, 4096)
        @test maximum(abs.(K_rigid .- K_tiny)) / maximum(K_rigid) < 1e-3
    end

    @testset "Float32 grid (gpu_precision=Float32 path)" begin
        # c_ms is a Float64 constant, so Δv must be narrowed back to T before it reaches
        # the Δv::T helpers — otherwise a Float32 grid dispatches to no method at all.
        # This is exercised only through the GPU Float32 path in the integration tests, so
        # assert it directly here.
        λ32 = Float32.(λs)
        for (vsini, α₂, α₄) in ((15000.0f0, 0.0f0, 0.0f0),    # analytic branch
                                (15000.0f0, 0.2f0, 0.1f0))    # arc-overlap branch
            K = FT._ring_doppler_kernel(0.5f0, vsini, 0.0f0, α₂, α₄, λ32, 256)
            @test eltype(K) === Float32
            @test all(isfinite, K)
            @test all(≥(0.0f0), K)
            # pairwise summation over Nλ≈2000 Float32 terms: ~log2(N)·eps ≈ 1e-6, so 1e-4
            # is loose enough not to be flaky while still catching a real normalization bug
            @test sum(K) ≈ 1.0f0 atol=1e-4
            # the analytic branch stays exactly mirror-symmetric even at Float32 (asin is
            # oddly symmetric in IEEE and the bin edges m±1/2 negate exactly); the
            # arc-overlap branch is symmetric only to Float32 roundoff on ~10³-magnitude
            # pixel coordinates, so allow a small fraction of a pixel (5e-2 px ≈ 25 m/s)
            @test abs(sum((eachindex(K) .- i0) .* K)) < 5e-2    # px
        end
    end

    @testset "degenerate rings" begin
        # μ→1 (disk centre): zero projected radius, so no rotational broadening
        K = FT._ring_doppler_kernel(1.0, 15000.0, 0.0, 0.0, 0.0, λs, 256)
        @test K[i0] == 1.0
        @test sum(K) == 1.0
        # vsini = 0
        K0 = FT._ring_doppler_kernel(0.5, 0.0, 0.0, 0.0, 0.0, λs, 256)
        @test K0[i0] == 1.0
    end

    @testset "truncation is reported, not silent" begin
        narrow = collect(range(5999.0, 6001.0, step=0.01))    # ±30 km/s only
        @test_logs (:warn, r"truncated by the wavelength window") match_mode=:any begin
            FT._ring_doppler_kernel(0.5, 200_000.0, 0.0, 0.0, 0.0, narrow, 256)
        end
        # a kernel that fits must NOT warn (guards against roundoff-triggered noise).
        # Base.CoreLogging.Warn rather than Logging.Warn: identical value, no test-target
        # dependency on the Logging stdlib.
        @test_logs min_level=Base.CoreLogging.Warn begin
            FT._ring_doppler_kernel(0.5, 5000.0, 0.0, 0.0, 0.0, narrow, 256)
        end
    end
end

end
