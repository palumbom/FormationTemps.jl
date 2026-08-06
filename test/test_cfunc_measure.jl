# `cont_func` is a per-interval INTEGRAL: sum(cont_func, dims=1) is the emergent flux, so the
# layer measure is baked into each element. Sums over it (the CDF behind form_temps) are
# therefore grid-independent, but any statistic that compares one interval against another is
# not. On the native MARCS grid the spacing changes by 2x at log tau_ref = -3 and +1, which
# inflates the topmost interval relative to a peak interval in the finely-sampled region.
#
# These tests pin the conversion from integral to density (`cfunc_per_dex`) and the resulting
# grid-invariance of `ceiling_ratio`.

@testset "Contribution function layer measure" begin

    # a smooth density in log10 tau_ref, sampled onto an arbitrary grid as per-interval
    # integrals -- i.e. what cont_func holds
    logmids(τ_ref) = [0.5 * (log10(τ_ref[k]) + log10(τ_ref[k+1])) for k in 1:length(τ_ref)-1]
    sample_integrals(f, τ_ref) = [f(x) * d for (x, d) in zip(logmids(τ_ref),
                                                             diff(log10.(τ_ref)))]

    @testset "cfunc_per_dex divides out the interval width" begin
        # deliberate 2x spacing change, mirroring the MARCS grid at log tau_ref = -3
        τ_ref = 10.0 .^ [-3.0, -2.8, -2.6, -2.5, -2.4, -2.3]
        Δ = diff(log10.(τ_ref))
        @test Δ ≈ [0.2, 0.2, 0.1, 0.1, 0.1]

        cfunc_dt = reshape([1.0, 1.0, 1.0, 1.0, 1.0], 5, 1)
        # equal integrals over unequal intervals means the density is NOT flat
        @test vec(FT.cfunc_per_dex(cfunc_dt, τ_ref)) ≈ 1.0 ./ Δ

        # and a flat density round-trips through the sampling
        flat = reshape(sample_integrals(_ -> 3.0, τ_ref), 5, 1)
        @test vec(FT.cfunc_per_dex(flat, τ_ref)) ≈ fill(3.0, 5)

        @test eltype(FT.cfunc_per_dex(Float32.(cfunc_dt), Float32.(τ_ref))) == Float32
    end

    @testset "ceiling_ratio recovers the density ratio, not the integral ratio" begin
        τ_ref = 10.0 .^ [-3.0, -2.8, -2.6, -2.5, -2.4, -2.3]
        f = x -> exp(-((x + 2.4) / 0.25)^2)          # peaked inside the fine region
        cfunc_dt = reshape(sample_integrals(f, τ_ref), 5, 1)

        xs = logmids(τ_ref)
        expected = f(first(xs)) / maximum(f.(xs))
        @test ceiling_ratio(cfunc_dt, τ_ref) ≈ [expected]

        # the un-normalized reduction sees the top interval inflated by its 2x width
        naive = cfunc_dt[1, 1] / maximum(cfunc_dt)
        @test naive > 1.5 * expected
    end

    @testset "ceiling_ratio is invariant to the layer grid" begin
        # the invariant that the removal of uniform-log-tau resampling broke: the same
        # physical contribution density must give the same statistic on either grid
        atm = FT.AtmosphereCPU(Korg.interpolate_marcs(5777.0, 4.44))
        τ_native = atm.τs
        @test !isempty(τ_native)

        # the native grid really is non-uniform, else this test proves nothing
        Δnative = diff(log10.(τ_native))
        @test maximum(Δnative) / minimum(Δnative) > 2.0

        τ_uniform = 10.0 .^ collect(range(log10(first(τ_native)), log10(last(τ_native)),
                                          length=length(τ_native)))

        for f in (x -> exp(-((x - 0.0) / 1.2)^2),      # peaked deep, near tau_ref = 1
                  x -> exp(-((x + 4.0) / 1.2)^2))      # peaked high, ceiling-contaminated
            c_native = reshape(sample_integrals(f, τ_native), :, 1)
            c_uniform = reshape(sample_integrals(f, τ_uniform), :, 1)

            r_native = ceiling_ratio(c_native, τ_native)
            r_uniform = ceiling_ratio(c_uniform, τ_uniform)
            # Residual differences are interval-averaging of the density, not measure error, so
            # they are bounded in absolute terms rather than relative ones. The deep-peaked case
            # puts the top node ~4σ out on a Gaussian tail, where the density drops by orders of
            # magnitude across one interval and the native top interval is 1.6x the uniform
            # width; the interval average then exceeds the midpoint value by a width-dependent
            # amount, and the ratio of two such tails differs by tens of percent while both are
            # ~1e-7. atol is the criterion that matches how the statistic is used — thresholded
            # against r_thresh — and the contaminated case below still pins the relative bound
            # where the statistic is O(0.1) and actually decides something.
            @test isapprox(r_native[1], r_uniform[1]; rtol=0.05, atol=1e-3)

            # the raw reduction is NOT invariant -- this is the artifact, asserted so the
            # conversion cannot be quietly dropped as redundant
            naive_native = c_native[1, 1] / maximum(c_native)
            naive_uniform = c_uniform[1, 1] / maximum(c_uniform)
            @test !isapprox(naive_native, naive_uniform; rtol=0.05)
        end
    end

    @testset "boundary_mask thresholds the measure-consistent statistic" begin
        τ_ref = 10.0 .^ [-3.0, -2.8, -2.6, -2.5]
        # integrals chosen so the two statistics straddle the default threshold: the top
        # interval is 0.2 dex wide against a 0.1 dex peak interval
        cfunc_dt = reshape([0.05, 0.02, 0.25], 3, 1)
        @test cfunc_dt[1] / maximum(cfunc_dt) ≈ 0.2         # naive: under threshold
        @test ceiling_ratio(cfunc_dt, τ_ref) ≈ [0.1]        # density: further under
        @test boundary_mask(cfunc_dt, τ_ref) == [false]
        @test boundary_mask(cfunc_dt, τ_ref; r_thresh=0.05) == [true]
    end

    @testset "an absent reference grid falls back to a uniform measure" begin
        # the Bezier tau path has no tau_ref, so no measure is available; the statistic is
        # then grid-dependent and says so rather than silently claiming otherwise
        cfunc_dt = reshape([0.2, 0.6, 0.2], 3, 1)
        logger = Test.TestLogger(respect_maxlog=false)
        r = Base.CoreLogging.with_logger(logger) do
            ceiling_ratio(cfunc_dt, Float64[])
        end
        @test r ≈ [0.2 / 0.6]
        @test any(l -> occursin("no reference optical depth", string(l.message)), logger.logs)
    end

    @testset "FormTempResult stores the measure-consistent statistic" begin
        atm = FT.AtmosphereCPU(Korg.interpolate_marcs(5777.0, 4.44))
        Natm = atm.Natm
        f = x -> exp(-((x + 2.0) / 1.0)^2)
        cont_func = reshape(sample_integrals(f, atm.τs), :, 1)

        res = FormTempResult([5000.0], [1.0], [5500.0], cont_func, atm)
        @test res.ceiling_ratio ≈ ceiling_ratio(cont_func, atm.τs)
        @test ceiling_ratio(res) == res.ceiling_ratio
        @test boundary_mask(res) == boundary_mask(cont_func, atm.τs)

        # and it is NOT the naive reduction
        @test !isapprox(res.ceiling_ratio[1], cont_func[1, 1] / maximum(cont_func); rtol=0.05)
    end

    @testset "form_temps_from_cfunc warns on the same statistic it is given a grid for" begin
        # consolidation invariant, now measure-aware: the warned set must equal the masked set
        atm = FT.AtmosphereCPU(Korg.interpolate_marcs(5777.0, 4.44))
        Ts = atm.Ts
        for f in (x -> exp(-((x - 0.0) / 1.2)^2), x -> exp(-((x + 4.6) / 0.8)^2))
            c = reshape(sample_integrals(f, atm.τs), :, 1)
            logger = Test.TestLogger(respect_maxlog=false)
            Base.CoreLogging.with_logger(logger) do
                form_temps_from_cfunc(c, Ts; τ_ref=atm.τs)
            end
            warned = any(l -> occursin("peaking at the top", string(l.message)), logger.logs)
            @test warned == any(boundary_mask(c, atm.τs))
        end
    end
end
