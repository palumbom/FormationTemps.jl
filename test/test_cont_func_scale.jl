# The absolute scale of cont_func. Every method reports each interval's contribution to the
# emergent surface flux F = 2π∫Iμdμ, in Korg's flux units. The other checks on cont_func are
# blind to its scale (normalized flux, form_temps and ceiling_ratio are all ratios), so this
# file is the only gate on the per-method factors applied when the result is assembled.
#
# The continuum level of a result is recovered as sum(cont_func, dims=1) ./ flux, since
# flux = sum(cfunc) / sum(cfunc_cont), and compared with Korg's continuum over the same window.
using FormationTemps; FT = FormationTemps
using Korg
using Statistics
using Test

isdefined(Main, :run_slow_tests) || include(joinpath(@__DIR__, "test_tiers.jl"))

let
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
Teff, logg, Fe_H = 5777.0, 4.44, 0.0
methods = (:disk, :quadrature, :hirano)
Nϕ = run_slow_tests() ? 64 : 32      # :disk tiling resolution; the tolerances below hold at both

cont_level(r) = median(vec(sum(r.cont_func, dims=1)) ./ r.flux)

function run_method(method; use_gpu=false, ρstar=1.0)
    star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=2100.0,
                        v_macro=3400.0, v_micro=850.0, ρstar=ρstar)
    kw = method === :hirano ? (u1=0.43, u2=0.31) :
         method === :disk   ? (Nϕ=Nϕ,) : NamedTuple()
    return calc_formation_temp(star, linelist; Δλ=0.01, use_gpu=use_gpu, method=method,
                               showprogress=false, ne_warn_thresh=Inf, kw...)
end

cpu = Dict(m => run_method(m) for m in methods)

A_X = Korg.format_A_X(Fe_H)
wavs = cpu[:disk].wavs
sol = Korg.synthesize(Korg.interpolate_marcs(Teff, logg, A_X), linelist, A_X,
                      (first(wavs), last(wavs)); vmic=0.0)
F_korg = median(sol.cntm)

@testset "cont_func absolute scale" begin
    @testset "continuum level matches Korg's surface flux for every method" begin
        # measured ratio to Korg: 0.99920 (:disk, Nϕ=64), 0.99919 (:quadrature, :hirano)
        for m in methods
            @test isapprox(cont_level(cpu[m]), F_korg; rtol=2e-3)
        end
    end

    @testset "methods agree on the scale" begin
        # measured |disk/quadrature - 1| = 7.6e-6 (Nϕ=64); the defect this guards was a factor of 2π
        ref = cont_level(cpu[:quadrature])
        for m in methods
            @test isapprox(cont_level(cpu[m]), ref; rtol=1e-4)
        end
    end

    @testset "ρstar does not scale :disk cont_func" begin
        # dA ∝ ρstar² is divided out; the residual is the finite observer distance in calc_mu
        r2 = run_method(:disk; ρstar=2.0)
        @test isapprox(r2.cont_func, cpu[:disk].cont_func; rtol=1e-9)
    end

    if FT.GPU_DEFAULT
        @testset "GPU paths apply the same scale" begin
            for m in methods
                @test isapprox(cont_level(run_method(m; use_gpu=true)), cont_level(cpu[m]);
                               rtol=1e-6)
            end
        end
    end
end
end
