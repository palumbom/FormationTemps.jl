let
# Validates the ring-by-ring μ-quadrature disk integration (method=:quadrature)
# against the explicit tile-based disk integration (method=:disk), which is the
# reference/ground truth. CPU-only.
#
# Tolerances are in Kelvin on form_temps and reflect a real, documented accuracy floor:
# the ring Doppler kernel lives on the wavelength pixel grid, so a narrow kernel (low
# vsini) is resolved only to ~pixel accuracy. See the `_ring_doppler_kernel` docstring
# and docs/src/methods.md for why that floor is accepted rather than engineered away.
using FormationTemps; FT = FormationTemps
using Korg
using Statistics
using Test

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff, logg, Fe_H = 5777.0, 4.44, 0.0
ζ_RT, ξ = 3400.0, 850.0
Δλ = 0.01
Nϕ = 64          # tiling ground-truth resolution

# tiling (ground truth) vs quadrature for one star; returns (tiling, quad, interior)
function run_pair(; vsini, istar, α₂=0.0, α₄=0.0, ζ=ζ_RT, Nμ=32, N_az=256)
    star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ,
                        v_micro=ξ, istar=istar, α₂=α₂, α₄=α₄)
    rt = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:disk,
                             Nϕ=Nϕ, showprogress=false, ne_warn_thresh=Inf)
    rq = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:quadrature,
                             Nμ=Nμ, N_az=N_az, ne_warn_thresh=Inf)
    λ0 = mean(rt.wavs)
    edge = ceil(Int, max(vsini, ζ) * 3 / (FT.c_ms * Δλ / λ0)) + 10
    n = length(rt.wavs)
    interior = (edge + 1):(n - edge)
    @test !isempty(interior)          # guard against a degenerate mask (would else error)
    return rt, rq, interior
end

@testset "Quadrature vs explicit disk integration" begin

    @testset "non-rotating (vsini=0) matches tiling" begin
        rt, rq, interior = run_pair(vsini=0.0, istar=90.0)
        @test length(rq.wavs) == length(rt.wavs)
        @test maximum(abs.(rq.flux[interior] .- rt.flux[interior])) < 1e-3
        @test maximum(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 2.0   # K
        @test mean(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 0.2       # K
    end

    @testset "rigid rotation (vsini=15 km/s, i=90) matches tiling" begin
        rt, rq, interior = run_pair(vsini=15000.0, istar=90.0)
        @test maximum(abs.(rq.flux[interior] .- rt.flux[interior])) < 1e-3
        @test maximum(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 3.0
        @test mean(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 0.3
        # quadrature must be genuinely distinct from tiling (guards a silent fallback
        # to method=:disk, which would make the tolerances above pass trivially)
        @test rq.form_temps != rt.form_temps
    end

    @testset "differential + inclined (α=(0.2,0.1), vsini=15 km/s, i=30)" begin
        rt, rq, interior = run_pair(vsini=15000.0, istar=30.0, α₂=0.2, α₄=0.1)
        @test maximum(abs.(rq.flux[interior] .- rt.flux[interior])) < 1e-3
        @test maximum(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 3.0
        @test mean(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 0.3
    end

    @testset "small vsini (2 km/s): narrow Doppler kernel" begin
        # narrow kernel is the hardest case (few pixels wide). The area-exact
        # nearest-bin kernel keeps the worst-pixel error ~2 K and the mean small;
        # part of even this is the finite-Nϕ tiling reference's own error.
        rt, rq, interior = run_pair(vsini=2000.0, istar=90.0)
        @test mean(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 0.3
        @test maximum(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 4.0
    end

    @testset "no macroturbulence (ζ=0) matches tiling" begin
        # exercises the ζ==0 short-circuit in the per-ring macro step
        rt, rq, interior = run_pair(vsini=15000.0, istar=90.0, ζ=0.0)
        @test maximum(abs.(rq.flux[interior] .- rt.flux[interior])) < 1e-3
        @test maximum(abs.(rq.form_temps[interior] .- rt.form_temps[interior])) < 3.0
    end

    @testset "convergence: more nodes → closer to tiling" begin
        # tiling ground truth + coarse quadrature from one call; fine quadrature separately
        rt, rq_coarse, interior = run_pair(vsini=15000.0, istar=90.0, Nμ=6, N_az=64)
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=15000.0, v_macro=ζ_RT,
                            v_micro=ξ, istar=90.0)
        rq_fine = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false,
                                      method=:quadrature, Nμ=24, N_az=256, ne_warn_thresh=Inf)
        coarse_err = maximum(abs.(rq_coarse.form_temps[interior] .- rt.form_temps[interior]))
        fine_err = maximum(abs.(rq_fine.form_temps[interior] .- rt.form_temps[interior]))
        @test fine_err < coarse_err
    end

    @testset "formation temps within atmosphere T range" begin
        _, rq, _ = run_pair(vsini=15000.0, istar=90.0)
        atm = rq.atmosphere
        @test all(rq.form_temps .>= minimum(FT.get_Ts(atm)))
        @test all(rq.form_temps .<= maximum(FT.get_Ts(atm)))
    end

    @testset "convolve deprecated alias still selects tiling/hirano" begin
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=0.0, v_macro=0.0, v_micro=ξ)
        r_alias = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, convolve=false,
                                      showprogress=false, ne_warn_thresh=Inf)
        r_method = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:disk,
                                       showprogress=false, ne_warn_thresh=Inf)
        @test r_alias.form_temps == r_method.form_temps
    end

    @testset "method/convolve resolution is warned about, not silent" begin
        # `convolve` is deprecated in favour of `method`. Two things must be audible:
        # the deprecation itself, and `convolve` being overridden when both are passed.
        # Uses :quadrature so each call is cheap.
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=0.0, v_macro=0.0, v_micro=ξ)

        # convolve=false is indistinguishable from the default, so it must stay quiet —
        # otherwise every existing caller gets noise for using the old default
        @test_logs min_level=Base.CoreLogging.Warn begin
            calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:quadrature,
                                convolve=false, ne_warn_thresh=Inf)
        end

        # convolve=true actually changes behaviour, so it must warn
        @test_logs (:warn, r"`convolve` is deprecated") match_mode=:any begin
            calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, convolve=true,
                                u1=0.43, u2=0.31, showprogress=false, ne_warn_thresh=Inf)
        end

        # both given: `method` wins, and says so rather than dropping convolve silently
        r = @test_logs (:warn, r"ignored because `method=") match_mode=:any begin
            calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:quadrature,
                                convolve=true, ne_warn_thresh=Inf)
        end
        # and it really did run the quadrature, not Hirano
        r_q = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:quadrature,
                                  ne_warn_thresh=Inf)
        @test r.form_temps == r_q.form_temps
    end

    @testset "unknown method is rejected" begin
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=0.0, v_macro=0.0, v_micro=ξ)
        @test_throws AssertionError calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false,
                                                        method=:nonsense, ne_warn_thresh=Inf)
    end
end

end
