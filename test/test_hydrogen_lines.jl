let
# Tests for hydrogen (Balmer/Brackett) line opacity, which is on by default.
#
# Korg computes hydrogen lines from dedicated Stark/MHD physics rather than from the
# linelist, so `Korg.line_absorption!` never emits them and FormationTemps has to add them
# explicitly. Three properties are asserted:
#
#   1. The opacity lands in `αs` only — never in `αs_cont`, and never in the 5000 Å
#      reference opacity `α_ref`. Otherwise Balmer lines would not appear as features
#      against the continuum, and the anchored τ scale would be biased.
#   2. `hydrogen_lines=false` suppresses it exactly, which is what makes comparison against
#      `Korg.synthesize(...; hydrogen_lines=false)` meaningful.
#   3. `use_MHD` follows FormationTemps' documented rule, which differs from
#      Korg's: Korg defaults `use_MHD_for_hydrogen_lines=true` at all wavelengths and merely
#      warns above 13000 Å, while we follow that warning automatically.
using FormationTemps; FT = FormationTemps
using Korg
using Statistics
using Test

A_X = Korg.format_A_X(0.0)
atm = FT.AtmosphereCPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
Natm = length(atm.zs)

# a single weak synthetic Fe I line, so any difference between runs is hydrogen, not metals
mkline(λ_ang) = [Korg.Line(λ_ang * FT.ANGSTROM_TO_CM, -2.880, Korg.Species("Fe I"),
                           2.223, exp10(8.31), exp10(-6.16), log10(exp10(-7.69)))]

# returns (αs, αs_cont, α_ref) on the given window
function alphas(λ_lo, λ_hi, step, line_λ; kwargs...)
    λs = collect(range(λ_lo, λ_hi, step=step))
    Nλ = length(λs)
    αs = zeros(Natm, Nλ); αs_cont = zeros(Natm, Nλ); α_ref = zeros(Natm)
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs), mkline(line_λ), atm, A_X;
                      α_ref_out=α_ref, ne_warn_thresh=Inf, kwargs...)
    return αs, αs_cont, α_ref, λs
end

@testset "Hydrogen line absorption" begin

    # Hα at 6562.8 Å (air); well inside the default 150 Å per-line window
    @testset "Balmer opacity appears in αs and only in αs" begin
        on  = alphas(6552.0, 6572.0, 0.02, 6558.0)
        off = alphas(6552.0, 6572.0, 0.02, 6558.0; hydrogen_lines=false)
        αs_on, cont_on, ref_on, λs = on
        αs_off, cont_off, ref_off, _ = off

        # (1) hydrogen adds real opacity near Hα
        icore = argmin(abs.(λs .- 6562.8))
        @test maximum(αs_on[:, icore]) > maximum(αs_off[:, icore])
        # substantial, not a rounding wiggle: Hα is a strong feature in the solar photosphere
        @test maximum(αs_on[:, icore]) > 1.05 * maximum(αs_off[:, icore])

        # (2) the continuum is untouched — bit-identical, not merely close
        @test cont_on == cont_off

        # (3) the 5000 Å reference opacity is untouched (matches Korg, which builds α5 from
        #     continuum + a 5000 Å linelist and adds hydrogen only to α)
        @test ref_on == ref_off
        @test all(ref_on .> 0)

        # (4) opacity is only ever added, never subtracted
        @test all(αs_on .>= αs_off .- 1e-30)
    end

    @testset "hydrogen_lines=false reproduces the no-hydrogen result exactly" begin
        # two independent runs with hydrogen off must agree bit-for-bit, and must differ
        # from the default — guards against the flag being silently ignored
        a1, c1, r1, _ = alphas(6552.0, 6572.0, 0.05, 6558.0; hydrogen_lines=false)
        a2, c2, r2, _ = alphas(6552.0, 6572.0, 0.05, 6558.0; hydrogen_lines=false)
        @test a1 == a2 && c1 == c2 && r1 == r2
        adef, _, _, _ = alphas(6552.0, 6572.0, 0.05, 6558.0)
        @test adef != a1
    end

    @testset "far from any hydrogen line the flag is a no-op" begin
        # Fe I 6173 sits 389 Å from Hα, outside the 150 Å window. This is the regime
        # compare_korg.jl works in, and it must be insensitive to the flag.
        on, _, _, _  = alphas(6172.8, 6173.8, 0.005, 6173.33)
        off, _, _, _ = alphas(6172.8, 6173.8, 0.005, 6173.33; hydrogen_lines=false)
        @test on == off
    end

    @testset "use_MHD rule (differs from Korg default)" begin
        # Below 13000 Å the default enables MHD, matching Korg.
        lo_default, _, _, _ = alphas(6552.0, 6572.0, 0.05, 6558.0)
        lo_true, _, _, _    = alphas(6552.0, 6572.0, 0.05, 6558.0; use_MHD=true)
        @test lo_default == lo_true

        # Above 13000 Å the default DISABLES MHD, following Korg's own recommendation
        # rather than Korg's default. This is the documented deviation; pin it.
        hi_default, _, _, _ = alphas(15000.0, 15100.0, 0.2, 15050.0)
        hi_false, _, _, _   = alphas(15000.0, 15100.0, 0.2, 15050.0; use_MHD=false)
        @test hi_default == hi_false

        # and the override is honoured and produces usable numbers
        hi_true, _, _, _ = alphas(15000.0, 15100.0, 0.2, 15050.0; use_MHD=true)
        @test all(isfinite, hi_true)
        @test all(hi_true .> 0)
    end

    @testset "window size is plumbed through" begin
        # shrinking the per-line window below the distance to Hα must remove its opacity
        # from a point that a wide window includes
        wide, _, _, _   = alphas(6440.0, 6450.0, 0.05, 6445.0)                                  # ~118 Å from Hα
        narrow, _, _, _ = alphas(6440.0, 6450.0, 0.05, 6445.0; hydrogen_line_window_size_Å=50.0)
        none, _, _, _   = alphas(6440.0, 6450.0, 0.05, 6445.0; hydrogen_lines=false)
        @test wide != narrow            # 150 Å window reaches here, 50 Å does not
        @test narrow == none            # with H out of range, identical to H disabled
    end
end

end
