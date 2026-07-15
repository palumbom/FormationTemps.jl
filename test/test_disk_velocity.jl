# Regression tests for the disk-integration line-of-sight velocity field.
#
# The projected solid-body field must be v_los = -vsini * (x_sky/ρs) = -vsini*cosϕ*sinθ,
# where vsini is the PROJECTED rotational velocity. This is independent of istar
# (inclination is degenerate with vsini for a rigid, unspotted sphere).
#
# Guards against two prior bugs, both masked at the default istar=90°:
#   A) normalizing the rotation vector to constant |v| dropped the cosϕ latitude term
#      (high-latitude tiles over-sped by 1/cosϕ);
#   B) re-projecting the already-projected vsini through the inclination matrix added
#      a spurious sin(i) (max Doppler shrank as sin(istar)).

@testset "Disk-integration rotational velocity field" begin
    ρs = 1.0
    vsini = 20000.0   # m/s, projected
    Nϕ = 64

    # rebuild the (ϕ, θ) layout used inside calc_stellar_grid_cpu
    ϕe = range(deg2rad(-90.0), deg2rad(90.0), length=Nϕ + 1)
    ϕc = FT.get_grid_centers(ϕe)
    Nθ = ceil.(Int, 2π .* cos.(ϕc) ./ step(ϕe))
    Nθ_max = maximum(Nθ)
    PHI = fill(NaN, Nϕ, Nθ_max)
    THETA = fill(NaN, Nϕ, Nθ_max)
    for m in 1:Nϕ
        dθ = 2π / Nθ[m]
        for n in 1:Nθ[m]
            THETA[m, n] = (dθ / 2) + (n - 1) * dθ
            PHI[m, n] = ϕc[m]
        end
    end

    @testset "Correct field at istar=90 (Defect A: cosϕ latitude term)" begin
        μs, dA, z_rot = FT.calc_stellar_grid_cpu(ρs, 90.0, vsini, Nϕ)
        idx = findall(x -> x > 0.0, μs)
        vcode = z_rot[idx] .* FT.c_ms
        vcorr = -vsini .* cos.(PHI[idx]) .* sin.(THETA[idx])   # -vsini * x_sky/ρs
        @test maximum(abs.(vcode .- vcorr)) < 1e-6 * vsini
    end

    @testset "Inclination independence (Defect B: spurious sin i)" begin
        # The field formula carries no istar dependence: each visible tile obeys
        # v_los = -vsini*cosϕ*sinθ at EVERY inclination (only the visibility mask,
        # via μ, depends on istar). Max line-of-sight Doppler must equal vsini for
        # every inclination, not shrink as sin(istar).
        for istar in (90.0, 60.0, 30.0, 10.0)
            μs, _, z_rot = FT.calc_stellar_grid_cpu(ρs, istar, vsini, Nϕ)
            idx = findall(x -> x > 0.0, μs)
            vcode = z_rot[idx] .* FT.c_ms
            vcorr = -vsini .* cos.(PHI[idx]) .* sin.(THETA[idx])
            @test maximum(abs.(vcode .- vcorr)) < 1e-6 * vsini   # correct formula, any istar
            @test isapprox(maximum(abs.(vcode)), vsini; rtol=1e-2)  # max Doppler ~vsini (grid-limited)
        end
    end

    @testset "vsini=0 gives zero velocity" begin
        _, _, z_rot = FT.calc_stellar_grid_cpu(ρs, 45.0, 0.0, Nϕ)
        @test all(iszero, z_rot)
    end

    @testset "Differential rotation: f(ϕ) = 1 - α₂·sin²ϕ - α₄·sin⁴ϕ" begin
        α₂, α₄ = 0.2, 0.1   # solar-like magnitudes

        @testset "α=0 reproduces solid body (bit-identical)" begin
            for istar in (90.0, 30.0)
                _, _, z_solid = FT.calc_stellar_grid_cpu(ρs, istar, vsini, Nϕ)
                _, _, z_zero  = FT.calc_stellar_grid_cpu(ρs, istar, vsini, Nϕ; α₂=0.0, α₄=0.0)
                @test z_zero == z_solid          # exact, not just ≈
            end
        end

        @testset "per-tile scaling by f(ϕ)" begin
            μs, _, z_solid = FT.calc_stellar_grid_cpu(ρs, 90.0, vsini, Nϕ)
            _,  _, z_diff  = FT.calc_stellar_grid_cpu(ρs, 90.0, vsini, Nϕ; α₂=α₂, α₄=α₄)
            idx = findall(x -> x > 0.0, μs)
            sϕ = sin.(PHI[idx])
            f = 1 .- α₂ .* sϕ.^2 .- α₄ .* sϕ.^4
            @test z_diff[idx] ≈ f .* z_solid[idx] rtol=1e-12
            # off-equator slowed (f<1); nowhere sped up (f≤1)
            @test all(abs.(z_diff[idx]) .<= abs.(z_solid[idx]) .+ 1e-12)
        end

        @testset "istar drives the dA-weighted velocity distribution only when α≠0" begin
            # dA-weighted mean |v_los| over visible tiles. For rigid rotation this is a
            # pure sky-plane integral → inclination-invariant (up to grid discretization).
            # Differential rotation makes it istar-dependent (low istar weights the slow
            # poles more).
            wmean(istar, a2, a4) = begin
                μs, dA, z = FT.calc_stellar_grid_cpu(ρs, istar, vsini, Nϕ; α₂=a2, α₄=a4)
                idx = findall(x -> x > 0.0, μs)
                sum(dA[idx] .* abs.(z[idx])) / sum(dA[idx])
            end
            rigid_sens = abs(wmean(90.0, 0.0, 0.0) - wmean(30.0, 0.0, 0.0)) / wmean(90.0, 0.0, 0.0)
            diff_sens  = abs(wmean(90.0, α₂, α₄) - wmean(30.0, α₂, α₄)) / wmean(90.0, α₂, α₄)
            @test rigid_sens < 0.02                 # istar-invariant (grid-limited)
            @test diff_sens  > 3 * rigid_sens       # differential rotation makes istar matter
        end
    end
end
