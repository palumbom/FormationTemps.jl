using FormationTemps; FT = FormationTemps
using Korg
using FFTW
using Statistics
using Test

# shared setup: atmosphere + absorption
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
ξ = 850.0
A_X = Korg.format_A_X(0.0)

atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm = length(atm_cpu.zs)

wls = [l.wl * 1e8 for l in linelist]
Δλ = 0.01
λs = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
Nλ = length(λs)

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs), linelist, atm_cpu, A_X;
                  ne_warn_thresh=Inf)

ws = FT.CPUTileWorkspace(Float64, Natm, Nλ)

@testset "In-place convolution correctness" begin
    @testset "Microturbulence: uniform σ_v matches allocating version" begin
        σ_v = fill(ξ, Natm)
        μ_v = fill(500.0, Natm)

        ref = FT.convolve_wavelength_axis(λs, αs, μ_v, σ_v)
        out = zeros(Natm, Nλ)
        FT._convolve_micro_inplace!(out, collect(λs), αs, μ_v, σ_v, ws)

        @test size(out) == size(ref)
        @test maximum(abs.(out .- ref)) < 1e-12
    end

    @testset "Microturbulence: non-uniform σ_v matches allocating version" begin
        σ_v = collect(range(700.0, 1000.0, length=Natm))
        μ_v = fill(300.0, Natm)

        ref = FT.convolve_wavelength_axis(λs, αs, μ_v, σ_v)
        out = zeros(Natm, Nλ)
        FT._convolve_micro_inplace!(out, collect(λs), αs, μ_v, σ_v, ws)

        @test maximum(abs.(out .- ref)) < 1e-12
    end

    @testset "Microturbulence: non-uniform μ_v matches allocating version" begin
        σ_v = fill(ξ, Natm)
        μ_v = collect(range(-200.0, 200.0, length=Natm))

        ref = FT.convolve_wavelength_axis(λs, αs, μ_v, σ_v)
        out = zeros(Natm, Nλ)
        FT._convolve_micro_inplace!(out, collect(λs), αs, μ_v, σ_v, ws)

        @test maximum(abs.(out .- ref)) < 1e-12
    end

    @testset "Microturbulence: zero velocity is identity-like" begin
        σ_v = fill(0.0, Natm)
        μ_v = fill(0.0, Natm)

        out = zeros(Natm, Nλ)
        FT._convolve_micro_inplace!(out, collect(λs), αs, μ_v, σ_v, ws)

        # with σ=0 the kernel degenerates to a delta; output should match input
        # (subject to the σ_floor clamp producing a very narrow Gaussian)
        @test maximum(abs.(out .- αs)) < 1e-3 * maximum(αs)
    end

    @testset "Macroturbulence: ζ > 0 matches allocating version" begin
        ζ_RT = 3400.0
        μ = 0.7

        # use cfunc-shaped input (Natm-1 rows)
        ws_mac = FT.CPUTileWorkspace(Float64, Natm, Nλ)
        ys = randn(Natm - 1, Nλ) .* 1e-5
        ref = FT.convolve_rt_macro(λs, ys, ζ_RT, μ)
        out = zeros(Natm - 1, Nλ)
        FT._convolve_macro_inplace!(out, collect(λs), ys, ζ_RT, μ, ws_mac)

        @test size(out) == size(ref)
        @test maximum(abs.(out .- ref)) < 1e-12
    end

    @testset "Macroturbulence: ζ = 0 copies input" begin
        ys = randn(Natm - 1, Nλ) .* 1e-5
        out = zeros(Natm - 1, Nλ)
        FT._convolve_macro_inplace!(out, collect(λs), ys, 0.0, 0.8, ws)

        @test out == ys
    end

    @testset "Macroturbulence: different μ values produce different results" begin
        ζ_RT = 3400.0
        ys = randn(Natm - 1, Nλ) .* 1e-5

        out1 = zeros(Natm - 1, Nλ)
        out2 = zeros(Natm - 1, Nλ)
        FT._convolve_macro_inplace!(out1, collect(λs), ys, ζ_RT, 0.3, ws)
        FT._convolve_macro_inplace!(out2, collect(λs), ys, ζ_RT, 0.9, ws)

        @test !isapprox(out1, out2; atol=1e-20)
    end
end

@testset "Workspace buffer isolation" begin
    @testset "Successive calls do not leak state" begin
        σ_v = fill(ξ, Natm)
        μ_v1 = fill(500.0, Natm)
        μ_v2 = fill(-500.0, Natm)

        out1 = zeros(Natm, Nλ)
        out2 = zeros(Natm, Nλ)

        FT._convolve_micro_inplace!(out1, collect(λs), αs, μ_v1, σ_v, ws)
        ref1 = copy(out1)

        FT._convolve_micro_inplace!(out2, collect(λs), αs, μ_v2, σ_v, ws)

        # re-run first call; should reproduce exactly
        FT._convolve_micro_inplace!(out1, collect(λs), αs, μ_v1, σ_v, ws)
        @test out1 == ref1
    end
end
