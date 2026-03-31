# Tests for microturbulence dispatch: scalar, mixed, and vector overloads.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Test

# ── shared setup ──────────────────────────────────────────────────────────────
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0; logg = 4.44
A_X = Korg.format_A_X(0.0)
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm = length(atm_gpu.zs)

wls = [l.wl * 1e8 for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=0.01)
Nλ = length(λs_korg)
xs = collect(Float64, λs_korg)

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  ne_warn_thresh=Inf)

Npad = 512

@testset "Microturbulence dispatch overloads" begin

    # ── scalar (μ_v::T, σ_v::T) ──────────────────────────────────────────────

    @testset "Scalar μ_v + σ_v: CPU vs GPU" begin
        μ_v_val = 500.0
        σ_v_val = 850.0
        result_cpu = FT.convolve_wavelength_axis(xs, αs, μ_v_val, σ_v_val)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(cmem, xs, αs, μ_v_val, σ_v_val))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    @testset "Scalar μ_v + σ_v: device-native xs" begin
        μ_v_val = 0.0
        σ_v_val = 1200.0
        result_cpu = FT.convolve_wavelength_axis(xs, αs, μ_v_val, σ_v_val)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, CuArray(xs), CuArray(αs), μ_v_val, σ_v_val))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    # ── mixed (μ_v::CA, σ_v::T) ──────────────────────────────────────────────

    @testset "Vector μ_v + scalar σ_v: CPU vs GPU" begin
        μ_v_vary = collect(range(-400.0, 400.0, length=Natm))
        σ_v_val = 850.0
        result_cpu = FT.convolve_wavelength_axis(xs, αs, μ_v_vary, fill(σ_v_val, Natm))
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, xs, αs, CuArray(μ_v_vary), σ_v_val))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    # ── vector (μ_v::CA, σ_v::CA) ────────────────────────────────────────────

    @testset "Vector μ_v + vector σ_v: CPU vs GPU" begin
        μ_v_vary = collect(range(-400.0, 400.0, length=Natm))
        σ_v_vary = collect(range(600.0, 1200.0, length=Natm))
        result_cpu = FT.convolve_wavelength_axis(xs, αs, μ_v_vary, σ_v_vary)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, xs, αs, CuArray(μ_v_vary), CuArray(σ_v_vary)))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    @testset "Vector μ_v + vector σ_v: device-native xs" begin
        μ_v_vary = collect(range(-300.0, 300.0, length=Natm))
        σ_v_vary = collect(range(600.0, 1200.0, length=Natm))
        result_cpu = FT.convolve_wavelength_axis(xs, αs, μ_v_vary, σ_v_vary)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, CuArray(xs), CuArray(αs), CuArray(μ_v_vary), CuArray(σ_v_vary)))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    # ── batched: scalar σ_v ──────────────────────────────────────────────────

    @testset "Batched: scalar σ_v" begin
        B = 3
        σ_v_val = 850.0
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        tile_vels = [0.0, 800.0, -600.0]
        μ_v_batch_cpu = zeros(Float64, B * Natm)
        for b in 1:B
            off = (b - 1) * Natm
            μ_v_batch_cpu[off+1:off+Natm] .= tile_vels[b]
        end

        result_gpu = Array(FT.convolve_wavelength_axis_batched!(
            bcmem, xs, αs, CuArray(μ_v_batch_cpu), σ_v_val, B))

        for b in 1:B
            off = (b - 1) * Natm
            result_cpu = FT.convolve_wavelength_axis(xs, αs, tile_vels[b], σ_v_val)
            tile_gpu = result_gpu[off+1:off+Natm, :]
            @test maximum(abs.(tile_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
        end
    end

    # ── batched: vector σ_v ──────────────────────────────────────────────────

    @testset "Batched: vector σ_v" begin
        B = 3
        σ_v_vary = collect(range(600.0, 1200.0, length=Natm))
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        tile_vels = [0.0, 800.0, -600.0]
        μ_v_batch_cpu = zeros(Float64, B * Natm)
        for b in 1:B
            off = (b - 1) * Natm
            μ_v_batch_cpu[off+1:off+Natm] .= tile_vels[b]
        end

        result_gpu = Array(FT.convolve_wavelength_axis_batched!(
            bcmem, xs, αs, CuArray(μ_v_batch_cpu), CuArray(σ_v_vary), B))

        for b in 1:B
            off = (b - 1) * Natm
            μ_v_tile = fill(tile_vels[b], Natm)
            result_cpu = FT.convolve_wavelength_axis(xs, αs, μ_v_tile, σ_v_vary)
            tile_gpu = result_gpu[off+1:off+Natm, :]
            @test maximum(abs.(tile_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
        end
    end

    # ── consistency: scalar vs vector with uniform values ────────────────────

    @testset "Scalar vs vector dispatch consistency" begin
        σ_v_val = 850.0
        μ_v_val = 500.0

        cmem_s = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_scalar = Array(FT.convolve_wavelength_axis_gpu(
            cmem_s, xs, αs, μ_v_val, σ_v_val))

        cmem_v = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_vector = Array(FT.convolve_wavelength_axis_gpu(
            cmem_v, xs, αs, CuArray(fill(μ_v_val, Natm)), CuArray(fill(σ_v_val, Natm))))

        @test maximum(abs.(result_scalar .- result_vector)) / maximum(abs.(result_scalar)) < 1e-10
    end
end
