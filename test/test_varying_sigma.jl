let
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

wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=0.01)
Nλ = length(λs_korg)
xs = collect(Float64, λs_korg)

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  ne_warn_thresh=Inf)

Npad = 512

@testset "Microturbulence dispatch overloads" begin

    # ── scalar (v_los::T, v_mic::T) ──────────────────────────────────────────────

    @testset "Scalar v_los + v_mic: CPU vs GPU" begin
        v_los_val = 500.0
        v_mic_val = 850.0
        result_cpu = FT.convolve_wavelength_axis(xs, αs, v_los_val, v_mic_val)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(cmem, xs, αs, v_los_val, v_mic_val))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    @testset "Scalar v_los + v_mic: device-native xs" begin
        v_los_val = 0.0
        v_mic_val = 1200.0
        result_cpu = FT.convolve_wavelength_axis(xs, αs, v_los_val, v_mic_val)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, CuArray(xs), CuArray(αs), v_los_val, v_mic_val))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    # ── mixed (v_los::CA, v_mic::T) ──────────────────────────────────────────────

    @testset "Vector v_los + scalar v_mic: CPU vs GPU" begin
        v_los_vary = collect(range(-400.0, 400.0, length=Natm))
        v_mic_val = 850.0
        result_cpu = FT.convolve_wavelength_axis(xs, αs, v_los_vary, fill(v_mic_val, Natm))
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, xs, αs, CuArray(v_los_vary), v_mic_val))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    # ── vector (v_los::CA, v_mic::CA) ────────────────────────────────────────────

    @testset "Vector v_los + vector v_mic: CPU vs GPU" begin
        v_los_vary = collect(range(-400.0, 400.0, length=Natm))
        v_mic_vary = collect(range(600.0, 1200.0, length=Natm))
        result_cpu = FT.convolve_wavelength_axis(xs, αs, v_los_vary, v_mic_vary)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, xs, αs, CuArray(v_los_vary), CuArray(v_mic_vary)))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    @testset "Vector v_los + vector v_mic: device-native xs" begin
        v_los_vary = collect(range(-300.0, 300.0, length=Natm))
        v_mic_vary = collect(range(600.0, 1200.0, length=Natm))
        result_cpu = FT.convolve_wavelength_axis(xs, αs, v_los_vary, v_mic_vary)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(
            cmem, CuArray(xs), CuArray(αs), CuArray(v_los_vary), CuArray(v_mic_vary)))
        @test maximum(abs.(result_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
    end

    # ── batched: scalar v_mic ──────────────────────────────────────────────────

    @testset "Batched: scalar v_mic" begin
        B = 3
        v_mic_val = 850.0
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        tile_vels = [0.0, 800.0, -600.0]
        v_los_batch_cpu = zeros(Float64, B * Natm)
        for b in 1:B
            off = (b - 1) * Natm
            v_los_batch_cpu[off+1:off+Natm] .= tile_vels[b]
        end

        result_gpu = Array(FT.convolve_wavelength_axis_batched!(
            bcmem, xs, αs, CuArray(v_los_batch_cpu), v_mic_val, B))

        for b in 1:B
            off = (b - 1) * Natm
            result_cpu = FT.convolve_wavelength_axis(xs, αs, tile_vels[b], v_mic_val)
            tile_gpu = result_gpu[off+1:off+Natm, :]
            @test maximum(abs.(tile_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
        end
    end

    # ── batched: vector v_mic ──────────────────────────────────────────────────

    @testset "Batched: vector v_mic" begin
        B = 3
        v_mic_vary = collect(range(600.0, 1200.0, length=Natm))
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        tile_vels = [0.0, 800.0, -600.0]
        v_los_batch_cpu = zeros(Float64, B * Natm)
        for b in 1:B
            off = (b - 1) * Natm
            v_los_batch_cpu[off+1:off+Natm] .= tile_vels[b]
        end

        result_gpu = Array(FT.convolve_wavelength_axis_batched!(
            bcmem, xs, αs, CuArray(v_los_batch_cpu), CuArray(v_mic_vary), B))

        for b in 1:B
            off = (b - 1) * Natm
            v_los_tile = fill(tile_vels[b], Natm)
            result_cpu = FT.convolve_wavelength_axis(xs, αs, v_los_tile, v_mic_vary)
            tile_gpu = result_gpu[off+1:off+Natm, :]
            @test maximum(abs.(tile_gpu .- result_cpu)) / maximum(abs.(result_cpu)) < 1e-10
        end
    end

    # ── consistency: scalar vs vector with uniform values ────────────────────

    @testset "Scalar vs vector dispatch consistency" begin
        v_mic_val = 850.0
        v_los_val = 500.0

        cmem_s = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_scalar = Array(FT.convolve_wavelength_axis_gpu(
            cmem_s, xs, αs, v_los_val, v_mic_val))

        cmem_v = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_vector = Array(FT.convolve_wavelength_axis_gpu(
            cmem_v, xs, αs, CuArray(fill(v_los_val, Natm)), CuArray(fill(v_mic_val, Natm))))

        @test maximum(abs.(result_scalar .- result_vector)) / maximum(abs.(result_scalar)) < 1e-10
    end
end

end
