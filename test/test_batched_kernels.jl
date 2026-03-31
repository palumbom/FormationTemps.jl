# Tests that batched GPU kernels produce results matching their single-tile counterparts.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Test
using Statistics

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
Fe_H = 0.0
ξ = 850.0
Δλ = 0.01
Npad = 512

A_X = Korg.format_A_X(Fe_H)
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm = length(atm_gpu.zs)

wls = [l.wl * 1e8 for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
Nλ = length(λs_korg)

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  α_ref_out=α_ref, ne_warn_thresh=Inf)

σ_v = CUDA.zeros(Float64, Natm) .+ ξ
log_τ_ref = CuArray{Float64}(log.(atm_gpu.τs))
ifactor_base = CuArray{Float64}(atm_gpu.τs ./ α_ref)
Ts_gpu = CuArray{Float64}(atm_gpu.Ts)
λs_gpu = CuArray{Float64}(collect(λs_korg))

# test tiles with different velocities and μ values
μ_vals = [0.95, 0.7, 0.4]
v_vals = [0.0, 1500.0, 3000.0]
B = length(μ_vals)
Natm1 = Natm - 1

@testset "Batched kernel equivalence" begin

    @testset "Batched convolution matches single-tile" begin
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        # build batched velocity array
        μ_v_batch_cpu = zeros(B * Natm)
        for bi in 1:B
            for k in 1:Natm
                μ_v_batch_cpu[(bi-1)*Natm + k] = v_vals[bi]
            end
        end
        μ_v_batch = CuArray{Float64}(μ_v_batch_cpu)

        # batched result
        bcmem.signal_cached = false
        αs_batch = Array(FT.convolve_wavelength_axis_batched!(bcmem, λs_korg, αs, μ_v_batch, σ_v, B))

        # single-tile results
        for bi in 1:B
            cmem.signal_cached = false
            μ_v = CUDA.zeros(Float64, Natm) .+ v_vals[bi]
            αs_single = Array(FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs, μ_v, σ_v))
            αs_batch_tile = αs_batch[(bi-1)*Natm+1 : bi*Natm, :]
            @test αs_batch_tile ≈ αs_single atol=1e-10
        end
    end

    @testset "Batched anchored tau matches single-tile" begin
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        # set up batched convolved opacities
        μ_v_batch_cpu = zeros(B * Natm)
        for bi in 1:B
            for k in 1:Natm; μ_v_batch_cpu[(bi-1)*Natm+k] = v_vals[bi]; end
        end
        μ_v_batch = CuArray{Float64}(μ_v_batch_cpu)
        bcmem.signal_cached = false
        αs_conv = FT.convolve_wavelength_axis_batched!(bcmem, λs_korg, αs, μ_v_batch, σ_v, B)

        # batched tau
        μ_tiles = CuArray{Float64}(μ_vals)
        τs_batch = CUDA.zeros(Float64, B * Natm, Nλ)
        FT.calc_tau_anchored_batched!(μ_tiles, log_τ_ref, ifactor_base, αs_conv, τs_batch, Natm, B)
        τs_batch_h = Array(τs_batch)

        # single-tile tau
        gpu_mem = FT.GPUMemory(λs_korg, atm_gpu, α_ref)
        for bi in 1:B
            cmem.signal_cached = false
            μ_v = CUDA.zeros(Float64, Natm) .+ v_vals[bi]
            αs_single = FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs, μ_v, σ_v)
            FT.calc_tau_anchored_gpu!(μ_vals[bi], log_τ_ref, ifactor_base, αs_single, gpu_mem.τs)
            τs_single = Array(gpu_mem.τs)
            τs_batch_tile = τs_batch_h[(bi-1)*Natm+1 : bi*Natm, :]
            @test τs_batch_tile ≈ τs_single atol=1e-10
        end
    end

    @testset "Batched cfunc_dt matches single-tile" begin
        # recompute absorption fresh to avoid state pollution from prior tests
        αs_fresh = zeros(Natm, Nλ)
        αs_fresh_c = zeros(Natm, Nλ)
        α_ref_fresh = zeros(Natm)
        FT.compute_alpha!(αs_fresh, αs_fresh_c, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                          α_ref_out=α_ref_fresh, ne_warn_thresh=Inf)
        log_τ_fresh = CuArray{Float64}(log.(atm_gpu.τs))
        ifact_fresh = CuArray{Float64}(atm_gpu.τs ./ α_ref_fresh)

        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        μ_v_batch_cpu = zeros(B * Natm)
        for bi in 1:B
            for k in 1:Natm; μ_v_batch_cpu[(bi-1)*Natm+k] = v_vals[bi]; end
        end
        μ_v_batch = CuArray{Float64}(μ_v_batch_cpu)
        μ_tiles = CuArray{Float64}(μ_vals)

        # batched pipeline: convolve → tau → cfunc_dt
        bcmem.signal_cached = false
        αs_conv = FT.convolve_wavelength_axis_batched!(bcmem, λs_korg, αs_fresh, μ_v_batch, σ_v, B)
        τs_batch = CUDA.zeros(Float64, B * Natm, Nλ)
        FT.calc_tau_anchored_batched!(μ_tiles, log_τ_fresh, ifact_fresh, αs_conv, τs_batch, Natm, B)
        cfdt_batch = CUDA.zeros(Float64, B * Natm1, Nλ)
        FT.calc_intensity_cfunc_dt_batched!(cfdt_batch, τs_batch, Ts_gpu, λs_gpu, Natm, B)
        cfdt_batch_h = Array(cfdt_batch)

        # single-tile pipeline (fresh objects per tile)
        for bi in 1:B
            gpu_mem_bi = FT.GPUMemory(λs_korg, atm_gpu, α_ref_fresh)
            cmem_bi = FT.ConvolutionMemory(Nλ, Natm, Npad)
            cmem_bi.signal_cached = false
            μ_v = CUDA.zeros(Float64, Natm) .+ v_vals[bi]
            result = FT.calc_intensity_quantities_inplace!(αs_fresh, atm_gpu, gpu_mem_bi, cmem_bi,
                μ_vals[bi], μ_v, σ_v)
            cfdt_single = Array(result.cfunc_dt)
            cfdt_batch_tile = cfdt_batch_h[(bi-1)*Natm1+1 : bi*Natm1, :]
            # looser tolerance: the batched and single-tile paths use different
            # ConvolutionMemory instances with separately-computed cuFFT plans,
            # which can produce ~1e-12 differences that compound through tau+cfunc
            @test cfdt_batch_tile ≈ cfdt_single rtol=1e-6
        end
    end

    @testset "Batched accumulation matches sequential" begin
        # create test cfunc_dt data
        cfdt_data = CUDA.rand(Float64, B * Natm1, Nλ) .* 1e-10
        dA = CuArray{Float64}([0.001, 0.002, 0.0015])

        # batched accumulation
        flux_b = CUDA.zeros(Float64, Nλ)
        cfunc_b = CUDA.zeros(Float64, Natm1, Nλ)
        flux_bc = CUDA.zeros(Float64, Nλ)
        cfunc_bc = CUDA.zeros(Float64, Natm1, Nλ)
        FT.accumulate_batch!(flux_b, cfunc_b, flux_bc, cfunc_bc, cfdt_data, dA, Natm1, B)

        # sequential accumulation
        flux_s = CUDA.zeros(Float64, Nλ)
        cfunc_s = CUDA.zeros(Float64, Natm1, Nλ)
        for bi in 1:B
            tile = @view cfdt_data[(bi-1)*Natm1+1 : bi*Natm1, :]
            # manual accumulate: can't use accumulate_tile! on SubArray, do it with broadcasts
            cfunc_s .+= tile .* Array(dA)[bi]
            flux_s .+= vec(sum(tile .* Array(dA)[bi], dims=1))
        end

        @test Array(flux_b) ≈ Array(flux_s) rtol=1e-10
        @test Array(cfunc_b) ≈ Array(cfunc_s) rtol=1e-10
    end

    @testset "Partial batch (Bcur < B) correctness" begin
        B_big = 8
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B_big, Npad)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        Bcur = 2  # only use 2 of 8

        μ_v_batch_cpu = zeros(Bcur * Natm)
        for bi in 1:Bcur
            for k in 1:Natm; μ_v_batch_cpu[(bi-1)*Natm+k] = v_vals[bi]; end
        end
        μ_v_batch = CuArray{Float64}(μ_v_batch_cpu)

        bcmem.signal_cached = false
        αs_batch = Array(FT.convolve_wavelength_axis_batched!(bcmem, λs_korg, αs, μ_v_batch, σ_v, Bcur))

        for bi in 1:Bcur
            cmem.signal_cached = false
            μ_v = CUDA.zeros(Float64, Natm) .+ v_vals[bi]
            αs_single = Array(FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs, μ_v, σ_v))
            @test αs_batch[(bi-1)*Natm+1:bi*Natm, :] ≈ αs_single atol=1e-10
        end
    end
end
