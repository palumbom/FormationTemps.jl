let
# test that the anchored and Bézier τ integrators produce physically consistent
# results, and that the fused tau+cfunc kernel matches the unfused pipeline.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Test
using Statistics

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
A_X = Korg.format_A_X(0.0)
Npad = 512
Δλ = 0.01

atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm = length(atm_gpu.zs)
Natm1 = Natm - 1

wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
Nλ = length(λs_korg)

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  α_ref_out=α_ref, ne_warn_thresh=Inf)

# anchored τ constants
log_τ_ref = CuArray{Float64}(log.(atm_gpu.τs))
ifactor_base = CuArray{Float64}(atm_gpu.τs ./ α_ref)
Ts_gpu = CuArray{Float64}(atm_gpu.Ts)
λs_gpu = CuArray{Float64}(collect(λs_korg))

μ_vals = [1.0, 0.8, 0.5, 0.2]

@testset "τ integrator equivalence" begin

    @testset "single-tile τ sanity: anchored at μ=$μ" for μ in μ_vals
        gpu_mem = FT.GPUMemory(λs_korg, atm_gpu, α_ref)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        μ_v = CUDA.zeros(Float64, Natm)
        σ_v = CUDA.zeros(Float64, Natm) .+ 850.0
        αs_conv = FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs, μ_v, σ_v)
        FT.calc_tau_anchored_gpu!(μ, log_τ_ref, ifactor_base, αs_conv, gpu_mem.τs)
        τ = Array(gpu_mem.τs)

        @test all(isfinite, τ)
        @test all(x -> x >= 0, τ)
        @test all(τ[1, :] .== 0.0)
        # monotonically non-decreasing with depth at every wavelength
        @test all(diff(τ, dims=1) .>= 0)
        @test maximum(τ[end, :]) > 10.0
    end

    @testset "single-tile τ sanity: Bézier at μ=$μ" for μ in μ_vals
        gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        μ_v = CUDA.zeros(Float64, Natm)
        σ_v = CUDA.zeros(Float64, Natm) .+ 850.0
        αs_conv = FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs, μ_v, σ_v)
        FT.calc_tau_bezier_cached!(μ, atm_gpu.zs_gpu, αs_conv, gpu_mem.τs,
                                   gpu_mem.tau_ds, gpu_mem.tau_alphaC)
        τ = Array(gpu_mem.τs)

        @test all(isfinite, τ)
        @test all(x -> x >= 0, τ)
        # Bézier initializes τ[1,:] = TAU_FLOOR (1e-5) to avoid log(0)
        @test all(τ[1, :] .== FT.TAU_FLOOR)
        @test all(diff(τ, dims=1) .>= 0)
        @test maximum(τ[end, :]) > 10.0
    end

    @testset "anchored vs Bézier: intensity agreement at μ=$μ" for μ in μ_vals
        gpu_mem_a = FT.GPUMemory(λs_korg, atm_gpu, α_ref)
        cmem_a = FT.ConvolutionMemory(Nλ, Natm, Npad)
        μ_v = CUDA.zeros(Float64, Natm)
        σ_v = CUDA.zeros(Float64, Natm) .+ 850.0
        result_a = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem_a, cmem_a, μ, μ_v, σ_v)
        I_anchored = Array(FT.get_intensity(result_a))

        gpu_mem_b = FT.GPUMemory(λs_korg, atm_gpu)
        cmem_b = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result_b = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem_b, cmem_b, μ, μ_v, σ_v)
        I_bezier = Array(FT.get_intensity(result_b))

        @test all(isfinite, I_anchored)
        @test all(isfinite, I_bezier)
        @test all(x -> x > 0, I_anchored)
        @test all(x -> x > 0, I_bezier)

        # intensity should agree within a few percent despite different τ quadrature.
        # observed: mean ~1-4%, max ~5-6% (largest in line cores at grazing μ)
        rel_err = abs.(I_anchored .- I_bezier) ./ I_anchored
        @test mean(rel_err) < 0.05
        @test maximum(rel_err) < 0.10
    end

    @testset "fused tau+cfunc matches unfused pipeline" begin
        B = length(μ_vals)
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)
        μ_tiles = CuArray{Float64}(μ_vals)
        μ_v_batch = CUDA.zeros(Float64, B * Natm)
        σ_v_scalar = 850.0

        bcmem.signal_cached = false
        αs_conv = FT.convolve_wavelength_axis_batched!(
            bcmem, λs_korg, αs, μ_v_batch, σ_v_scalar, B)

        # unfused: anchored tau → cfunc_dt (separate kernels)
        τs_batch = CUDA.zeros(Float64, B * Natm, Nλ)
        FT.calc_tau_anchored_batched!(μ_tiles, log_τ_ref, ifactor_base,
                                      αs_conv, τs_batch, Natm, B)
        cfdt_unfused = CUDA.zeros(Float64, B * Natm1, Nλ)
        FT.calc_intensity_cfunc_dt_batched!(cfdt_unfused, τs_batch, Ts_gpu, λs_gpu, Natm, B)

        # fused: single kernel, τ in registers
        cfdt_fused = CUDA.zeros(Float64, B * Natm1, Nλ)
        FT.calc_tau_cfunc_dt_fused!(cfdt_fused, αs_conv, log_τ_ref, ifactor_base,
                                     μ_tiles, Ts_gpu, λs_gpu, Natm, B)

        cfdt_u = Array(cfdt_unfused)
        cfdt_f = Array(cfdt_fused)

        @test all(isfinite, cfdt_f)
        # identical math and thread mapping → bit-exact
        @test cfdt_f ≈ cfdt_u atol=1e-14
    end

    @testset "fused kernel with tile_offset" begin
        B = length(μ_vals)
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)
        μ_v_batch = CUDA.zeros(Float64, B * Natm)
        σ_v_scalar = 850.0

        # pad μ_tiles to test that tile_offset indexes correctly
        μ_padded = CuArray{Float64}([0.99, 0.88, μ_vals...])
        offset = 2

        bcmem.signal_cached = false
        αs_conv = FT.convolve_wavelength_axis_batched!(
            bcmem, λs_korg, αs, μ_v_batch, σ_v_scalar, B)

        # with offset: reads μ_padded[3:6] = μ_vals
        cfdt_offset = CUDA.zeros(Float64, B * Natm1, Nλ)
        FT.calc_tau_cfunc_dt_fused!(cfdt_offset, αs_conv, log_τ_ref, ifactor_base,
                                     μ_padded, Ts_gpu, λs_gpu, Natm, B;
                                     tile_offset=offset)

        # without offset: reads μ_vals[1:4] directly
        μ_tiles = CuArray{Float64}(μ_vals)
        cfdt_direct = CUDA.zeros(Float64, B * Natm1, Nλ)
        FT.calc_tau_cfunc_dt_fused!(cfdt_direct, αs_conv, log_τ_ref, ifactor_base,
                                     μ_tiles, Ts_gpu, λs_gpu, Natm, B)

        @test Array(cfdt_offset) ≈ Array(cfdt_direct) atol=1e-14
    end
end

end
