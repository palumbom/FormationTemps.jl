let
# Tests that fused GPU kernels produce bit-identical results to the
# non-fused (separate kernel launch) equivalents.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Test

# ── shared setup ───────────────────────────────────────────────────────────────
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
Fe_H = 0.0
ξ = 850.0
Δλ = 0.01

A_X = Korg.format_A_X(Fe_H)
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm = length(atm_gpu.zs)

wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
Nλ = length(λs_korg)

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  α_ref_out=α_ref, ne_warn_thresh=Inf)

Npad = 512
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu, α_ref)
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
v_mic = CUDA.zeros(Float64, Natm) .+ ξ
μ_tile = 0.85
v_los_rot = CUDA.zeros(Float64, Natm) .+ 1200.0

# ── tests ──────────────────────────────────────────────────────────────────────
@testset "Fused kernel equivalence" begin

    @testset "Fused cfunc+cfunc_dt vs separate (intensity)" begin
        # non-fused path: calc_intensity_quantities uses separate cfunc + compute_cfunc_dt
        cmem.signal_cached = false
        result_nonfused = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μ_tile, v_los_rot, v_mic)
        cfunc_nf = Array(result_nonfused.cfunc)
        cfunc_dt_nf = Array(result_nonfused.cfunc_dt)

        # fused path: calc_intensity_quantities_inplace! uses calc_intensity_cfunc_dt!
        cmem.signal_cached = false
        result_fused = FT.calc_intensity_quantities_inplace!(αs, atm_gpu, gpu_mem, cmem, μ_tile, v_los_rot, v_mic)
        cfunc_f = Array(result_fused.cfunc)
        cfunc_dt_f = Array(result_fused.cfunc_dt)

        @test cfunc_f == cfunc_nf
        @test cfunc_dt_f == cfunc_dt_nf
    end

    @testset "Fused cfunc+cfunc_dt vs separate (flux)" begin
        # non-fused path: call the old wrapper + compute_cfunc_dt separately
        cmem.signal_cached = false
        FT.calc_flux_cfunc!(αs, atm_gpu, gpu_mem, cmem, v_mic)
        FT.compute_cfunc_dt!(gpu_mem.cfunc_dt, gpu_mem.cfunc, gpu_mem.τs)
        cfunc_nf = Array(gpu_mem.cfunc)
        cfunc_dt_nf = Array(gpu_mem.cfunc_dt)

        # fused path: calc_flux_quantities uses calc_flux_cfunc_dt!
        cmem.signal_cached = false
        result_fused = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, v_mic)
        cfunc_f = Array(result_fused.cfunc)
        cfunc_dt_f = Array(result_fused.cfunc_dt)

        @test cfunc_f == cfunc_nf
        @test cfunc_dt_f == cfunc_dt_nf
    end

    @testset "Fused accumulation vs separate copyto+sum+broadcast" begin
        # create a source matrix (use cfunc_dt from a real computation)
        cmem.signal_cached = false
        result = FT.calc_intensity_quantities_inplace!(αs, atm_gpu, gpu_mem, cmem, μ_tile, v_los_rot, v_mic)
        src = copy(result.cfunc_dt)
        Natm1 = Natm - 1
        dA_i = 0.00123

        # non-fused: copyto + sum! + broadcasts
        flux_nf = CUDA.zeros(Float64, Nλ)
        cfunc_nf = CUDA.zeros(Float64, Natm1, Nλ)
        tile_buf = CUDA.zeros(Float64, 1, Nλ)
        copyto!(cfunc_nf, src)
        # undo the copyto to make cfunc_nf an accumulator
        cfunc_nf .= zero(Float64)
        sum!(tile_buf, src)
        flux_nf .+= vec(tile_buf) .* dA_i
        cfunc_nf .+= src .* dA_i

        # fused: one kernel
        flux_f = CUDA.zeros(Float64, Nλ)
        cfunc_f = CUDA.zeros(Float64, Natm1, Nλ)
        flux_c = CUDA.zeros(Float64, Nλ)
        cfunc_c = CUDA.zeros(Float64, Natm1, Nλ)
        FT.accumulate_tile!(flux_f, cfunc_f, flux_c, cfunc_c, src, dA_i)

        @test Array(flux_f) ≈ Array(flux_nf)
        @test Array(cfunc_f) ≈ Array(cfunc_nf)
    end
end

end
