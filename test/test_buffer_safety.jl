# Tests that GPU buffer-returning functions are safe for general use:
# - calc_intensity_quantities returns independent copies
# - calc_flux_quantities returns independent copies
# - GPU broadening functions return via cmem.out_gpu (callers must Array() before next call)
# - End-to-end disk integration CPU/GPU agreement
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics

# ── shared setup ───────────────────────────────────────────────────────────────
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
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
cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)
σ_v = CUDA.zeros(Float64, Natm) .+ ξ
μ_v_zero = CUDA.zeros(Float64, Natm)

# ── tests ──────────────────────────────────────────────────────────────────────
@testset "Buffer safety" begin
    @testset "calc_intensity_quantities returns independent copies" begin
        μ_v_rot = CUDA.zeros(Float64, Natm) .+ 500.0
        result1 = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, 0.9, μ_v_rot, σ_v)
        cfunc_dt_1 = Array(result1.cfunc_dt)

        # second call with different αs overwrites gpu_mem internals
        result2 = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, 0.5, μ_v_rot, σ_v)
        cfunc_dt_1_after = Array(result1.cfunc_dt)

        # result1 should be unchanged — it holds an independent copy
        @test cfunc_dt_1 ≈ cfunc_dt_1_after
        # and it should differ from result2 (different αs and μ)
        cfunc_dt_2 = Array(result2.cfunc_dt)
        @test !isapprox(cfunc_dt_1, cfunc_dt_2; atol=1e-20)
    end

    @testset "calc_flux_quantities returns independent copies" begin
        result1 = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
        cfunc_dt_1 = Array(result1.cfunc_dt)

        result2 = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v)
        cfunc_dt_1_after = Array(result1.cfunc_dt)

        @test cfunc_dt_1 ≈ cfunc_dt_1_after
        cfunc_dt_2 = Array(result2.cfunc_dt)
        @test !isapprox(cfunc_dt_1, cfunc_dt_2; atol=1e-20)
    end

    @testset "GPU broadening out_gpu: Array() captures before next call" begin
        # get a contribution function to convolve
        cfunc_flux = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
        tbc = Array(cfunc_flux.cfunc_dt)

        # first call
        result1_gpu = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_RT, 0.9)
        result1_captured = Array(result1_gpu)

        # second call with different μ overwrites cmem_mac.out_gpu
        result2_gpu = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_RT, 0.3)
        result2_captured = Array(result2_gpu)

        # the captured Array should be independent
        @test !isapprox(result1_captured, result2_captured; atol=1e-20)
        # and result1_captured should not have been corrupted by the second call
        @test sum(abs.(result1_captured)) > 0.0
        @test sum(abs.(result2_captured)) > 0.0
    end

    @testset "GPU broadening functions: iso_rt, gray_rot, hirano" begin
        cfunc_flux = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
        tbc = Array(cfunc_flux.cfunc_dt)

        # iso RT macro
        r1 = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_RT))
        r2 = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_RT * 0.5))
        @test !isapprox(r1, r2; atol=1e-20)
        @test sum(abs.(r1)) > 0.0

        # gray rotation
        r1 = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, tbc, vsini, 0.4))
        r2 = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, tbc, vsini * 0.5, 0.4))
        @test !isapprox(r1, r2; atol=1e-20)
        @test sum(abs.(r1)) > 0.0

        # hirano
        r1 = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini, ζ_RT, 0.43, 0.31))
        r2 = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini * 0.5, ζ_RT, 0.43, 0.31))
        @test !isapprox(r1, r2; atol=1e-20)
        @test sum(abs.(r1)) > 0.0
    end

    @testset "End-to-end disk integration CPU/GPU agreement" begin
        star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini,
                            v_macro=ζ_RT, v_micro=ξ)

        result_cpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=16,
                                         use_gpu=false, ne_warn_thresh=Inf)
        result_gpu = calc_formation_temp(star, linelist; Δλ=Δλ, Nϕ=16,
                                         use_gpu=true, ne_warn_thresh=Inf)

        @test length(result_gpu.wavs) == length(result_cpu.wavs)

        # flux agreement (micro broadening CPU/GPU difference ~4e-4)
        @test maximum(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-3
        @test mean(abs.(result_gpu.flux .- result_cpu.flux)) < 1e-4

        # formation temperatures: exclude edges where padding conventions differ
        λ0_val = mean(result_cpu.wavs)
        edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0_val)) + 10
        interior = (edge_px+1):(length(result_cpu.wavs) - edge_px)
        if length(interior) > 10
            @test maximum(abs.(result_gpu.form_temps[interior] .- result_cpu.form_temps[interior])) < 5.0
        end
    end
end
