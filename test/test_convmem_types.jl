# Tests that both ConvolutionMemory (micro) and MacroConvolutionMemory dispatch
# correctly through AbstractConvolutionMemory, and that the type hierarchy works
# end-to-end for all GPU convolution paths.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Test

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

gpu_mem = FT.GPUMemory(λs_korg, atm_gpu, α_ref)
σ_v = CUDA.zeros(Float64, Natm) .+ ξ
μ_v_rot = CUDA.zeros(Float64, Natm) .+ 1200.0
μ_tile = 0.85

@testset "ConvolutionMemory type hierarchy" begin
    @testset "type relationships" begin
        cmem_micro = FT.ConvolutionMemory(Nλ, Natm, Npad)
        cmem_macro = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)

        @test cmem_micro isa FT.AbstractConvolutionMemory
        @test cmem_macro isa FT.AbstractConvolutionMemory
        @test cmem_micro isa FT.ConvolutionMemory
        @test cmem_macro isa FT.MacroConvolutionMemory
        @test !(cmem_micro isa FT.MacroConvolutionMemory)
        @test !(cmem_macro isa FT.ConvolutionMemory)
    end

    @testset "convolve_wavelength_axis_gpu with ConvolutionMemory" begin
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        result = Array(FT.convolve_wavelength_axis_gpu(cmem, λs_korg, αs, μ_v_rot, σ_v))
        @test size(result) == (Natm, Nλ)
        @test all(isfinite, result)
    end

    @testset "convolve_wavelength_axis_gpu with MacroConvolutionMemory" begin
        cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm, Npad)
        result = Array(FT.convolve_wavelength_axis_gpu(cmem_mac, λs_korg, αs, μ_v_rot, σ_v))
        @test size(result) == (Natm, Nλ)
        @test all(isfinite, result)
    end

    @testset "micro and macro produce identical convolution results" begin
        cmem_micro = FT.ConvolutionMemory(Nλ, Natm, Npad)
        cmem_macro = FT.MacroConvolutionMemory(Nλ, Natm, Npad)

        result_micro = Array(FT.convolve_wavelength_axis_gpu(cmem_micro, λs_korg, αs, μ_v_rot, σ_v))
        result_macro = Array(FT.convolve_wavelength_axis_gpu(cmem_macro, λs_korg, αs, μ_v_rot, σ_v))

        @test result_micro == result_macro
    end

    @testset "calc_intensity_quantities_inplace! works with ConvolutionMemory" begin
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        cmem.signal_cached = false
        result = FT.calc_intensity_quantities_inplace!(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
        @test size(result.cfunc) == (Natm - 1, Nλ)
        @test size(result.cfunc_dt) == (Natm - 1, Nλ)
        @test all(isfinite, Array(result.cfunc))
    end

    @testset "calc_flux_quantities works with ConvolutionMemory" begin
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        cmem.signal_cached = false
        result = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
        @test size(result.cfunc) == (Natm - 1, Nλ)
        @test size(result.cfunc_dt) == (Natm - 1, Nλ)
        @test all(isfinite, Array(result.cfunc))
    end

    @testset "MacroConvolutionMemory-specific fields" begin
        cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)

        # macro-specific fields exist and have correct sizes
        @test length(cmem_mac.padded_kernel_gpu) == cmem_mac.L
        @test length(cmem_mac.shift_kernel_gpu) == cmem_mac.L
        @test length(cmem_mac.xs_gpu) == Nλ
        @test size(cmem_mac.out_gpu) == (Natm - 1, Nλ)
        @test length(cmem_mac.kr_1d) == cmem_mac.L
        @test length(cmem_mac.kernel_row_ft_1d) == fld(cmem_mac.L, 2) + 1
        @test length(cmem_mac.kc_1d) == Nλ

        # these fields should NOT exist on ConvolutionMemory
        cmem_micro = FT.ConvolutionMemory(Nλ, Natm, Npad)
        @test !hasproperty(cmem_micro, :padded_kernel_gpu)
        @test !hasproperty(cmem_micro, :shift_kernel_gpu)
        @test !hasproperty(cmem_micro, :out_gpu)
        # xs_gpu, kr_1d, plan_fwd_1d now exist on both (micro kernel infrastructure)
        @test hasproperty(cmem_micro, :xs_gpu)
        @test hasproperty(cmem_micro, :kr_1d)
        @test hasproperty(cmem_micro, :plan_fwd_1d)
        @test hasproperty(cmem_micro, :kernel_cached)
    end

    @testset "removed fields are absent" begin
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        @test !hasproperty(cmem, :λc_gpu)
        @test !hasproperty(cmem, :σ_fac_gpu)
        @test !hasproperty(cmem, :σ_v_cpu)
        @test !hasproperty(cmem, :μ_v_cpu)

        @test !hasproperty(gpu_mem, :flux)

        @test !hasproperty(atm_gpu, :vx)
        @test !hasproperty(atm_gpu, :vy)
        @test !hasproperty(atm_gpu, :vz)
    end
end
