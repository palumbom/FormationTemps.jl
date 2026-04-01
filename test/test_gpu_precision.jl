# Tests that gpu_precision=Float32 produces correct results and that all GPU structs
# are properly typed at the requested precision.
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
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0
Δλ = 0.01

star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

@testset "GPU precision control" begin

    @testset "AtmosphereGPU{Float32} construction" begin
        A_X = Korg.format_A_X(Fe_H)
        korg_atm = Korg.interpolate_marcs(Teff, logg, A_X)
        atm32 = FT.AtmosphereGPU(korg_atm; T=Float32)
        atm64 = FT.AtmosphereGPU(korg_atm; T=Float64)

        @test atm32 isa FT.AtmosphereGPU{Float32}
        @test eltype(atm32.zs) == Float32
        @test eltype(atm32.Ts) == Float32
        @test eltype(atm32.τs) == Float32
        @test eltype(atm32.nₑ) == Float32
        @test eltype(atm32.nd) == Float32
        @test atm32.reference_wavelength isa Float32
        @test eltype(atm32.zs_gpu) == Float32
        @test eltype(atm32.Ts_gpu) == Float32
        @test eltype(atm32.nd_gpu) == Float32

        # values should match after conversion
        @test Float32.(atm64.Ts) ≈ atm32.Ts
        @test Float32.(atm64.zs) ≈ atm32.zs
    end

    @testset "GPU struct allocation at Float32" begin
        Nλ = 200
        Natm = 80
        Npad = 512

        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad; T=Float32)
        @test cmem isa FT.ConvolutionMemory{Float32}

        cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad; T=Float32)
        @test cmem_mac isa FT.MacroConvolutionMemory{Float32}

        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, 4, Npad; T=Float32)
        @test bcmem isa FT.BatchedMicroConvMem{Float32}
        @test eltype(bcmem.signal_gpu) == Float32
        @test eltype(bcmem.conv_gpu) == Float32
    end

    @testset "Batched convolution at Float32" begin
        A_X = Korg.format_A_X(Fe_H)
        korg_atm = Korg.interpolate_marcs(Teff, logg, A_X)
        atm_f64 = FT.AtmosphereGPU(korg_atm; T=Float64)
        Natm = length(atm_f64.zs)

        wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
        λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
        Nλ = length(λs_korg)
        Npad = 512

        # compute absorption at Float64
        αs = zeros(Float64, Natm, Nλ)
        αs_cont = zeros(Float64, Natm, Nλ)
        FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_f64, A_X;
                          ne_warn_thresh=Inf)

        # Float64 reference
        σ_v_64 = CUDA.zeros(Float64, Natm) .+ ξ
        μ_v_64 = CUDA.zeros(Float64, Natm) .+ 1500.0
        cmem64 = FT.ConvolutionMemory(Nλ, Natm, Npad; T=Float64)
        cmem64.signal_cached = false
        ref = Array(FT.convolve_wavelength_axis_gpu(cmem64, collect(λs_korg), αs, μ_v_64, σ_v_64))

        # Float32
        αs32 = Float32.(αs)
        λs32 = Float32.(collect(λs_korg))
        σ_v_32 = CUDA.zeros(Float32, Natm) .+ Float32(ξ)
        μ_v_32 = CUDA.zeros(Float32, Natm) .+ Float32(1500.0)
        cmem32 = FT.ConvolutionMemory(Nλ, Natm, Npad; T=Float32)
        cmem32.signal_cached = false
        res32 = Array(FT.convolve_wavelength_axis_gpu(cmem32, λs32, αs32, μ_v_32, σ_v_32))

        @test size(res32) == size(ref)
        @test eltype(res32) == Float32
        # Float32 vs Float64 microturbulence should agree within single-precision tolerance
        # Float32 analytical Fourier Gaussian differs from Float64 sampled kernel by ~0.4%
        @test maximum(abs.(Float64.(res32) .- ref)) / maximum(abs.(ref)) < 5e-3
    end

    @testset "Batched kernels at Float32" begin
        A_X = Korg.format_A_X(Fe_H)
        korg_atm = Korg.interpolate_marcs(Teff, logg, A_X)
        atm32 = FT.AtmosphereGPU(korg_atm; T=Float32)
        atm_f64 = FT.AtmosphereGPU(korg_atm; T=Float64)
        Natm = length(atm32.zs)

        wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
        λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
        Nλ = length(λs_korg)
        Npad = 512
        B = 3

        # Float64 absorption
        αs = zeros(Float64, Natm, Nλ)
        αs_cont = zeros(Float64, Natm, Nλ)
        α_ref = zeros(Float64, Natm)
        FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_f64, A_X;
                          α_ref_out=α_ref, ne_warn_thresh=Inf)
        αs32 = Float32.(αs)
        α_ref32 = Float32.(α_ref)
        λs32 = Float32.(collect(λs_korg))
        λs64 = collect(Float64, λs_korg)

        μ_vals_f32 = Float32[0.95, 0.7, 0.4]
        μ_vals_f64 = Float64[0.95, 0.7, 0.4]
        v_vals_f32 = Float32[0.0, 1500.0, 3000.0]
        v_vals_f64 = Float64[0.0, 1500.0, 3000.0]
        Natm1 = Natm - 1

        # ── Float64 reference path ──
        σ_v_64 = Float64(ξ)
        log_τ_ref_64 = CuArray{Float64}(log.(atm_f64.τs))
        ifactor_base_64 = CuArray{Float64}(atm_f64.τs ./ α_ref)
        Ts_gpu_64 = CuArray{Float64}(atm_f64.Ts)
        λs_gpu_64 = CuArray{Float64}(λs64)

        bcmem_64 = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad; T=Float64)
        μ_v_batch_cpu_64 = zeros(Float64, B * Natm)
        for bi in 1:B, k in 1:Natm
            μ_v_batch_cpu_64[(bi-1)*Natm+k] = v_vals_f64[bi]
        end
        μ_v_batch_64 = CuArray{Float64}(μ_v_batch_cpu_64)
        bcmem_64.signal_cached = false
        αs_conv_64 = FT.convolve_wavelength_axis_batched!(bcmem_64, λs64, αs, μ_v_batch_64, σ_v_64, B)

        μ_tiles_64 = CuArray{Float64}(μ_vals_f64)
        τs_batch_64 = CUDA.zeros(Float64, B * Natm, Nλ)
        FT.calc_tau_anchored_batched!(μ_tiles_64, log_τ_ref_64, ifactor_base_64,
                                      αs_conv_64, τs_batch_64, Natm, B)
        cfdt_batch_64 = CUDA.zeros(Float64, B * Natm1, Nλ)
        FT.calc_intensity_cfunc_dt_batched!(cfdt_batch_64, τs_batch_64, Ts_gpu_64, λs_gpu_64, Natm, B)
        τs_ref = Array(τs_batch_64)
        cfdt_ref = Array(cfdt_batch_64)

        # ── Float32 path ──
        σ_v_32 = Float32(ξ)
        log_τ_ref_32 = CuArray{Float32}(log.(atm32.τs))
        ifactor_base_32 = CuArray{Float32}(atm32.τs ./ α_ref32)
        Ts_gpu_32 = CuArray{Float32}(atm32.Ts)
        λs_gpu_32 = CuArray{Float32}(λs32)

        bcmem_32 = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad; T=Float32)
        μ_v_batch_cpu_32 = zeros(Float32, B * Natm)
        for bi in 1:B, k in 1:Natm
            μ_v_batch_cpu_32[(bi-1)*Natm+k] = v_vals_f32[bi]
        end
        μ_v_batch_32 = CuArray{Float32}(μ_v_batch_cpu_32)
        bcmem_32.signal_cached = false
        αs_conv_32 = FT.convolve_wavelength_axis_batched!(bcmem_32, λs32, αs32, μ_v_batch_32, σ_v_32, B)
        @test eltype(αs_conv_32) == Float32

        μ_tiles_32 = CuArray{Float32}(μ_vals_f32)
        τs_batch_32 = CUDA.zeros(Float32, B * Natm, Nλ)
        FT.calc_tau_anchored_batched!(μ_tiles_32, log_τ_ref_32, ifactor_base_32,
                                      αs_conv_32, τs_batch_32, Natm, B)
        τs_h = Array(τs_batch_32)
        @test eltype(τs_h) == Float32
        # τ relative error: Float32 FFT convolution introduces ~1e-3 relative error
        @test maximum(abs.(Float64.(τs_h) .- τs_ref)) / maximum(abs.(τs_ref)) < 5e-3

        cfdt_batch_32 = CUDA.zeros(Float32, B * Natm1, Nλ)
        FT.calc_intensity_cfunc_dt_batched!(cfdt_batch_32, τs_batch_32, Ts_gpu_32, λs_gpu_32, Natm, B)
        cfdt_h = Array(cfdt_batch_32)
        @test eltype(cfdt_h) == Float32
        # cfunc_dt relative error: dominated by Float32 FFT convolution of αs
        @test maximum(abs.(Float64.(cfdt_h) .- cfdt_ref)) / maximum(abs.(cfdt_ref)) < 2e-2

        # batched accumulation
        flux_acc = CUDA.zeros(Float32, Nλ)
        cfunc_acc = CUDA.zeros(Float32, Natm1, Nλ)
        flux_comp = CUDA.zeros(Float32, Nλ)
        cfunc_comp = CUDA.zeros(Float32, Natm1, Nλ)
        dA_tiles = CuArray{Float32}([0.001f0, 0.002f0, 0.0015f0])
        FT.accumulate_batch!(flux_acc, cfunc_acc, flux_comp, cfunc_comp, cfdt_batch_32, dA_tiles, Natm1, B)

        flux_acc_64 = CUDA.zeros(Float64, Nλ)
        cfunc_acc_64 = CUDA.zeros(Float64, Natm1, Nλ)
        flux_comp_64 = CUDA.zeros(Float64, Nλ)
        cfunc_comp_64 = CUDA.zeros(Float64, Natm1, Nλ)
        dA_tiles_64 = CuArray{Float64}([0.001, 0.002, 0.0015])
        FT.accumulate_batch!(flux_acc_64, cfunc_acc_64, flux_comp_64, cfunc_comp_64,
                             cfdt_batch_64, dA_tiles_64, Natm1, B)
        @test eltype(Array(flux_acc)) == Float32
        # accumulated flux: error from per-tile Float32 cfunc_dt, not accumulation
        @test maximum(abs.(Float64.(Array(flux_acc)) .- Array(flux_acc_64))) /
              maximum(abs.(Array(flux_acc_64))) < 2e-2
    end

    @testset "End-to-end: gpu_precision=Float32 convolve=true" begin
        u1 = 0.43
        u2 = 0.31
        res64 = calc_formation_temp(star, linelist; Δλ=Δλ, gpu_precision=Float64,
                                    convolve=true, u1=u1, u2=u2, Nϕ=32,
                                    showprogress=false, ne_warn_thresh=Inf)
        res32 = calc_formation_temp(star, linelist; Δλ=Δλ, gpu_precision=Float32,
                                    convolve=true, u1=u1, u2=u2, Nϕ=32,
                                    showprogress=false, ne_warn_thresh=Inf)

        @test res32 isa FT.FormTempResult{Float32}
        @test eltype(res32.wavs) == Float32
        @test eltype(res32.flux) == Float32
        @test eltype(res32.form_temps) == Float32
        @test eltype(res32.cont_func) == Float32
        @test res32.atmosphere isa FT.AtmosphereGPU{Float32}

        # Float32 vs Float64 agreement
        @test length(res32.wavs) == length(res64.wavs)
        @test all(isfinite, res32.flux)
        @test all(isfinite, res32.form_temps)
        # Float32 flux residuals are ~6e-4; formation temp ~1-2 K (see diagnose_f32_residuals.jl)
        @test maximum(abs.(Float64.(res32.flux) .- res64.flux)) < 2e-3
        # mask edges for formation temp comparison (Hirano kernel edge effects)
        λ0 = mean(res64.wavs)
        edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
        interior = (edge_px+1):(length(res64.wavs) - edge_px)
        @test maximum(abs.(Float64.(res32.form_temps[interior]) .- res64.form_temps[interior])) < 5.0
    end

    @testset "End-to-end: gpu_precision=Float32 disk integration" begin
        res64 = calc_formation_temp(star, linelist; Δλ=Δλ, gpu_precision=Float64,
                                    convolve=false, Nϕ=32,
                                    showprogress=false, ne_warn_thresh=Inf)
        res32 = calc_formation_temp(star, linelist; Δλ=Δλ, gpu_precision=Float32,
                                    convolve=false, Nϕ=32,
                                    showprogress=false, ne_warn_thresh=Inf)

        @test res32 isa FT.FormTempResult{Float32}
        @test eltype(res32.flux) == Float32
        @test eltype(res32.form_temps) == Float32

        @test all(isfinite, res32.flux)
        @test all(isfinite, res32.form_temps)
        # Float32 flux residuals ~6e-4; formation temp ~1-2 K (see diagnose_f32_residuals.jl)
        @test maximum(abs.(Float64.(res32.flux) .- res64.flux)) < 2e-3

        λ0 = mean(res64.wavs)
        edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
        interior = (edge_px+1):(length(res64.wavs) - edge_px)
        @test maximum(abs.(Float64.(res32.form_temps[interior]) .- res64.form_temps[interior])) < 5.0

        # formation temps should be within the atmospheric temperature range
        Ts = res32.atmosphere.Ts
        @test all(res32.form_temps .>= minimum(Ts))
        @test all(res32.form_temps .<= maximum(Ts))
    end

    @testset "gpu_precision=Float64 is default and matches prior behavior" begin
        res_default = calc_formation_temp(star, linelist; Δλ=Δλ, convolve=true,
                                          u1=0.43, u2=0.31, Nϕ=32,
                                          showprogress=false, ne_warn_thresh=Inf)
        res_explicit = calc_formation_temp(star, linelist; Δλ=Δλ, gpu_precision=Float64,
                                           convolve=true, u1=0.43, u2=0.31, Nϕ=32,
                                           showprogress=false, ne_warn_thresh=Inf)

        @test res_default.flux ≈ res_explicit.flux atol=1e-12
        @test res_default.form_temps ≈ res_explicit.form_temps atol=1e-12
    end

    @testset "Broadening kernels at Float32" begin
        A_X = Korg.format_A_X(Fe_H)
        korg_atm = Korg.interpolate_marcs(Teff, logg, A_X)
        atm_f64 = FT.AtmosphereGPU(korg_atm; T=Float64)
        Natm = length(atm_f64.zs)
        Natm1 = Natm - 1

        wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
        λs = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
        Nλ = length(λs)
        Npad = 512

        αs = zeros(Float64, Natm, Nλ)
        αs_cont = zeros(Float64, Natm, Nλ)
        α_ref = zeros(Float64, Natm)
        FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs), linelist, atm_f64, A_X;
                          α_ref_out=α_ref, ne_warn_thresh=Inf)

        # Float64 reference cfunc_dt
        gpu_mem = FT.GPUMemory(collect(λs), atm_f64, α_ref)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        σ_v = CUDA.zeros(Float64, Natm) .+ ξ
        cfunc_struct = FT.calc_flux_quantities(αs, atm_f64, gpu_mem, cmem, σ_v)
        tbc_f64 = Array(cfunc_struct.cfunc_dt)
        tbc = Float32.(tbc_f64)

        λs64 = collect(Float64, λs)
        λs32 = Float32.(λs64)
        cmem_mac64 = FT.MacroConvolutionMemory(Nλ, Natm1, Npad; T=Float64)
        cmem_mac32 = FT.MacroConvolutionMemory(Nλ, Natm1, Npad; T=Float32)

        # gray rotation: Float32 vs Float64 GPU
        ref_gray = Array(FT.convolve_gray_rotation_gpu(cmem_mac64, λs64, tbc_f64, vsini, 0.43))
        res_gray = Array(FT.convolve_gray_rotation_gpu(cmem_mac32, λs32, tbc, Float32(vsini), 0.43f0))
        @test eltype(res_gray) == Float32
        @test maximum(abs.(Float64.(res_gray) .- ref_gray)) / maximum(abs.(ref_gray)) < 5e-3

        # isotropic RT macro
        ref_iso = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac64, λs64, tbc_f64, ζ_RT))
        res_iso = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac32, λs32, tbc, Float32(ζ_RT)))
        @test eltype(res_iso) == Float32
        @test maximum(abs.(Float64.(res_iso) .- ref_iso)) / maximum(abs.(ref_iso)) < 5e-3

        # anisotropic RT macro
        ref_rt = Array(FT.convolve_rt_macro_gpu(cmem_mac64, λs64, tbc_f64, ζ_RT, 0.9))
        res_rt = Array(FT.convolve_rt_macro_gpu(cmem_mac32, λs32, tbc, Float32(ζ_RT), 0.9f0))
        @test eltype(res_rt) == Float32
        @test maximum(abs.(Float64.(res_rt) .- ref_rt)) / maximum(abs.(ref_rt)) < 5e-3

        # Hirano rotation+macro
        ref_hirano = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac64, λs64, tbc_f64, vsini, ζ_RT, 0.43, 0.31))
        res_hirano = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac32, λs32, tbc,
                                                            Float32(vsini), Float32(ζ_RT), 0.43f0, 0.31f0))
        @test eltype(res_hirano) == Float32
        @test maximum(abs.(Float64.(res_hirano) .- ref_hirano)) / maximum(abs.(ref_hirano)) < 5e-3
    end
end
