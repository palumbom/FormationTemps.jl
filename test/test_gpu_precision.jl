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

        wls = [l.wl * 1e8 for l in linelist]
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
        Natm = length(atm32.zs)

        wls = [l.wl * 1e8 for l in linelist]
        λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
        Nλ = length(λs_korg)
        Npad = 512
        B = 3

        # Float64 absorption converted to Float32
        αs = zeros(Float64, Natm, Nλ)
        αs_cont = zeros(Float64, Natm, Nλ)
        α_ref = zeros(Float64, Natm)
        atm_f64 = FT.AtmosphereGPU(korg_atm; T=Float64)
        FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_f64, A_X;
                          α_ref_out=α_ref, ne_warn_thresh=Inf)
        αs32 = Float32.(αs)
        α_ref32 = Float32.(α_ref)
        λs32 = Float32.(collect(λs_korg))

        σ_v = CUDA.zeros(Float32, Natm) .+ Float32(ξ)
        log_τ_ref = CuArray{Float32}(log.(atm32.τs))
        ifactor_base = CuArray{Float32}(atm32.τs ./ α_ref32)
        Ts_gpu = CuArray{Float32}(atm32.Ts)
        λs_gpu = CuArray{Float32}(λs32)

        μ_vals = Float32[0.95, 0.7, 0.4]
        v_vals = Float32[0.0, 1500.0, 3000.0]
        Natm1 = Natm - 1

        # batched convolution
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad; T=Float32)
        μ_v_batch_cpu = zeros(Float32, B * Natm)
        for bi in 1:B
            for k in 1:Natm
                μ_v_batch_cpu[(bi-1)*Natm+k] = v_vals[bi]
            end
        end
        μ_v_batch = CuArray{Float32}(μ_v_batch_cpu)
        bcmem.signal_cached = false
        αs_conv = FT.convolve_wavelength_axis_batched!(bcmem, λs32, αs32, μ_v_batch, σ_v, B)
        @test eltype(αs_conv) == Float32

        # batched tau
        μ_tiles = CuArray{Float32}(μ_vals)
        τs_batch = CUDA.zeros(Float32, B * Natm, Nλ)
        FT.calc_tau_anchored_batched!(μ_tiles, log_τ_ref, ifactor_base, αs_conv, τs_batch, Natm, B)
        τs_h = Array(τs_batch)
        @test eltype(τs_h) == Float32
        @test all(isfinite, τs_h)

        # batched cfunc_dt
        cfdt_batch = CUDA.zeros(Float32, B * Natm1, Nλ)
        FT.calc_intensity_cfunc_dt_batched!(cfdt_batch, τs_batch, Ts_gpu, λs_gpu, Natm, B)
        cfdt_h = Array(cfdt_batch)
        @test eltype(cfdt_h) == Float32
        @test all(isfinite, cfdt_h)

        # batched accumulation
        flux_acc = CUDA.zeros(Float32, Nλ)
        cfunc_acc = CUDA.zeros(Float32, Natm1, Nλ)
        dA_tiles = CuArray{Float32}([0.001f0, 0.002f0, 0.0015f0])
        FT.accumulate_batch!(flux_acc, cfunc_acc, cfdt_batch, dA_tiles, Natm1, B)
        @test eltype(Array(flux_acc)) == Float32
        @test all(isfinite, Array(flux_acc))
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
        @test maximum(abs.(Float64.(res32.flux) .- res64.flux)) < 5e-3
        # mask edges for formation temp comparison (Hirano kernel edge effects)
        λ0 = mean(res64.wavs)
        edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
        interior = (edge_px+1):(length(res64.wavs) - edge_px)
        @test maximum(abs.(Float64.(res32.form_temps[interior]) .- res64.form_temps[interior])) < 10.0
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
        @test maximum(abs.(Float64.(res32.flux) .- res64.flux)) < 5e-3

        # formation temps: Float32 introduces ~1-5 K differences in disk integration
        λ0 = mean(res64.wavs)
        edge_px = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
        interior = (edge_px+1):(length(res64.wavs) - edge_px)
        @test maximum(abs.(Float64.(res32.form_temps[interior]) .- res64.form_temps[interior])) < 10.0

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

        wls = [l.wl * 1e8 for l in linelist]
        λs = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
        Nλ = length(λs)
        Npad = 512

        αs = zeros(Float64, Natm, Nλ)
        αs_cont = zeros(Float64, Natm, Nλ)
        α_ref = zeros(Float64, Natm)
        FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs), linelist, atm_f64, A_X;
                          α_ref_out=α_ref, ne_warn_thresh=Inf)

        # get a Float32 cfunc_dt to convolve
        gpu_mem = FT.GPUMemory(collect(λs), atm_f64, α_ref)
        cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
        σ_v = CUDA.zeros(Float64, Natm) .+ ξ
        cfunc_struct = FT.calc_flux_quantities(αs, atm_f64, gpu_mem, cmem, σ_v)
        tbc = Float32.(Array(cfunc_struct.cfunc_dt))

        λs32 = Float32.(collect(λs))
        cmem_mac32 = FT.MacroConvolutionMemory(Nλ, Natm1, Npad; T=Float32)

        # gray rotation
        res_gray = FT.convolve_gray_rotation_gpu(cmem_mac32, λs32, tbc,
                                                  Float32(vsini), Float32(0.43))
        @test eltype(res_gray) == Float32
        @test all(isfinite, Array(res_gray))

        # isotropic RT macro
        res_iso = FT.convolve_iso_rt_macro_gpu(cmem_mac32, λs32, tbc, Float32(ζ_RT))
        @test eltype(res_iso) == Float32
        @test all(isfinite, Array(res_iso))

        # anisotropic RT macro
        res_rt = FT.convolve_rt_macro_gpu(cmem_mac32, λs32, tbc, Float32(ζ_RT), Float32(0.9))
        @test eltype(res_rt) == Float32
        @test all(isfinite, Array(res_rt))

        # Hirano rotation+macro
        res_hirano = FT.convolve_hirano_rotmacro_gpu(cmem_mac32, λs32, tbc,
                                                      Float32(vsini), Float32(ζ_RT),
                                                      Float32(0.43), Float32(0.31))
        @test eltype(res_hirano) == Float32
        @test all(isfinite, Array(res_hirano))
    end
end
