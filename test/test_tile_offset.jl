# Tests that tile_offset parameter correctly indexes into pre-uploaded arrays.
# Verifies: processing tiles with offset gives same results as processing
# the corresponding slice with offset=0.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using LinearAlgebra
using Test
using Statistics

if !CUDA.functional()
    @info "CUDA not available, skipping tile_offset tests"
    exit()
end

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
Natm1 = Natm - 1

wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
Nλ = length(λs_korg)

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  α_ref_out=α_ref, ne_warn_thresh=Inf)

log_τ_ref = CuArray{Float64}(log.(atm_gpu.τs))
ifactor_base = CuArray{Float64}(atm_gpu.τs ./ α_ref)
Ts_gpu = CuArray{Float64}(atm_gpu.Ts)
λs_gpu = CuArray{Float64}(collect(λs_korg))

# 6 tiles with varying μ and velocity
Ntiles = 6
μ_vals = [0.95, 0.8, 0.7, 0.5, 0.3, 0.15]
v_vals = [0.0, 800.0, 1500.0, 2500.0, 3000.0, 3500.0]
dA_vals = [0.001, 0.0015, 0.002, 0.0012, 0.001, 0.0005]

# pre-upload all tile parameters (mimics production layout)
all_μ_tiles = CuArray(Float64.(μ_vals))
all_dA_tiles = CuArray(Float64.(dA_vals))
all_μ_v = CuArray(repeat(Float64.(v_vals), inner=Natm))

σ_v = Float64(ξ)

@testset "tile_offset correctness" begin

    @testset "convolve_wavelength_axis_batched! with offset" begin
        B = 4
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        # reference: tiles 3-4 with offset=0 using a sliced μ_v array
        bcmem.signal_cached = false
        μ_v_slice = CuArray(repeat(Float64.(v_vals[3:4]), inner=Natm))
        ref = Array(FT.convolve_wavelength_axis_batched!(bcmem, λs_korg, αs,
            μ_v_slice, σ_v, 2))

        # test: tiles 3-4 via tile_offset=2 into the full array
        bcmem.signal_cached = false
        test = Array(FT.convolve_wavelength_axis_batched!(bcmem, λs_korg, αs,
            all_μ_v, σ_v, 2; tile_offset=2))

        @test test ≈ ref atol=1e-12
    end

    @testset "calc_tau_anchored_batched! with offset" begin
        B = 4
        bcmem = FT.BatchedMicroConvMem(Nλ, Natm, B, Npad)

        # convolve tiles 3-4 to get realistic αs input
        bcmem.signal_cached = false
        αs_conv = FT.convolve_wavelength_axis_batched!(bcmem, λs_korg, αs,
            all_μ_v, σ_v, 2; tile_offset=2)

        # reference: τ with offset=0 using sliced μ_tiles
        μ_slice = CuArray(Float64.(μ_vals[3:4]))
        τs_ref = CUDA.zeros(Float64, 2 * Natm, Nλ)
        FT.calc_tau_anchored_batched!(μ_slice, log_τ_ref, ifactor_base,
            αs_conv, τs_ref, Natm, 2)

        # test: τ with tile_offset=2 into full μ_tiles
        τs_test = CUDA.zeros(Float64, 2 * Natm, Nλ)
        FT.calc_tau_anchored_batched!(all_μ_tiles, log_τ_ref, ifactor_base,
            αs_conv, τs_test, Natm, 2; tile_offset=2)

        @test Array(τs_test) ≈ Array(τs_ref) atol=1e-14
    end

    @testset "accumulate_batch! with offset" begin
        B = 4
        cfdt_data = CUDA.rand(Float64, B * Natm1, Nλ) .* 1e-10

        # extract tiles 3-4 as contiguous CuArray (accumulate_batch! requires CuArray, not view)
        cfdt_tiles34 = CuArray(Array(cfdt_data)[(2*Natm1+1):(4*Natm1), :])

        # reference: accumulate tiles 3-4 with offset=0 using sliced dA
        dA_slice = CuArray(Float64.(dA_vals[3:4]))
        flux_ref = CUDA.zeros(Float64, Nλ)
        cfunc_ref = CUDA.zeros(Float64, Natm1, Nλ)
        flux_comp_ref = CUDA.zeros(Float64, Nλ)
        cfunc_comp_ref = CUDA.zeros(Float64, Natm1, Nλ)
        FT.accumulate_batch!(flux_ref, cfunc_ref, flux_comp_ref, cfunc_comp_ref,
            cfdt_tiles34, dA_slice, Natm1, 2)

        # test: same cfdt tiles, but dA read from full array via tile_offset=2
        flux_test = CUDA.zeros(Float64, Nλ)
        cfunc_test = CUDA.zeros(Float64, Natm1, Nλ)
        flux_comp_test = CUDA.zeros(Float64, Nλ)
        cfunc_comp_test = CUDA.zeros(Float64, Natm1, Nλ)
        FT.accumulate_batch!(flux_test, cfunc_test, flux_comp_test, cfunc_comp_test,
            cfdt_tiles34, all_dA_tiles, Natm1, 2; tile_offset=2)

        @test Array(flux_test) ≈ Array(flux_ref) atol=1e-14
        @test Array(cfunc_test) ≈ Array(cfunc_ref) atol=1e-14
    end

    @testset "batched_macro_multiply_accumulate! with offset" begin
        # tile_offset shifts μ_idx and dA indexing only; signal_ft is always
        # batch-local (rows 1:Bcur*Natm1). Each batch pads+FFTs its own cfdt
        # slice, matching production.
        ζ_rt = 3500.0
        cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm1, Npad)
        L_mac = cmem_mac.L
        pad_left_mac = cmem_mac.pad_left
        nfreq_mac = fld(L_mac, 2) + 1

        macro_kernel_cache = Dict{Float64, CuVector{Complex{Float64}}}()
        for μ_val in μ_vals
            macro_kernel_cache[μ_val] = FT.precompute_rt_macro_kernel_ft(cmem_mac, λs_korg, ζ_rt, μ_val)
        end
        unique_μ_sorted = sort(collect(keys(macro_kernel_cache)))
        μ_to_idx = Dict(μ => Int32(i) for (i, μ) in enumerate(unique_μ_sorted))
        kernel_cache_flat = CUDA.zeros(Complex{Float64}, length(unique_μ_sorted), nfreq_mac)
        for (i, μ) in enumerate(unique_μ_sorted)
            copyto!(view(kernel_cache_flat, i, :), macro_kernel_cache[μ])
        end
        μ_idx_all = CuArray(Int32[μ_to_idx[μ_vals[i]] for i in 1:Ntiles])

        cfdt_6_h = rand(Float64, Ntiles * Natm1, Nλ) .* 1e-10
        ts_pad = (32, 32)

        # reference: all 6 tiles padded+FFT'd together, accumulated at once
        cfdt_6 = CuArray(cfdt_6_h)
        mac_pad_all = CUDA.zeros(Float64, Ntiles * Natm1, L_mac)
        bs_all = (cld(Ntiles * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
        @cuda threads=ts_pad blocks=bs_all FT.pad_signal!(mac_pad_all, cfdt_6,
                                                            Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_all = CUDA.CUFFT.plan_rfft(mac_pad_all, 2)
        mac_ft_all = CUDA.zeros(Complex{Float64}, Ntiles * Natm1, nfreq_mac)
        mul!(mac_ft_all, plan_all, mac_pad_all)
        acc_ref = CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac)
        FT.batched_macro_multiply_accumulate!(acc_ref, mac_ft_all, kernel_cache_flat,
            μ_idx_all, all_dA_tiles, Natm1, Ntiles)

        # test: 3+3 tiles, each batch pads+FFTs its own slice separately
        half = 3
        acc_split = CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac)

        cfdt_b1 = CuArray(cfdt_6_h[1:half*Natm1, :])
        mac_pad_b1 = CUDA.zeros(Float64, half * Natm1, L_mac)
        bs_half = (cld(half * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
        @cuda threads=ts_pad blocks=bs_half FT.pad_signal!(mac_pad_b1, cfdt_b1,
                                                             Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_b1 = CUDA.CUFFT.plan_rfft(mac_pad_b1, 2)
        mac_ft_b1 = CUDA.zeros(Complex{Float64}, half * Natm1, nfreq_mac)
        mul!(mac_ft_b1, plan_b1, mac_pad_b1)
        FT.batched_macro_multiply_accumulate!(acc_split, mac_ft_b1, kernel_cache_flat,
            μ_idx_all, all_dA_tiles, Natm1, half; tile_offset=0)

        cfdt_b2 = CuArray(cfdt_6_h[half*Natm1+1:Ntiles*Natm1, :])
        mac_pad_b2 = CUDA.zeros(Float64, half * Natm1, L_mac)
        @cuda threads=ts_pad blocks=bs_half FT.pad_signal!(mac_pad_b2, cfdt_b2,
                                                             Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_b2 = CUDA.CUFFT.plan_rfft(mac_pad_b2, 2)
        mac_ft_b2 = CUDA.zeros(Complex{Float64}, half * Natm1, nfreq_mac)
        mul!(mac_ft_b2, plan_b2, mac_pad_b2)
        FT.batched_macro_multiply_accumulate!(acc_split, mac_ft_b2, kernel_cache_flat,
            μ_idx_all, all_dA_tiles, Natm1, half; tile_offset=half)

        @test Array(acc_split) ≈ Array(acc_ref) rtol=1e-12
    end
end
