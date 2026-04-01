# Tests that the batched Fourier-domain macro accumulation produces
# the same results as the old per-tile convolve + accumulate path.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using LinearAlgebra
using Test
using Statistics

if !CUDA.functional()
    @info "CUDA not available, skipping batched macro tests"
    exit()
end

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

Teff = 5777.0
logg = 4.44
Fe_H = 0.0
ξ = 850.0
ζ_rt = 3500.0
Δλ = 0.01
Npad = 512

A_X = Korg.format_A_X(Fe_H)
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm = length(atm_gpu.zs)
Natm1 = Natm - 1

wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
Nλ = length(λs_korg)
λs_gpu = CuArray{Float64}(collect(λs_korg))

# absorption
αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                  α_ref_out=α_ref, ne_warn_thresh=Inf)

# geometry for a few test tiles with distinct μ values
μ_vals = [0.95, 0.7, 0.4, 0.2]
dA_vals = [0.001, 0.002, 0.0015, 0.0008]
B = length(μ_vals)

# macro kernel setup
cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm1, Npad)
L_mac = cmem_mac.L
pad_left_mac = cmem_mac.pad_left
nfreq_mac = fld(L_mac, 2) + 1

# precompute per-μ kernel FFTs
macro_kernel_cache = Dict{Float64, CuVector{Complex{Float64}}}()
for μ_val in μ_vals
    macro_kernel_cache[μ_val] = FT.precompute_rt_macro_kernel_ft(cmem_mac, λs_korg, ζ_rt, μ_val)
end

# generate synthetic cfdt data (mimics contribution function structure)
cfdt_batch = CUDA.rand(Float64, B * Natm1, Nλ) .* 1e-10

@testset "Batched Fourier macro accumulation" begin

    @testset "Batched path matches per-tile path" begin
        # ── reference: per-tile convolve + accumulate (old path) ──────────
        ref_cfunc = CUDA.zeros(Float64, Natm1, Nλ)
        ref_cfunc_comp = CUDA.zeros(Float64, Natm1, Nλ)
        ref_flux = CUDA.zeros(Float64, Nλ)
        ref_flux_comp = CUDA.zeros(Float64, Nλ)

        for bi in 1:B
            tile_cfdt = @view cfdt_batch[(bi-1)*Natm1+1 : bi*Natm1, :]
            convolved = FT.convolve_rt_macro_gpu_cached(cmem_mac, tile_cfdt,
                                                         macro_kernel_cache[μ_vals[bi]])
            FT.accumulate_tile!(ref_flux, ref_cfunc, ref_flux_comp, ref_cfunc_comp,
                                convolved, Float64(dA_vals[bi]))
        end
        ref_cfunc_h = Array(ref_cfunc)

        # ── test: batched Fourier path ────────────────────────────────────
        # flatten kernel cache
        unique_μ_sorted = sort(collect(keys(macro_kernel_cache)))
        μ_to_idx = Dict(μ => Int32(i) for (i, μ) in enumerate(unique_μ_sorted))
        kernel_cache_flat = CUDA.zeros(Complex{Float64}, length(unique_μ_sorted), nfreq_mac)
        for (i, μ) in enumerate(unique_μ_sorted)
            copyto!(view(kernel_cache_flat, i, :), macro_kernel_cache[μ])
        end
        μ_idx = CuArray(Int32[μ_to_idx[μ_vals[bi]] for bi in 1:B])
        dA_gpu = CuArray(Float64.(dA_vals))

        # pad + batched FFT
        mac_pad = CUDA.zeros(Float64, B * Natm1, L_mac)
        ts_pad = (32, 32)
        bs_pad = (cld(B * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
        @cuda threads=ts_pad blocks=bs_pad FT.pad_signal!(mac_pad, cfdt_batch,
                                                            Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_fwd = CUDA.CUFFT.plan_rfft(mac_pad, 2)
        mac_ft = CUDA.zeros(Complex{Float64}, B * Natm1, nfreq_mac)
        mul!(mac_ft, plan_fwd, mac_pad)

        # Fourier-domain multiply-accumulate
        acc_ft = CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac)
        FT.batched_macro_multiply_accumulate!(acc_ft, mac_ft, kernel_cache_flat,
                                               μ_idx, dA_gpu, Natm1, B)

        # final IFFT + extract
        mac_ifft = CUDA.zeros(Float64, Natm1, L_mac)
        plan_bwd = CUDA.CUFFT.plan_irfft(CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac), L_mac, 2)
        mul!(mac_ifft, plan_bwd, acc_ft)
        mac_out = CUDA.zeros(Float64, Natm1, Nλ)
        ts_ext = (32, 32)
        bs_ext = (cld(Natm1, ts_ext[1]), cld(Nλ, ts_ext[2]))
        @cuda threads=ts_ext blocks=bs_ext FT.extract_valid!(mac_out, mac_ifft, pad_left_mac, Nλ)

        batched_cfunc_h = Array(mac_out)

        @test batched_cfunc_h ≈ ref_cfunc_h rtol=1e-8
    end

    @testset "Partial batch (Bcur < B)" begin
        B_big = 8
        Bcur = 3  # only first 3 tiles

        cfdt_big = CUDA.rand(Float64, B_big * Natm1, Nλ) .* 1e-10
        μ_vals_big = [0.95, 0.7, 0.4, 0.2, 0.8, 0.6, 0.3, 0.15]
        dA_vals_big = [0.001, 0.002, 0.0015, 0.0008, 0.001, 0.002, 0.001, 0.001]

        for μ_val in μ_vals_big
            if !haskey(macro_kernel_cache, μ_val)
                macro_kernel_cache[μ_val] = FT.precompute_rt_macro_kernel_ft(cmem_mac, λs_korg, ζ_rt, μ_val)
            end
        end

        # reference: per-tile on first Bcur tiles only
        ref_cfunc = CUDA.zeros(Float64, Natm1, Nλ)
        ref_comp = CUDA.zeros(Float64, Natm1, Nλ)
        ref_flux = CUDA.zeros(Float64, Nλ)
        ref_flux_comp = CUDA.zeros(Float64, Nλ)
        for bi in 1:Bcur
            tile_cfdt = @view cfdt_big[(bi-1)*Natm1+1 : bi*Natm1, :]
            convolved = FT.convolve_rt_macro_gpu_cached(cmem_mac, tile_cfdt,
                                                         macro_kernel_cache[μ_vals_big[bi]])
            FT.accumulate_tile!(ref_flux, ref_cfunc, ref_flux_comp, ref_comp,
                                convolved, Float64(dA_vals_big[bi]))
        end

        # batched Fourier path with Bcur < B_big
        unique_μ_sorted = sort(collect(keys(macro_kernel_cache)))
        μ_to_idx = Dict(μ => Int32(i) for (i, μ) in enumerate(unique_μ_sorted))
        kernel_cache_flat = CUDA.zeros(Complex{Float64}, length(unique_μ_sorted), nfreq_mac)
        for (i, μ) in enumerate(unique_μ_sorted)
            copyto!(view(kernel_cache_flat, i, :), macro_kernel_cache[μ])
        end
        μ_idx = CuArray(Int32[μ_to_idx[μ_vals_big[bi]] for bi in 1:B_big])
        dA_gpu = CuArray(Float64.(dA_vals_big))

        mac_pad = CUDA.zeros(Float64, B_big * Natm1, L_mac)
        ts_pad = (32, 32)
        bs_pad = (cld(B_big * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
        @cuda threads=ts_pad blocks=bs_pad FT.pad_signal!(mac_pad, cfdt_big,
                                                            Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_fwd = CUDA.CUFFT.plan_rfft(mac_pad, 2)
        mac_ft = CUDA.zeros(Complex{Float64}, B_big * Natm1, nfreq_mac)
        mul!(mac_ft, plan_fwd, mac_pad)

        acc_ft = CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac)
        FT.batched_macro_multiply_accumulate!(acc_ft, mac_ft, kernel_cache_flat,
                                               μ_idx, dA_gpu, Natm1, Bcur)

        mac_ifft = CUDA.zeros(Float64, Natm1, L_mac)
        plan_bwd = CUDA.CUFFT.plan_irfft(CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac), L_mac, 2)
        mul!(mac_ifft, plan_bwd, acc_ft)
        mac_out = CUDA.zeros(Float64, Natm1, Nλ)
        ts_ext = (32, 32)
        bs_ext = (cld(Natm1, ts_ext[1]), cld(Nλ, ts_ext[2]))
        @cuda threads=ts_ext blocks=bs_ext FT.extract_valid!(mac_out, mac_ifft, pad_left_mac, Nλ)

        @test Array(mac_out) ≈ Array(ref_cfunc) rtol=1e-8
    end

    @testset "Accumulates across multiple batches" begin
        # simulate 2 batches of B=2 tiles each, verify same result as one batch of 4
        # key: each batch pads+FFTs its own cfdt slice (matching production), and
        # tile_offset only shifts μ_idx/dA indexing
        B2 = 2
        cfdt_4_h = rand(Float64, 4 * Natm1, Nλ) .* 1e-10
        cfdt_4 = CuArray(cfdt_4_h)
        μ_vals_4 = [0.95, 0.7, 0.4, 0.2]
        dA_vals_4 = Float64[0.001, 0.002, 0.0015, 0.0008]

        unique_μ_sorted = sort(collect(keys(macro_kernel_cache)))
        μ_to_idx = Dict(μ => Int32(i) for (i, μ) in enumerate(unique_μ_sorted))
        kernel_cache_flat = CUDA.zeros(Complex{Float64}, length(unique_μ_sorted), nfreq_mac)
        for (i, μ) in enumerate(unique_μ_sorted)
            copyto!(view(kernel_cache_flat, i, :), macro_kernel_cache[μ])
        end
        μ_idx_4 = CuArray(Int32[μ_to_idx[μ_vals_4[bi]] for bi in 1:4])
        dA_gpu_4 = CuArray(dA_vals_4)

        # single batch of 4
        mac_pad_4 = CUDA.zeros(Float64, 4 * Natm1, L_mac)
        ts_pad = (32, 32)
        bs_pad4 = (cld(4 * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
        @cuda threads=ts_pad blocks=bs_pad4 FT.pad_signal!(mac_pad_4, cfdt_4,
                                                             Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_fwd_4 = CUDA.CUFFT.plan_rfft(mac_pad_4, 2)
        mac_ft_4 = CUDA.zeros(Complex{Float64}, 4 * Natm1, nfreq_mac)
        mul!(mac_ft_4, plan_fwd_4, mac_pad_4)
        acc_one = CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac)
        FT.batched_macro_multiply_accumulate!(acc_one, mac_ft_4, kernel_cache_flat,
                                               μ_idx_4, dA_gpu_4, Natm1, 4)

        # two batches of 2 — each batch pads+FFTs its own cfdt slice
        acc_two = CUDA.zeros(Complex{Float64}, Natm1, nfreq_mac)

        # batch 1: tiles 1-2
        cfdt_b1 = CuArray(cfdt_4_h[1:B2*Natm1, :])
        mac_pad_b1 = CUDA.zeros(Float64, B2 * Natm1, L_mac)
        bs_pad2 = (cld(B2 * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
        @cuda threads=ts_pad blocks=bs_pad2 FT.pad_signal!(mac_pad_b1, cfdt_b1,
                                                             Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_fwd_b1 = CUDA.CUFFT.plan_rfft(mac_pad_b1, 2)
        mac_ft_b1 = CUDA.zeros(Complex{Float64}, B2 * Natm1, nfreq_mac)
        mul!(mac_ft_b1, plan_fwd_b1, mac_pad_b1)
        FT.batched_macro_multiply_accumulate!(acc_two, mac_ft_b1, kernel_cache_flat,
                                               μ_idx_4, dA_gpu_4, Natm1, B2; tile_offset=0)

        # batch 2: tiles 3-4
        cfdt_b2 = CuArray(cfdt_4_h[B2*Natm1+1:4*Natm1, :])
        mac_pad_b2 = CUDA.zeros(Float64, B2 * Natm1, L_mac)
        @cuda threads=ts_pad blocks=bs_pad2 FT.pad_signal!(mac_pad_b2, cfdt_b2,
                                                             Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
        plan_fwd_b2 = CUDA.CUFFT.plan_rfft(mac_pad_b2, 2)
        mac_ft_b2 = CUDA.zeros(Complex{Float64}, B2 * Natm1, nfreq_mac)
        mul!(mac_ft_b2, plan_fwd_b2, mac_pad_b2)
        FT.batched_macro_multiply_accumulate!(acc_two, mac_ft_b2, kernel_cache_flat,
                                               μ_idx_4, dA_gpu_4, Natm1, B2; tile_offset=2)

        @test Array(acc_two) ≈ Array(acc_one) rtol=1e-12
    end
end
