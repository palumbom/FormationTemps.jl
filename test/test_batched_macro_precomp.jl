# Tests that batched macro kernel precomputation (compute_rt_macro_dft_layout_2d!)
# produces the same kernel FFTs as the serial precompute_rt_macro_kernel_ft path.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using LinearAlgebra
using Test

if !CUDA.functional()
    @info "CUDA not available, skipping batched macro precomp tests"
    exit()
end

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
Δλ = 0.01
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
Nλ = length(λs_korg)
λs_gpu = CuArray(Float64.(collect(λs_korg)))

Natm1 = 55
Npad = 512
cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm1, Npad)
L = cmem_mac.L
nfreq = fld(L, 2) + 1
i0 = Nλ ÷ 2 + 1
ζ_rt = 3500.0

test_μs = [0.95, 0.7, 0.4, 0.2, 0.05]

@testset "Batched macro kernel precomputation" begin

    @testset "Batched kernel FFTs match serial precompute_rt_macro_kernel_ft" begin
        # serial reference
        serial_fts = Dict{Float64, Vector{Complex{Float64}}}()
        for μ in test_μs
            serial_fts[μ] = Array(FT.precompute_rt_macro_kernel_ft(cmem_mac, λs_korg, ζ_rt, μ))
        end

        # batched path
        N_unique = length(test_μs)
        μ_vals_gpu = CuArray(Float64.(test_μs))
        kbuf = CUDA.zeros(Float64, N_unique, L)
        ts = (32, 32)
        bs = (cld(Nλ, ts[1]), cld(N_unique, ts[2]))
        @cuda threads=ts blocks=bs FT.compute_rt_macro_dft_layout_2d!(
            kbuf, λs_gpu, μ_vals_gpu, Int32(i0), ζ_rt, Int32(Nλ), Int32(L))
        kbuf ./= sum(kbuf, dims=2)
        plan = CUDA.CUFFT.plan_rfft(kbuf, 2)
        batched_ft = CUDA.zeros(Complex{Float64}, N_unique, nfreq)
        mul!(batched_ft, plan, kbuf)
        batched_ft_h = Array(batched_ft)

        for (i, μ) in enumerate(test_μs)
            @test batched_ft_h[i, :] ≈ serial_fts[μ] atol=1e-12
        end
    end

    @testset "DFT layout: zero-lag at index L" begin
        # the kernel peak (v=0, j=i0) should map to DFT index L
        μ_vals_gpu = CuArray([0.5])
        kbuf = CUDA.zeros(Float64, 1, L)
        ts = (32, 32)
        bs = (cld(Nλ, ts[1]), 1)
        @cuda threads=ts blocks=bs FT.compute_rt_macro_dft_layout_2d!(
            kbuf, λs_gpu, μ_vals_gpu, Int32(i0), ζ_rt, Int32(Nλ), Int32(L))
        kbuf_h = Array(kbuf)
        @test argmax(kbuf_h[1, :]) == L
    end

    @testset "Float32 precision" begin
        # compare Float32 batched vs Float32 serial (same precision, same path)
        cmem32 = FT.MacroConvolutionMemory(Nλ, Natm1, Npad; T=Float32)
        L32 = cmem32.L
        nfreq32 = fld(L32, 2) + 1
        λs_f32 = Float32.(collect(λs_korg))

        serial_ft_f32 = Array(FT.precompute_rt_macro_kernel_ft(cmem32, λs_f32, Float32(ζ_rt), Float32(0.7)))

        μ_vals_gpu32 = CuArray(Float32[0.7])
        λs_gpu32 = CuArray(λs_f32)
        kbuf32 = CUDA.zeros(Float32, 1, L32)
        ts = (32, 32)
        bs = (cld(Nλ, ts[1]), 1)
        @cuda threads=ts blocks=bs FT.compute_rt_macro_dft_layout_2d!(
            kbuf32, λs_gpu32, μ_vals_gpu32, Int32(i0), Float32(ζ_rt), Int32(Nλ), Int32(L32))
        kbuf32 ./= sum(kbuf32, dims=2)
        plan32 = CUDA.CUFFT.plan_rfft(kbuf32, 2)
        ft32 = CUDA.zeros(Complex{Float32}, 1, nfreq32)
        mul!(ft32, plan32, kbuf32)

        @test Array(ft32)[1, :] ≈ serial_ft_f32 atol=1e-5
    end
end
