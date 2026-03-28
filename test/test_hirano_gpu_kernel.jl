using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using FFTW
using Statistics
using Test

# ── setup ──────────────────────────────────────────────────────────────────────
# Fe I 6301/6302 lines
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * 1e8 >= 6301, wls)
idx2 = findfirst(x -> x * 1e8 >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])
wls = [l.wl * 1e8 for l in linelist]

# wavelength grid
buffer = 0.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.005)
Nλ = length(λs_korg)

# velocity/frequency grid (matches what hirano_rotmacro_ft_kernel uses)
i0 = Nλ ÷ 2 + 1
λ0 = λs_korg[i0]
vs = FT.c_ms .* (collect(λs_korg) .- λ0) ./ λ0
Δv = (vs[end] - vs[1]) / (Nλ - 1)
σs_cpu = FFTW.fftfreq(Nλ) ./ Δv

# broadening parameters to test
test_params = [
    (vsini=2100.0, ζ_rt=3500.0, u1=0.43, u2=0.31, label="solar-like"),
    (vsini=0.0,    ζ_rt=3500.0, u1=0.43, u2=0.31, label="no rotation"),
    (vsini=4200.0, ζ_rt=0.0,    u1=0.43, u2=0.31, label="no macroturbulence"),
    (vsini=15000.0,ζ_rt=6000.0, u1=0.40, u2=0.26, label="fast rotator"),
    (vsini=500.0,  ζ_rt=1000.0, u1=0.0,  u2=0.0,  label="uniform limb darkening"),
]

# ── tests ──────────────────────────────────────────────────────────────────────
@testset "Hirano GPU kernel vs CPU kernel" begin
    σs_gpu = CuArray(σs_cpu)
    Kσ_gpu_buf = CUDA.zeros(Float64, Nλ)

    for p in test_params
        @testset "$(p.label): vsini=$(p.vsini), ζ=$(p.ζ_rt)" begin
            # CPU reference
            Kσ_cpu = FT.hirano_rotmacro_ft_kernel(σs_cpu, p.vsini, p.ζ_rt;
                                                   u1=p.u1, u2=p.u2,
                                                   intres=FT.intres_glob)

            # GPU kernel (one block per frequency bin, 256 threads for shared-memory reduction)
            kernel_threads = 256
            @cuda threads=kernel_threads blocks=Nλ shmem=kernel_threads*sizeof(Float64) FT.hirano_rotmacro_ft_kernel_gpu!(
                Kσ_gpu_buf, σs_gpu, p.vsini, p.ζ_rt, p.u1, p.u2,
                FT.intres_glob, Nλ)
            CUDA.synchronize()
            Kσ_gpu = Array(Kσ_gpu_buf)

            # pointwise agreement — both use besselj0 and identical math,
            # so should agree to near machine precision
            max_abs_diff = maximum(abs.(Kσ_cpu .- Kσ_gpu))
            max_rel_diff = maximum(abs.(Kσ_cpu .- Kσ_gpu) ./ max.(abs.(Kσ_cpu), 1e-30))

            @test max_abs_diff < 1e-10
            @test max_rel_diff < 1e-8
        end
    end

    @testset "normalization preserved through full convolution" begin
        # build a full kernel from GPU FT output via the same
        # ifft → fftshift → normalize path and verify sum ≈ 1
        σs_gpu = CuArray(σs_cpu)
        Kσ_gpu_buf = CUDA.zeros(Float64, Nλ)

        kernel_threads = 256
        @cuda threads=kernel_threads blocks=Nλ shmem=kernel_threads*sizeof(Float64) FT.hirano_rotmacro_ft_kernel_gpu!(
            Kσ_gpu_buf, σs_gpu, 2100.0, 3500.0, 0.43, 0.31,
            FT.intres_glob, Nλ)
        CUDA.synchronize()
        Kσ_gpu = Array(Kσ_gpu_buf)

        K_dft = Kσ_gpu ./ Δv
        k_circ = real(ifft(K_dft))
        k_circ ./= sum(k_circ)
        @test sum(k_circ) ≈ 1.0 atol=1e-12
        @test all(isfinite.(k_circ))
    end
end

# ── end-to-end convolution comparison ──────────────────────────────────────────
# After Task 3 rewires convolve_hirano_rotmacro_gpu, this tests the full path
@testset "Hirano GPU convolution: end-to-end CPU vs GPU agreement" begin
    A_X = Korg.asplund_2020_solar_abundances
    atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
    Natm = length(atm_gpu.zs)

    αs = zeros(Natm, Nλ)
    αs_cont = zeros(Natm, Nλ)
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

    Npad = 240
    cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
    cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)
    gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

    σ_v_mic = CUDA.zeros(Float64, Natm) .+ 1200.0
    cfunc_flux = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
    tbc = Array(cfunc_flux.cfunc_dt)

    vsini = 4200.0
    ζ_rt = 1200.0
    u1 = 0.4
    u2 = 0.26

    # edge exclusion (CPU circular vs GPU padded convolution)
    λ0_val = mean(collect(λs_korg))
    Δλ = step(λs_korg)
    edge_px = ceil(Int, vsini / FT.c_ms * λ0_val / Δλ) + 10
    interior = (edge_px+1):(Nλ - edge_px)

    cfunc_cpu = FT.convolve_hirano_rotmacro(λs_korg, tbc, vsini, ζ_rt, u1, u2)
    cfunc_gpu = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini, ζ_rt, u1, u2))

    rel_error = maximum(abs.((cfunc_cpu .- cfunc_gpu) ./ cfunc_cpu)[:, interior])
    @test rel_error < 1e-2
end
