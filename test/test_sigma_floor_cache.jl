# σ_floor is a grid-invariant constant. These tests pin (a) that factoring the expression
# into `_sigma_floor` is bit-identical to the nine inline copies it replaced, and (b) that
# caching it on the convolution memory is selected by object identity, so callers passing
# their own grid keep the recomputed value.

# `using` at top level, not inside the `let`: macros resolve at lowering time, so a `using
# Test` inside the block leaves `@testset` undefined when the file is run on its own. The
# other test files inherit theirs from runtests.jl and are include-only; this one works
# either way.
using FormationTemps
using Test
using Statistics

# guarded at top level, as runtests.jl does: the struct constructors allocate CuArrays
if FormationTemps.GPU_DEFAULT
    using CUDA
end

let
    FT = FormationTemps

    inline_sigma_floor(xs::AbstractVector{T}) where {T} =
        T(max(eps(T) * mean(xs), T(0.25) * median(diff(xs))))

    @testset "σ_floor helper and cache" begin
        @testset "_sigma_floor reproduces the inline expression bit-for-bit" begin
            for T in (Float32, Float64)
                # production-like grid, a coarse grid, and a deliberately non-uniform one
                grids = [
                    collect(T, range(T(6172.0) - T(0.0005) * 660, step=T(0.0005), length=1321)),
                    collect(T, range(T(5000.0), T(5010.0), length=501)),
                    T.(cumsum([6000.0; fill(0.01, 200); fill(0.03, 100)])),
                ]
                for xs in grids
                    @test FT._sigma_floor(xs) === inline_sigma_floor(xs)
                    # the two-argument form is what the two Npad-carrying CPU sites call
                    @test FT._sigma_floor(xs, median(diff(xs))) === inline_sigma_floor(xs)
                end
            end
        end

        @testset "σ_floor uses median(diff), not the first step" begin
            # Δλ at microturbulence.jl:16 and :38 feeds both _sigma_floor and
            # conv_npad_for_velocity. "median(diff(xs)) is expensive, the grid is uniform,
            # use xs[2]-xs[1]" is the obvious follow-on edit and it moves the kernel width
            # and the derived padding together, silently. Coarse steps first, so the first
            # step and the median genuinely differ on this grid.
            xs_nu = cumsum([6000.0; fill(0.01, 100); fill(0.002, 300)])
            @test median(diff(xs_nu)) != xs_nu[2] - xs_nu[1]          # guard on the fixture
            @test FT._sigma_floor(xs_nu) === FT._sigma_floor(xs_nu, median(diff(xs_nu)))
            @test FT._sigma_floor(xs_nu) !== FT._sigma_floor(xs_nu, xs_nu[2] - xs_nu[1])
        end

        if FT.GPU_DEFAULT
            @testset "_init_micro_params! caches σ_floor on the struct" begin
                Nλ, Natm, Npad = 1321, 12, 512
                xs = collect(Float64, range(6172.0 - 0.0005 * 660, step=0.0005, length=Nλ))
                for cmem in (FT.ConvolutionMemory(Nλ, Natm, Npad),
                             FT.MacroConvolutionMemory(Nλ, Natm, Npad),
                             FT.BatchedMicroConvMem(Nλ, Natm, 2, Npad))
                    FT._init_micro_params!(cmem, xs)
                    @test cmem.σ_floor === inline_sigma_floor(xs)
                    @test cmem.doppler_ready
                end
            end

            @testset "_sigma_floor_cached selects on identity, not equality" begin
                Nλ, Natm, Npad = 501, 8, 512
                xs = collect(Float64, range(5000.0, 5010.0, length=Nλ))
                cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
                FT._init_micro_params!(cmem, xs)

                # the grid the cmem holds → cached value, same object
                @test FT._sigma_floor_cached(cmem, cmem.xs_cpu) === cmem.σ_floor

                # an equal but distinct array → recomputed, and equal
                xs_copy = copy(cmem.xs_cpu)
                @test xs_copy !== cmem.xs_cpu
                @test xs_copy == cmem.xs_cpu
                @test FT._sigma_floor_cached(cmem, xs_copy) === inline_sigma_floor(xs_copy)

                # a genuinely different grid → the caller's grid wins, cache is not consulted
                xs_other = collect(Float64, range(5000.0, 5100.0, length=Nλ))
                @test FT._sigma_floor_cached(cmem, xs_other) === inline_sigma_floor(xs_other)
                @test FT._sigma_floor_cached(cmem, xs_other) != cmem.σ_floor
            end

            # A line profile, not a ramp: convolving a function that is linear in wavelength
            # with any symmetric unit-sum kernel returns it unchanged in the interior, so a
            # ramp would make every kernel-width test below sensitive only to edge effects.
            line_alphas(Natm, Nλ) = [1.0 + 0.01 * i + 0.5 * exp(-((j - Nλ ÷ 2) / 5.0)^2)
                                     for i in 1:Natm, j in 1:Nλ]

            @testset "cached σ_floor leaves the GPU convolution bit-identical" begin
                Nλ, Natm, Npad = 501, 8, 512
                xs = collect(Float64, range(5000.0, 5010.0, length=Nλ))
                αs = line_alphas(Natm, Nλ)
                v_los = CuArray(collect(range(-800.0, 800.0, length=Natm)))
                v_mic = CuArray(collect(range(800.0, 1600.0, length=Natm)))

                # device-xs overload (cache hit) vs host-xs overload (recompute) on the same
                # grid must agree exactly — the two paths differ only in where σ_floor came from
                cmem_d = FT.ConvolutionMemory(Nλ, Natm, Npad)
                out_d = Array(FT.convolve_wavelength_axis_gpu(cmem_d, CuArray(xs), CuArray(αs),
                                                              v_los, v_mic))
                cmem_h = FT.ConvolutionMemory(Nλ, Natm, Npad)
                out_h = Array(FT.convolve_wavelength_axis_gpu(cmem_h, xs, αs, v_los, v_mic))
                @test out_d == out_h
            end

            @testset "host-xs overload sources σ_floor from the caller's grid" begin
                # The three single-tile sites must go through _sigma_floor_cached. Reading
                # cmem.σ_floor directly — "for symmetry with the two batched sites" — changes
                # the kernel width whenever the caller's grid differs from the cmem's, and the
                # direct _sigma_floor_cached asserts above keep passing after the call sites
                # stop using it. v_mic = 0 pins σ to σ_floor, the only regime it is visible in.
                Nλ, Natm, Npad = 501, 4, 512
                xs_fine   = collect(Float64, range(5000.0, 5010.0, length=Nλ))   # Δλ = 0.02
                xs_coarse = collect(Float64, range(5000.0, 5100.0, length=Nλ))   # Δλ = 0.2
                αs_h  = line_alphas(Natm, Nλ)
                v_los = CUDA.zeros(Float64, Natm)
                v_mic = CUDA.zeros(Float64, Natm)

                # both cmems hold the FINE grid on the device, so only σ_floor can differ
                cmem_a = FT.ConvolutionMemory(Nλ, Natm, Npad)
                FT._init_micro_params!(cmem_a, xs_fine)
                out_a = Array(FT.convolve_wavelength_axis_gpu(cmem_a, xs_coarse, αs_h, v_los, v_mic))

                cmem_b = FT.ConvolutionMemory(Nλ, Natm, Npad)
                FT._init_micro_params!(cmem_b, xs_fine)
                out_b = Array(FT.convolve_wavelength_axis_gpu(cmem_b, xs_fine, αs_h, v_los, v_mic))

                @test all(isfinite, out_a)
                @test all(isfinite, out_b)
                @test out_a != out_b
            end

            @testset "σ_floor governs the kernel at v_mic = 0: CPU == GPU" begin
                # The consumption side of the cache. Population is pinned above by comparing
                # cmem.σ_floor to the inline expression; nothing there proves the kernel build
                # actually reads it. A zero floor here does not merely shift the result — it
                # divides by zero in the Gaussian — so this fails loudly, not subtly.
                Nλ, Natm, Npad = 501, 4, 512
                xs_z = collect(Float64, range(5000.0, 5010.0, length=Nλ))
                αs_z = line_alphas(Natm, Nλ)
                ref = FT.convolve_wavelength_axis(xs_z, αs_z, zeros(Natm), zeros(Natm))
                cmem_z = FT.ConvolutionMemory(Nλ, Natm, Npad)
                got = Array(FT.convolve_wavelength_axis_gpu(cmem_z, CuArray(xs_z), CuArray(αs_z),
                                                            CUDA.zeros(Float64, Natm),
                                                            CUDA.zeros(Float64, Natm)))
                @test all(isfinite, got)
                @test maximum(abs.(got .- ref)) < 1e-12
            end

            @testset "σ_floor tracks xs_cpu across re-initialisation" begin
                # σ_floor and xs_cpu are co-dependent, and _init_micro_params! is their only
                # writer. Anything that assigns cmem.xs_cpu on its own path — a re-gridding
                # fast path, an "avoid reallocating when the grid is unchanged" tweak — leaves
                # a stale floor that _sigma_floor_cached will happily return, because
                # xs_h === cmem.xs_cpu still holds.
                Nλ, Natm, Npad = 501, 4, 512
                xs_1 = collect(Float64, range(5000.0, 5010.0, length=Nλ))
                xs_2 = collect(Float64, range(5000.0, 5100.0, length=Nλ))
                cmem_r = FT.ConvolutionMemory(Nλ, Natm, Npad)
                FT._init_micro_params!(cmem_r, xs_1)
                @test cmem_r.σ_floor == FT._sigma_floor(cmem_r.xs_cpu)
                FT._init_micro_params!(cmem_r, xs_2)
                @test cmem_r.xs_cpu == xs_2
                @test cmem_r.σ_floor == FT._sigma_floor(cmem_r.xs_cpu)
                @test cmem_r.σ_floor != FT._sigma_floor(xs_1)
            end
        end
    end
end
