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
        end
    end
end
