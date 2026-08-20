# The GPU convolution must stay free of host synchronisation, or CUDA graph capture — and the
# launch collapse it enables — becomes impossible. Nothing else in the suite would notice a
# reintroduced readback, so assert the property itself. Allocation is a separate matter: an
# allocation node is capturable but makes the graph single-use, which this file does not cover.
#
# This is ALSO the only performance-regression gate for the paced underflow readback. Hoisting
# that readback out of its branch — computing it unconditionally and gating only the @warn —
# restores ~23 μs per render (12.5% of an Anemoi ESS iteration) and passes every assertion in
# test_underflow_guard.jl, because the *warning* stays paced. Only the capture below notices.
# GPU-only, so CI cannot protect it: treat a failure here as a perf regression, not as broken
# graph infrastructure, and do not delete this file as speculative.

using FormationTemps
using Test

if FormationTemps.GPU_DEFAULT
    using CUDA

    let
        FT = FormationTemps

        @testset "GPU convolution is graph-capturable" begin
            Nλ, Natm, Npad = 501, 8, 512
            xs = collect(Float64, range(5000.0, 5010.0, length=Nλ))
            αs = [1.0 + 0.01 * i + 0.5 * exp(-((j - Nλ ÷ 2) / 5.0)^2)
                  for i in 1:Natm, j in 1:Nλ]
            xs_d = CuArray(xs)
            αs_d = CuArray(αs)
            v_los = CuArray(fill(500.0, Natm))
            v_mic = CuArray(fill(1200.0, Natm))
            cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

            # warm: kernel compilation and FFT plan setup cannot happen inside a capture
            FT.convolve_wavelength_axis_gpu(cmem, xs_d, αs_d, v_los, v_mic)

            # (a) a build the pacing policy SKIPS. Captures with or without the capture guard,
            # so this rung isolates "nothing else in this path blocks capture" — including
            # row_sums, whose allocation is capturable and only breaks repeat launch.
            cmem.n_kernel_builds = FT._UNDERFLOW_EAGER_BUILDS
            @test !FT._should_check_underflow(cmem.n_kernel_builds + 1)
            graph_skipped = CUDA.capture(throw_error=false) do
                FT.convolve_wavelength_axis_gpu(cmem, xs_d, αs_d, v_los, v_mic)
            end
            @test graph_skipped isa CUDA.CuGraph

            # (b) a build the policy says is DUE. Captures only because the readback is gated
            # on !CUDA.is_capturing(); this is the rung that fails if that term is dropped,
            # ordered after the readback, or if the readback is hoisted out of the branch.
            cmem.n_kernel_builds = 0
            @test FT._should_check_underflow(cmem.n_kernel_builds + 1)
            graph_due = CUDA.capture(throw_error=false) do
                FT.convolve_wavelength_axis_gpu(cmem, xs_d, αs_d, v_los, v_mic)
            end
            @test graph_due isa CUDA.CuGraph
        end
    end
end
