# Regression test for the zero-sum kernel normalization guard in
# `src/microturbulence.jl`. See `.claude/CLAUDE.md` "Kernel normalization
# underflow guard" for the motivation and physical-correctness analysis.
#
# Case A: per-row underflow (vector v_los with one extreme entry)
# Case B: scalar-v_los whole-column wipe-out
# Case C: bit-equivalence on moderate v_los (the ifelse branch is not taken)
#
# The guard replaces a zero-sum kernel divisor with 1, so the affected row's
# convolved αs becomes exactly zero (the discrete-convolution limit "shifted
# out of window contributes nothing") instead of NaN.

let
    using FormationTemps; FT = FormationTemps
    using Test
    using Statistics

    # ── grid setup ────────────────────────────────────────────────────────────
    # Half-window = 0.10 Å, σ_floor = 0.25·Δλ = 2.5e-3 Å.
    # Underflow threshold for the kernel: |shift| ≳ half_window + ~27·σ_floor
    # ≈ 0.17 Å → v_los ≳ c·0.17/6000 ≈ 8.5 km/s.
    Nλ = 21
    Δλ = 0.01
    λ0 = 6000.0
    xs = collect(range(λ0 - (Nλ ÷ 2) * Δλ, λ0 + (Nλ ÷ 2) * Δλ, length=Nλ))
    Natm = 5
    αs = [1.0 + 0.01 * (i + 0.1 * j) for i in 1:Natm, j in 1:Nλ]

    v_extreme  = 100.0e3   # m/s; shift ≈ 5 Å, far past 0.10 Å half-window
    v_moderate = 1.0e3     # m/s; shift ≈ 0.02 Å, comfortably inside

    @testset "Zero-sum kernel normalization guard" begin

        # ── Case A: per-row CPU vector — covers site #2 ──────────────────────
        v_los_A = [10.0, 50.0, v_extreme, 30.0, 20.0]
        v_mic_A = fill(1200.0, Natm)

        out_A_cpu = @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
            FT.convolve_wavelength_axis(xs, αs, v_los_A, v_mic_A)
        end
        @test all(isfinite, out_A_cpu)
        @test all(out_A_cpu[3, :] .== 0.0)
        @test count(iszero, out_A_cpu) == Nλ  # only row 3 is fully zero
        # Surviving rows should be non-degenerate (the smooth input convolved
        # by a normalized Gaussian stays nonzero).
        for r in (1, 2, 4, 5)
            @test all(out_A_cpu[r, :] .> 0)
        end

        # ── Case B: scalar v_los — covers site #1 (CPU) and site #5 (GPU 1D) ─
        out_B_cpu = @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
            FT.convolve_wavelength_axis(xs, αs, v_extreme, 1200.0)
        end
        @test all(isfinite, out_B_cpu)
        @test all(out_B_cpu .== 0.0)

        # ── Case C: moderate v_los, no warning, output is sane ───────────────
        # The guard's `ifelse(iszero(s), 1, s)` branch must NOT fire when
        # s > 0, i.e. behavior is bit-identical to the unguarded form.
        # Use a uniform αs so the convolution is exactly volume-preserving
        # (kernel sums to 1 by construction) and any deviation from `ones`
        # would indicate the guard altered the non-degenerate path.
        αs_uniform = ones(Natm, Nλ)
        out_C_cpu = FT.convolve_wavelength_axis(xs, αs_uniform, v_moderate, 1200.0)
        @test all(isfinite, out_C_cpu)
        @test !any(iszero, out_C_cpu)
        @test maximum(abs.(out_C_cpu .- 1.0)) < 1e-10
        # Also run on the non-uniform αs to ensure no NaN/zero appears for
        # in-window v_los regardless of input profile.
        out_C_nonuniform = FT.convolve_wavelength_axis(xs, αs, v_moderate, 1200.0)
        @test all(isfinite, out_C_nonuniform)
        @test !any(iszero, out_C_nonuniform)

        # ── GPU mirror (only if CUDA is available) ───────────────────────────
        if FT.GPU_DEFAULT
            using CUDA
            Npad = 32
            cmem  = FT.ConvolutionMemory(Nλ, Natm, Npad)
            xs_d  = CuArray(xs)
            αs_d  = CuArray(αs)

            # Case A on GPU: vector v_los, scalar v_mic — covers site #6
            v_los_A_d = CuArray(v_los_A)
            out_A_gpu = @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
                FT.convolve_wavelength_axis_gpu(cmem, xs_d, αs_d, v_los_A_d, 1200.0)
            end
            out_A_gpu_h = Array(out_A_gpu)
            @test all(isfinite, out_A_gpu_h)
            @test all(out_A_gpu_h[3, :] .== 0.0)

            # CPU/GPU agreement on the surviving rows
            for r in (1, 2, 4, 5)
                rel = maximum(abs.((out_A_gpu_h[r, :] .- out_A_cpu[r, :]) ./ out_A_cpu[r, :]))
                @test rel < 1e-8
            end

            # Case B on GPU: scalar v_los — covers site #5 (1D scalar path)
            cmem_B = FT.ConvolutionMemory(Nλ, Natm, Npad)  # fresh cmem to reset doppler_ready
            xs_d_B = CuArray(xs)
            αs_d_B = CuArray(αs)
            out_B_gpu = @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
                FT.convolve_wavelength_axis_gpu(cmem_B, xs_d_B, αs_d_B, v_extreme, 1200.0)
            end
            out_B_gpu_h = Array(out_B_gpu)
            @test all(isfinite, out_B_gpu_h)
            @test all(out_B_gpu_h .== 0.0)

            # Case C on GPU: moderate v_los on uniform αs, should equal 1
            cmem_C = FT.ConvolutionMemory(Nλ, Natm, Npad)
            xs_d_C = CuArray(xs)
            αs_d_C = CuArray(αs_uniform)
            out_C_gpu = FT.convolve_wavelength_axis_gpu(cmem_C, xs_d_C, αs_d_C, v_moderate, 1200.0)
            out_C_gpu_h = Array(out_C_gpu)
            @test all(isfinite, out_C_gpu_h)
            @test !any(iszero, out_C_gpu_h)
            @test maximum(abs.(out_C_gpu_h .- 1.0)) < 1e-10
            @test maximum(abs.(out_C_gpu_h .- out_C_cpu)) < 1e-12

            # ── Case D on GPU: vector v_los AND vector v_mic — covers site #7 ──
            cmem_D = FT.ConvolutionMemory(Nλ, Natm, Npad)
            xs_d_D = CuArray(xs)
            αs_d_D = CuArray(αs)
            v_mic_D_d = CuArray(v_mic_A)  # all 1200.0 m/s, matches scalar case
            out_D_gpu = @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
                FT.convolve_wavelength_axis_gpu(cmem_D, xs_d_D, αs_d_D, v_los_A_d, v_mic_D_d)
            end
            out_D_gpu_h = Array(out_D_gpu)
            @test all(isfinite, out_D_gpu_h)
            @test all(out_D_gpu_h[3, :] .== 0.0)
            # uniform v_mic vector should match the scalar v_mic path bit-tight
            @test maximum(abs.(out_D_gpu_h .- out_A_gpu_h)) < 1e-12

            # ── Pacing: the guard is unconditional, the diagnostic is sampled ──────
            # Base.CoreLogging, not the Logging stdlib: identical values, no test-target
            # dependency (same reasoning as test_ring_kernel.jl).
            @testset "underflow readback pacing" begin
                # the policy itself, independent of any GPU state
                @test FT._should_check_underflow(1)
                @test FT._should_check_underflow(FT._UNDERFLOW_EAGER_BUILDS)
                @test !FT._should_check_underflow(FT._UNDERFLOW_EAGER_BUILDS + 1)
                @test !FT._should_check_underflow(FT._UNDERFLOW_CHECK_STRIDE - 1)
                @test FT._should_check_underflow(FT._UNDERFLOW_CHECK_STRIDE)
                @test !FT._should_check_underflow(FT._UNDERFLOW_CHECK_STRIDE + 1)
                @test FT._should_check_underflow(2 * FT._UNDERFLOW_CHECK_STRIDE)

                cmem_P = FT.ConvolutionMemory(Nλ, Natm, Npad)
                xs_d_P = CuArray(xs)
                αs_d_P = CuArray(αs)
                v_mic_P_d = CuArray(v_mic_A)
                @test cmem_P.n_kernel_builds == 0

                # build 1 is inside the eager window: same warning as before
                @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
                    FT.convolve_wavelength_axis_gpu(cmem_P, xs_d_P, αs_d_P, v_los_A_d, v_mic_P_d)
                end
                @test cmem_P.n_kernel_builds == 1

                # walk to the end of the eager window; these builds still check, and their
                # warnings would reach the ambient logger, so silence them
                Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                    for _ in 2:FT._UNDERFLOW_EAGER_BUILDS
                        FT.convolve_wavelength_axis_gpu(cmem_P, xs_d_P, αs_d_P, v_los_A_d, v_mic_P_d)
                    end
                end
                @test cmem_P.n_kernel_builds == FT._UNDERFLOW_EAGER_BUILDS

                # the next build is past it: identical underflow, no log at all
                out_P = @test_logs min_level=Base.CoreLogging.Warn match_mode=:all begin
                    FT.convolve_wavelength_axis_gpu(cmem_P, xs_d_P, αs_d_P, v_los_A_d, v_mic_P_d)
                end
                @test cmem_P.n_kernel_builds == FT._UNDERFLOW_EAGER_BUILDS + 1

                # ...and the guard still fired: row 3 is zero, nothing is NaN. Read out_P
                # before the next convolution — it is a view into cmem_P.conv_gpu, which every
                # subsequent build overwrites (CLAUDE.md, "results alias workspace scratch").
                out_P_h = Array(out_P)
                @test all(isfinite, out_P_h)
                @test all(out_P_h[3, :] .== 0.0)

                # The stride half of the policy, without doing thousands of builds: seed the
                # counter so the next build lands exactly on a checkpoint. The
                # _should_check_underflow asserts above test the predicate in isolation; this
                # tests that the call site consults it. Inlining the eager term at the site
                # leaves the predicate correct, unused, and the diagnostic dark forever after.
                cmem_P.n_kernel_builds = FT._UNDERFLOW_CHECK_STRIDE - 1
                @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
                    FT.convolve_wavelength_axis_gpu(cmem_P, xs_d_P, αs_d_P, v_los_A_d, v_mic_P_d)
                end
                @test cmem_P.n_kernel_builds == FT._UNDERFLOW_CHECK_STRIDE

                # The batched path is deliberately NOT paced — per-tile v_los varies, so a
                # late batch can be the first to underflow. Pin that, or a future "unify the
                # two paths" edit silently drops the diagnostic for disk integration.
                bcmem_P = FT.BatchedMicroConvMem(Nλ, Natm, 1, Npad)
                @test !hasproperty(bcmem_P, :n_kernel_builds)
                v_los_b = CuArray(v_los_A)
                Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                    for _ in 1:FT._UNDERFLOW_EAGER_BUILDS
                        FT.convolve_wavelength_axis_batched!(bcmem_P, xs, αs, v_los_b, 1200.0, 1)
                    end
                end
                # well past where the single-tile policy would have gone quiet
                @test_logs (:warn, r"Doppler kernel underflowed") match_mode=:any begin
                    FT.convolve_wavelength_axis_batched!(bcmem_P, xs, αs, v_los_b, 1200.0, 1)
                end
            end
        end
    end
end
