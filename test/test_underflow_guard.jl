# Regression test for the zero-sum kernel normalization guard in
# `src/microturbulence.jl`. See `.claude/CLAUDE.md` "Kernel normalization
# underflow guard" for the motivation and physical-correctness analysis.
#
# Case A: per-row underflow (vector v_los with one extreme entry)
# Case B: scalar-v_los whole-column wipe-out
# Case C: bit-equivalence on moderate v_los (the ifelse branch is not taken)
# GPU only: a nonzero row sum whose reciprocal overflows still normalizes to unit sum
#
# On the CPU the guard replaces a zero-sum kernel divisor with 1, so the affected
# row's convolved αs becomes exactly zero instead of NaN. On the GPU a row whose
# kernel leaves the tap span is rendered by direct convolution with replicate
# padding instead: a kernel entirely past the window reads the edge pixel, so
# a redshifted row equals its first opacity.

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
            @test all(out_A_gpu_h[3, :] .== αs[3, 1])

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
            @test all(out_B_gpu_h .== αs[:, 1:1])

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
            @test all(out_D_gpu_h[3, :] .== αs[3, 1])
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

                # ...and the row is still rendered: row 3 reads the window's first pixel and
                # nothing is NaN. Read out_P before the next convolution — it is a view into
                # cmem_P.conv_gpu, which every subsequent build overwrites (CLAUDE.md, "results
                # alias workspace scratch").
                out_P_h = Array(out_P)
                @test all(isfinite, out_P_h)
                @test all(out_P_h[3, :] .== αs[3, 1])

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

            # ── Subnormal row sums: the reciprocal of a nonzero sum can overflow ──────
            # A kernel whose center sits about ten σ past the tap span (Float32) has only
            # subnormal taps, and below 1/floatmax(T) the reciprocal of their sum is Inf.
            # Such a row must still reach unit sum; every row whose reciprocal is finite
            # must keep the bits of x * (1/s).
            @testset "subnormal row sums" begin
                for T in (Float32, Float64)
                    Lr = 300
                    cols = 1:13:248   # 20 taps across the lanes, the 4-run loop and the tail loop
                    rows = zeros(T, 4, Lr)
                    # every value below is a dyadic multiple, so its sum is exact in any order
                    rows[1, cols] .= T.(1:20) .* T(0.125)                  # normal sum
                    rows[2, cols[1:3]] .= T.(1:3) .* (floatmin(T) / 8)     # subnormal sum, finite 1/s
                    rows[3, cols] .= T.(1:20) .* nextfloat(zero(T))        # subnormal sum, 1/s = Inf
                    # row 4 stays zero
                    A = CuArray(rows)
                    FT._normalize_rows!(A, 4)
                    out = Array(A)

                    @test all(isfinite, out)
                    for r in (1, 2)
                        s = sum(rows[r, :])
                        @test isfinite(one(T) / s)
                        @test out[r, :] == rows[r, :] .* (one(T) / s)
                    end
                    s3 = sum(rows[3, :])
                    @test !iszero(s3) && isinf(one(T) / s3)
                    exact = big.(1:20) ./ 210
                    @test all(abs.(big.(out[3, cols]) .- exact) .<= 2 * eps(T) .* exact)
                    @test abs(sum(big.(out[3, :])) - 1) <= 2 * eps(T)
                    @test count(!iszero, out[3, :]) == 20
                    @test all(iszero, out[4, :])
                end

                # A per-row kernel build at Float32 with one row's center 9.8 σx past the tap
                # span. Its edge tap is subnormal and every other tap is smaller by a factor
                # over e^8, so the row sum lies between that tap and twice it. The kernels are
                # read before the FFT: the convolution renders such a row by the direct
                # fallback, which would hide a non-finite kernel in that row.
                T = Float32
                σx = λ0 * 1200.0 / FT.c_ms
                k_past = 9.8
                edge_tap = T(exp(-k_past^2))
                @test !iszero(edge_tap) && isinf(one(T) / (2 * edge_tap))
                v_band = ((Nλ ÷ 2) * Δλ + k_past * σx) / λ0 * FT.c_ms
                v_los_S = fill(T(v_moderate), Natm)
                v_los_S[3] = T(v_band)
                cmem_S = FT.ConvolutionMemory(Nλ, Natm, Npad; T=T)
                FT._init_micro_params!(cmem_S, T.(xs), xs)
                FT._build_per_row_kernels!(cmem_S, cmem_S.xs_cpu, CuArray(v_los_S), T(1200))
                K = Array(cmem_S.conv_gpu)
                @test all(isfinite, K)
                @test abs(sum(big.(K[3, :])) - 1) <= 2 * eps(T)
            end
        end
    end
end
