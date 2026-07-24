let
# Tests for kernel-support-derived convolution padding.
#
# All broadening is applied as a padded linear convolution with edge replication: the signal
# is padded to an FFT-friendly length L, convolved, and the valid region extracted from
# pad_left+1 : pad_left+Nλ. That equals a true linear convolution only while the kernel's
# half-support fits inside pad_left; beyond it the extraction pulls samples wrapped from the
# opposite edge, with no NaN and no warning.
#
# The rotational kernel's half-support is vsini/Δv pixels, which grows as Δλ shrinks, so a
# vsini that is safe at Δλ=0.01 Å can wrap at Δλ=0.002 Å. conv_npad_for_velocity derives the
# padding from the kernel support, with a floor so padding is only ever added.
using FormationTemps; FT = FormationTemps
using Korg
using Statistics
using Test

# required half-support, in pixels, for a kernel reaching `vmax` m/s
half_support_px(λ0, Δλ, vmax) = ceil(Int, vmax / (FT.c_ms * Δλ / λ0))

@testset "Kernel-support-derived convolution padding" begin

    @testset "conv_kernel_vmax" begin
        # vsini enters at full magnitude (it displaces the kernel centre); ζ and ξ enter at
        # 3σ (they set its width)
        @test FT.conv_kernel_vmax(15000.0, 3400.0, 850.0) ≈ 15000 + 3*3400 + 3*850
        @test FT.conv_kernel_vmax(0.0, 0.0, 0.0) == 0.0
        # sign-insensitive: a negative v_los displaces the kernel just as far
        @test FT.conv_kernel_vmax(-15000.0, 3400.0, 850.0) == FT.conv_kernel_vmax(15000.0, 3400.0, 850.0)
        # monotone in each argument
        @test FT.conv_kernel_vmax(20000.0, 3400.0, 850.0) > FT.conv_kernel_vmax(15000.0, 3400.0, 850.0)
        @test FT.conv_kernel_vmax(15000.0, 5000.0, 850.0) > FT.conv_kernel_vmax(15000.0, 3400.0, 850.0)
        @test FT.conv_kernel_vmax(15000.0, 3400.0, 1200.0) > FT.conv_kernel_vmax(15000.0, 3400.0, 850.0)

        # per-layer ξ uses the widest layer, and matches the scalar form at that value
        ξvec = [800.0, 900.0, 1000.0]
        @test FT.conv_kernel_vmax(1000.0, 0.0, ξvec) == FT.conv_kernel_vmax(1000.0, 0.0, 1000.0)
        @test FT.conv_kernel_vmax(1000.0, 0.0, Float64[]) == FT.conv_kernel_vmax(1000.0, 0.0, 0.0)
    end

    @testset "conv_npad_for_velocity respects the padding floor" begin
        # a narrow kernel gets the floor value, so results stay bit-identical wherever the
        # floor already suffices
        @test FT.conv_npad_for_velocity(6000.0, 0.01, 0.0) == 512
        @test FT.conv_npad_for_velocity(6000.0, 0.01, FT.conv_kernel_vmax(2100.0, 3400.0, 850.0)) == 512
        @test FT.conv_npad_for_velocity(6000.0, 0.01, FT.conv_kernel_vmax(15000.0, 3400.0, 850.0)) == 512
        # and it must grow once the kernel outgrows that floor
        @test FT.conv_npad_for_velocity(6000.0, 0.002, FT.conv_kernel_vmax(50000.0, 3400.0, 850.0)) > 512
    end

    @testset "invariant: pad_left ≥ kernel half-support" begin
        # the property that actually matters, swept over the grid/rotation space
        Nλ = 2048
        for λ0 in (5000.0, 6000.0, 16000.0)
            for Δλ in (0.01, 0.005, 0.002, 0.001)
                for (vsini, ζ, ξ) in ((0.0, 0.0, 0.0), (2100.0, 3400.0, 850.0),
                                      (15000.0, 3400.0, 850.0), (50000.0, 5000.0, 1200.0),
                                      (150000.0, 5000.0, 1200.0))
                    vmax = FT.conv_kernel_vmax(vsini, ζ, ξ)
                    Npad = FT.conv_npad_for_velocity(λ0, Δλ, vmax)
                    _, _, pad_left, _ = FT._conv_mem_geometry(Nλ, Npad)
                    @test pad_left >= half_support_px(λ0, Δλ, vmax)
                end
            end
        end
    end

    @testset "wraparound is real, and the derived padding fixes it" begin
        # Direct demonstration at the convolution layer, where the wrap actually happens.
        # A kernel wider than pad_left must give a wrong "valid" region at Npad=512 and a
        # correct one at the derived padding. Reference: the same convolution with grossly
        # generous padding.
        #
        # The severity depends entirely on how different the signal's two ENDS are, because
        # the padding is edge-replicated: when both edges sit on the same continuum level the
        # wrapped samples nearly equal the replicated ones and the error is ~1e-6 (which is
        # why this bug stayed quiet). When one edge carries a feature it is ~20-30% of the
        # continuum. Both cases are asserted below, because the contrast is the explanation.
        λ0, Δλ, Nλ = 6000.0, 0.002, 2048
        vsini, ζ, ξ = 50000.0, 5000.0, 1200.0
        vmax = FT.conv_kernel_vmax(vsini, ζ, ξ)
        h = half_support_px(λ0, Δλ, vmax)            # ≈ 687 px, vs pad_left(512) = 256
        @test h > FT._conv_mem_geometry(Nλ, 512)[3]  # guard: we are in the broken regime

        λs = λ0 .+ (collect(0:Nλ-1) .- Nλ÷2) .* Δλ
        i0 = Nλ ÷ 2 + 1
        # box kernel of exactly the rotational half-support, normalized, centred at i0
        kernel = zeros(Float64, Nλ)
        kernel[(i0-h):(i0+h)] .= 1.0
        kernel ./= sum(kernel)

        Npad_derived = FT.conv_npad_for_velocity(λ0, Δλ, vmax)
        @test Npad_derived > 512
        conv(sig, Npad) = FT._padded_convolve(sig, kernel; Npad=Npad)
        err(sig, Npad) = maximum(abs.(conv(sig, Npad) .- conv(sig, 8 * Nλ)))

        # realistic severe case: a strong line sitting on one edge of the window, which is
        # routine when synthesizing in chunks (see chunked.jl)
        edge_line = @. 1.0 - 0.7 * exp(-0.5 * ((λs - λs[1]) / 0.15)^2)
        @test err(edge_line, 512) > 0.05             # measured 0.219
        @test err(edge_line, Npad_derived) < 1e-12   # measured 8e-16

        # the masked case: with both edges on the same continuum, the same under-padding is
        # nearly harmless
        flat_edges = @. 1.0 - 0.6 * exp(-0.5 * ((λs - (λ0 - 1.5)) / 0.05)^2) -
                              0.3 * exp(-0.5 * ((λs - (λ0 + 1.2)) / 0.08)^2)
        @test err(flat_edges, 512) < 1e-5            # measured 9.4e-7
        @test err(flat_edges, Npad_derived) < 1e-12
    end

    @testset "narrow kernels are bit-identical at the padding floor" begin
        # regression guard: in the regime the rest of the suite lives in, the
        # derived padding must reproduce the floor exactly, not merely closely
        λ0, Δλ, Nλ = 6000.0, 0.01, 2048
        vmax = FT.conv_kernel_vmax(2100.0, 3400.0, 850.0)
        @test FT.conv_npad_for_velocity(λ0, Δλ, vmax) == 512

        λs = λ0 .+ (collect(0:Nλ-1) .- Nλ÷2) .* Δλ
        signal = @. 1.0 - 0.5 * exp(-0.5 * ((λs - λ0) / 0.05)^2)
        K = FT._ring_doppler_kernel(0.5, 2100.0, 0.0, 0.0, 0.0, λs, 256)
        @test FT._padded_convolve(signal, K; Npad=FT.conv_npad_for_velocity(λ0, Δλ, vmax)) ==
              FT._padded_convolve(signal, K; Npad=512)
    end

    @testset "pipeline reaches the >512 regime and stays physical" begin
        # end-to-end at a grid/rotation combination that the rest of the suite never hits.
        # :quadrature keeps this affordable; the numerical proof of the fix is the
        # convolution-level test above.
        linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16010]
        linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
        Δλ = 0.002
        star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=50000.0,
                            v_macro=5000.0, v_micro=1200.0)

        wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
        λ0 = 0.5 * (first(wls) + last(wls))
        vmax = FT.conv_kernel_vmax(star.vsini, star.ζ, star.ξ)
        # confirm this configuration exceeds the padding floor
        @test FT.conv_npad_for_velocity(λ0, Δλ, vmax) > 512

        r = calc_formation_temp(star, linelist; Δλ=Δλ, use_gpu=false, method=:quadrature,
                                minλ=λ0 - 6.0, maxλ=λ0 + 6.0, ne_warn_thresh=Inf)
        @test all(isfinite, r.form_temps)
        @test all(isfinite, r.flux)
        Ts = FT.get_Ts(r.atmosphere)
        @test all(r.form_temps .>= minimum(Ts))
        @test all(r.form_temps .<= maximum(Ts))
        # a 50 km/s rotator has no deep cores left: normalized flux stays near 1
        @test minimum(r.flux) > 0.5
        @test maximum(r.flux) < 1.05
    end
end

end
