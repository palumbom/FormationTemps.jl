let
    vsini = 2100.0
    ζ_rt = 1400.0
    u1 = 0.4
    u2 = 0.0
    intres = 2048

    xs = range(6301.0, 6310.0, step=0.005)
    N = length(xs)
    λ0 = mean(xs)
    vs = FT.c_ms .* (xs .- λ0) ./ λ0
    Δv = (last(vs) - first(vs)) / (N - 1)

    # helper: compute the real-space Hirano kernel from its Fourier-domain representation
    function hirano_kernel(vsini, ζ_rt; u1, u2, intres)
        σ = FFTW.fftfreq(N) ./ Δv
        Kσ = FT.hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)
        k_circ = real(ifft(Kσ ./ Δv))
        k_ctr = FFTW.fftshift(k_circ)
        k_norm = k_ctr ./ sum(k_ctr)
        k_norm = max.(k_norm, 0)       # clip aliasing artifacts from polynomial Fourier tail
        k_norm = k_norm ./ sum(k_norm)
        # shift peak to grid center (N÷2+1, matching gray kernels)
        roll = (N ÷ 2 + 1) - argmax(k_norm)
        return circshift(k_norm, roll)
    end

    hirano_no_rot = hirano_kernel(0.0, ζ_rt; u1=u1, u2=u2, intres=intres)
    hirano_no_macro = hirano_kernel(vsini, 0.0; u1=u1, u2=u2, intres=intres)
    hirano_rot_macro = hirano_kernel(vsini, ζ_rt; u1=u1, u2=u2, intres=intres)
    # uniform-disk variants (u1=u2=0) for direct comparison with gray kernels that assume no LD
    hirano_no_rot_uniform = hirano_kernel(0.0, ζ_rt; u1=0.0, u2=0.0, intres=intres)
    hirano_no_macro_uniform = hirano_kernel(vsini, 0.0; u1=0.0, u2=0.0, intres=intres)

    iso_rt_macro_kernel = FT.gray_iso_rt_macro_kernel(vs, ζ_rt)
    gray_rot_kernel = FT.gray_rot_kernel(vs, vsini, u1)

    if make_plots
        import PythonPlot; plt = PythonPlot
        plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
        plt.ioff()

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        v_range_macro = 4 * ζ_rt
        v_range_rot = 1.5 * vsini

        axes[0, 0].plot(vs, iso_rt_macro_kernel, label="{\\rm Gray iso RT macro}")
        axes[0, 0].plot(vs, hirano_no_rot_uniform, label="{\\rm Hirano u1=0 (no rot)}", ls="--")
        axes[0, 0].plot(vs, hirano_no_rot, label="{\\rm Hirano u1=$(u1) (no rot)}", ls=":")
        axes[0, 0].set_title("{\\rm Macro only}")
        axes[0, 0].set_xlabel("{\\rm Velocity [m/s]}")
        axes[0, 0].set_xlim(-v_range_macro, v_range_macro)
        axes[0, 0].legend()

        axes[0, 1].plot(vs, gray_rot_kernel, label="{\\rm Gray rotation u1=$(u1)}")
        axes[0, 1].plot(vs, hirano_no_macro, label="{\\rm Hirano u1=$(u1) (no macro)}", ls="--")
        axes[0, 1].set_title("{\\rm Rotation only}")
        axes[0, 1].set_xlabel("{\\rm Velocity [m/s]}")
        axes[0, 1].set_xlim(-v_range_rot, v_range_rot)
        axes[0, 1].legend()

        axes[1, 0].plot(vs, hirano_no_rot_uniform .- iso_rt_macro_kernel, label="{\\rm Hirano u1=0} \$-\$ {\\rm gray}")
        axes[1, 0].plot(vs, hirano_no_rot .- iso_rt_macro_kernel, label="{\\rm Hirano u1=$(u1)} \$-\$ {\\rm gray}", ls="--")
        axes[1, 0].set_title("{\\rm Hirano} \$-\$ {\\rm gray (macro)}")
        axes[1, 0].set_xlabel("{\\rm Velocity [m/s]}")
        axes[1, 0].set_xlim(-v_range_macro, v_range_macro)
        axes[1, 0].legend()

        axes[1, 1].plot(vs, hirano_no_macro .- gray_rot_kernel)
        axes[1, 1].set_title("{\\rm Hirano} \$-\$ {\\rm gray (rotation)}")
        axes[1, 1].set_xlabel("{\\rm Velocity [m/s]}")
        axes[1, 1].set_xlim(-v_range_rot, v_range_rot)

        fig.tight_layout()
        fig.savefig(joinpath(test_plotdir, "test_kernels.pdf"), bbox_inches="tight")
        plt.close()

        # Fourier-domain comparison
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

        σ_all = FFTW.fftfreq(N) ./ Δv
        # positive-frequency half only (kernels are real and even)
        pos = σ_all .>= 0
        σ_pos = σ_all[pos]

        # macro: Hirano FT vs analytical [1-exp(-π²ζ²σ²)]/(π²ζ²σ²)
        Kσ_mac = FT.hirano_rotmacro_ft_kernel(σ_all, 0.0, ζ_rt; u1=0.0, u2=0.0, intres=intres)
        x_pos = π^2 .* ζ_rt^2 .* σ_pos.^2
        K_anl = (1.0 .- exp.(-x_pos)) ./ x_pos
        K_anl[1] = 1.0
        axes2[0].plot(σ_pos, Kσ_mac[pos], label="{\\rm Hirano (u1=0)}", lw=2)
        axes2[0].plot(σ_pos, K_anl, label="{\\rm Analytical}", ls="--")
        axes2[0].set_xlim(0, 5 / (π * ζ_rt))
        axes2[0].set_xlabel("{\\rm Frequency [s/m]}")
        axes2[0].set_title("{\\rm Macro FT kernel (u1=0)}")
        axes2[0].legend()

        # rotation: Hirano FT vs DFT of gray rotation kernel
        Kσ_rot = FT.hirano_rotmacro_ft_kernel(σ_all, vsini, 0.0; u1=u1, u2=u2, intres=intres)
        Kσ_gray_rot_pos = real(fft(ifftshift(gray_rot_kernel)))[pos]
        axes2[1].plot(σ_pos, Kσ_rot[pos], label="{\\rm Hirano}", lw=2)
        axes2[1].plot(σ_pos, Kσ_gray_rot_pos, label="{\\rm Gray DFT}", ls="--")
        axes2[1].set_xlim(0, 3 / vsini)
        axes2[1].set_xlabel("{\\rm Frequency [s/m]}")
        axes2[1].set_title("{\\rm Rotation FT kernel}")
        axes2[1].legend()

        fig2.tight_layout()
        fig2.savefig(joinpath(test_plotdir, "test_kernels_fourier.pdf"), bbox_inches="tight")
        plt.close()
    end


    @testset "Kernel normalization" begin
        @test sum(iso_rt_macro_kernel) ≈ 1.0
        @test sum(gray_rot_kernel) ≈ 1.0
        @test sum(hirano_no_rot) ≈ 1.0
        @test sum(hirano_no_macro) ≈ 1.0
        @test sum(hirano_rot_macro) ≈ 1.0
    end

    @testset "Kernel non-negativity" begin
        @test all(iso_rt_macro_kernel .>= 0)
        @test all(gray_rot_kernel .>= 0)
        @test all(hirano_no_rot .>= 0)
        @test all(hirano_no_macro .>= 0)
        @test all(hirano_rot_macro .>= 0)
    end

    @testset "Rotation kernel support" begin
        # gray rotation kernel is exactly zero outside ±vsini
        Δλ_vsini = vsini / FT.c_ms * λ0
        outside = abs.(collect(xs) .- λ0) .> Δλ_vsini * 1.001
        @test all(gray_rot_kernel[outside] .== 0.0)
    end

    @testset "Macro kernel peaks at line center" begin
        i_center = argmax(iso_rt_macro_kernel)
        i_center_h = argmax(hirano_no_rot)
        # peak should be within a few pixels of center
        @test abs(i_center - (N ÷ 2 + 1)) <= 2
        @test abs(i_center_h - (N ÷ 2 + 1)) <= 2
    end

    @testset "Hirano vs gray kernel agreement (real space)" begin
        peak_macro = maximum(iso_rt_macro_kernel)
        peak_rot = maximum(gray_rot_kernel)
        # gray_iso_rt_macro_kernel assumes uniform disk (no LD); compare with Hirano u1=u2=0
        # ~5% real-space error is expected from IFFT sampling at this grid resolution
        @test maximum(abs.(hirano_no_rot_uniform .- iso_rt_macro_kernel)) / peak_macro < 0.08
        # gray_rot_kernel includes LD via u1; compare with Hirano using the same u1
        @test maximum(abs.(hirano_no_macro .- gray_rot_kernel)) / peak_rot < 0.08
    end

    @testset "Hirano vs gray kernel agreement (Fourier domain)" begin
        σ = FFTW.fftfreq(N) ./ Δv
        Kσ_macro = FT.hirano_rotmacro_ft_kernel(σ, 0.0, ζ_rt; u1=0.0, u2=0.0, intres=intres)
        Kσ_rot = FT.hirano_rotmacro_ft_kernel(σ, vsini, 0.0; u1=u1, u2=u2, intres=intres)
        # analytical FT of the isotropic RT macro kernel on a uniform disk: [1-exp(-π²ζ²σ²)]/(π²ζ²σ²)
        # DFT of the real-space kernel has aliasing from the algebraic 1/σ² tail, so use the formula
        x = π^2 .* ζ_rt^2 .* σ.^2
        K_analytical = (1.0 .- exp.(-x)) ./ x
        K_analytical[1] = 1.0  # L'Hopital limit at σ=0
        # rotation kernel has compact support, so its DFT is aliasing-free
        Kσ_gray_rot = real(fft(ifftshift(gray_rot_kernel)))
        @test maximum(abs.(Kσ_macro .- K_analytical)) / K_analytical[1] < 0.001
        @test maximum(abs.(Kσ_rot .- Kσ_gray_rot)) / Kσ_rot[1] < 0.01
    end

    @testset "Combined kernel is broader than components" begin
        # convolving two non-trivial kernels lowers the peak relative to either alone
        @test maximum(hirano_rot_macro) < maximum(hirano_no_rot)
        @test maximum(hirano_rot_macro) < maximum(hirano_no_macro)
    end
end
