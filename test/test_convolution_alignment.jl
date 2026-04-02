let
# Verify that GPU convolution output is correctly aligned with the input
# wavelength grid for both microturbulence and macroturbulence paths.
# A delta-spike convolved with a near-identity kernel must land at the same index.
using FormationTemps; FT = FormationTemps
using CUDA
using FFTW
using Statistics

# ── shared setup ───────────────────────────────────────────────────────────────
Nλ = 200
Natm = 4
Npad = 64
λs = collect(range(6300.0, 6302.0, length=Nλ))
Δλ = λs[2] - λs[1]

spike_positions = [20, Nλ ÷ 2, Nλ - 20]

cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm, Npad)

σ_v_tiny = CUDA.zeros(Float64, Natm) .+ 1.0
μ_v_zero = CUDA.zeros(Float64, Natm)

# ── tests ──────────────────────────────────────────────────────────────────────
@testset "Convolution alignment" begin
    @testset "Microturbulence: spike recovery (near-identity)" begin
        for spike_idx in spike_positions
            ys = zeros(Natm, Nλ)
            ys[:, spike_idx] .= 1.0
            cmem.signal_cached = false
            result = Array(FT.convolve_wavelength_axis_gpu(cmem, λs, ys, μ_v_zero, σ_v_tiny))
            peak_idx = argmax(result[1, :])
            @test peak_idx == spike_idx
        end
    end

    @testset "Microturbulence: Doppler shift direction" begin
        vsini_test = 2000.0
        μ_v_shift = CUDA.zeros(Float64, Natm) .+ vsini_test
        λ0 = λs[Nλ ÷ 2 + 1]
        expected_shift_pix = vsini_test / FT.c_ms * λ0 / Δλ

        spike_idx = Nλ ÷ 2
        ys = zeros(Natm, Nλ)
        ys[:, spike_idx] .= 1.0
        cmem.signal_cached = false
        result = Array(FT.convolve_wavelength_axis_gpu(cmem, λs, ys, μ_v_shift, σ_v_tiny))
        peak_idx = argmax(result[1, :])
        actual_shift = peak_idx - spike_idx
        @test abs(actual_shift - round(Int, expected_shift_pix)) <= 1
    end

    @testset "RT macroturbulence: spike recovery (near-identity)" begin
        ζ_tiny = 10.0
        μ_val = 0.9
        for spike_idx in spike_positions
            ys = zeros(Natm, Nλ)
            ys[:, spike_idx] .= 1.0
            result = Array(FT.convolve_rt_macro_gpu(cmem_mac, λs, ys, ζ_tiny, μ_val))
            peak_idx = argmax(result[1, :])
            @test peak_idx == spike_idx
        end
    end

    @testset "RT macroturbulence: spike recovery with real broadening" begin
        ζ_real = 3000.0
        μ_val = 0.9
        for spike_idx in spike_positions
            ys = zeros(Natm, Nλ)
            ys[:, spike_idx] .= 1.0
            result = Array(FT.convolve_rt_macro_gpu(cmem_mac, λs, ys, ζ_real, μ_val))
            peak_idx = argmax(result[1, :])
            @test peak_idx == spike_idx
        end
    end

    @testset "Iso RT macroturbulence: spike recovery" begin
        ζ_tiny = 10.0
        for spike_idx in spike_positions
            ys = zeros(Natm, Nλ)
            ys[:, spike_idx] .= 1.0
            result = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac, λs, ys, ζ_tiny))
            peak_idx = argmax(result[1, :])
            @test peak_idx == spike_idx
        end
    end

    @testset "Gray rotation: spike recovery" begin
        vsini_small = 100.0
        u1 = 0.4
        for spike_idx in spike_positions
            ys = zeros(Natm, Nλ)
            ys[:, spike_idx] .= 1.0
            result = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs, ys, vsini_small, u1))
            peak_idx = argmax(result[1, :])
            @test peak_idx == spike_idx
        end
    end

    @testset "CPU/GPU zero lag: rt_macro" begin
        ζ_real = 3000.0
        μ_val = 0.9
        ys_full = randn(Natm, Nλ) .* 1e-10 .+ 1e-8
        ys_full[:, Nλ÷2] .+= 1e-6

        result_cpu = FT.convolve_rt_macro(λs, ys_full, ζ_real, μ_val)
        result_gpu = Array(FT.convolve_rt_macro_gpu(cmem_mac, λs, ys_full, ζ_real, μ_val))

        cpu_row = result_cpu[1, :] .- mean(result_cpu[1, :])
        gpu_row = result_gpu[1, :] .- mean(result_gpu[1, :])
        cpu_row ./= maximum(abs.(cpu_row))
        gpu_row ./= maximum(abs.(gpu_row))

        xcorr = real(ifft(fft(cpu_row) .* conj(fft(gpu_row))))
        xcorr_shift = fftshift(xcorr)
        peak_lag = argmax(xcorr_shift) - (Nλ ÷ 2 + 1)
        @test peak_lag == 0
    end

    @testset "CPU/GPU zero lag: microturbulence" begin
        σ_v_real = 1200.0
        ys_full = randn(Natm, Nλ) .* 1e-10 .+ 1e-8
        ys_full[:, Nλ÷2] .+= 1e-6

        μ_v_zero_cpu = zeros(Natm)
        σ_v_cpu = fill(σ_v_real, Natm)
        σ_v_gpu = CUDA.zeros(Float64, Natm) .+ σ_v_real
        μ_v_gpu = CUDA.zeros(Float64, Natm)

        cmem.signal_cached = false
        result_cpu = FT.convolve_wavelength_axis(λs, ys_full, μ_v_zero_cpu, σ_v_cpu)
        result_gpu = Array(FT.convolve_wavelength_axis_gpu(cmem, λs, ys_full, μ_v_gpu, σ_v_gpu))

        cpu_row = result_cpu[1, :] .- mean(result_cpu[1, :])
        gpu_row = result_gpu[1, :] .- mean(result_gpu[1, :])
        cpu_row ./= maximum(abs.(cpu_row))
        gpu_row ./= maximum(abs.(gpu_row))

        xcorr = real(ifft(fft(cpu_row) .* conj(fft(gpu_row))))
        xcorr_shift = fftshift(xcorr)
        peak_lag = argmax(xcorr_shift) - (Nλ ÷ 2 + 1)
        @test peak_lag == 0
    end
end

end
