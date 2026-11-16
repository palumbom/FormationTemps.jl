# A Doppler row whose kernel reaches past the window's taps is recomputed by direct convolution,
# on the device by doppler_direct_kernel! and on the host by _doppler_direct_rows_cpu!. The two
# share the domain test and the taps but sum differently, so an edit to one that is not mirrored
# in the other shows up here as a CPU/GPU disagreement on those rows. GPU-only, so CI cannot run
# it.

using FormationTemps
using Test

if FormationTemps.GPU_DEFAULT
    using CUDA

    let
        FT = FormationTemps

        @testset "CPU and GPU Doppler fallback agree" begin
            # ±0.1 Å at 1 mÅ: the domain ends where |v_los| + 3.7·v_mic reaches about 4.9 km/s
            Nλ = 201
            xs = collect(range(6173.0 - 0.1, 6173.0 + 0.1, length=Nλ))
            λ0 = xs[Nλ ÷ 2 + 1]
            σf = FT._sigma_floor(xs)

            # sloped continuum and a deep line, so the two window edges differ and a clamp to the
            # wrong edge shows
            profile(scale, tilt) = [scale * (1 + tilt * (x - λ0) / 0.1 +
                                             60 * exp(-((x - 6173.02) / 0.01)^2)) for x in xs]
            # rows 1-3 inside the domain; 4-7 shifted past the span on either side; 8-9 wider
            # than the window, with kernel mass beyond both ends; 10-11 wholly past one end
            v_los = [0.0, 1e3, 0.0, 2e3, -2e3, 4e3, -6e3, 0.0, 1e3, 20e3, -20e3]
            v_mic = [1e3, 0.8e3, 0.3e3, 1e3, 1e3, 0.5e3, 0.5e3, 3e3, 6e3, 1e3, 1e3]
            Natm = length(v_los)
            ys = permutedims(hcat([profile(1 + 0.2 * r, isodd(r) ? 0.3 : -0.3) for r in 1:Natm]...))
            indom = [FT._doppler_in_domain_cpu(xs, v_los[r], v_mic[r], σf) for r in 1:Natm]
            out = .!indom
            @test indom == [true, true, true, false, false, false, false, false, false, false, false]

            Npad = FT.conv_npad_for_velocity(λ0, xs[2] - xs[1],
                                             FT.conv_kernel_vmax(maximum(abs, v_los), 0.0, v_mic))
            function gpu(y, vl, vm)
                cmem = FT.ConvolutionMemory(Nλ, Natm, Npad; T=Float64)
                FT._init_micro_params!(cmem, xs, xs)
                vl_d = vl isa AbstractVector ? CuArray(vl) : vl
                vm_d = vm isa AbstractVector ? CuArray(vm) : vm
                return Array(FT.convolve_wavelength_axis_gpu(cmem, CuArray(xs), CuArray(y), vl_d, vm_d))
            end
            function cpu_inplace(y, vl, vm)
                ws = FT.CPUTileWorkspace(Float64, Natm, Nλ; Npad=Npad)
                o = similar(y)
                FT._convolve_micro_inplace!(o, xs, y, vl, vm, ws)
                return o
            end
            relmax(a, b) = maximum(abs.(a .- b) ./ abs.(b))

            @testset "per-row velocities" begin
                c = FT.convolve_wavelength_axis(xs, ys, v_los, v_mic)
                g = gpu(ys, v_los, v_mic)
                @test relmax(c[out, :], g[out, :]) < 1e-12
                @test relmax(c[indom, :], g[indom, :]) < 1e-8
                # a kernel wholly past one end reads that end's pixel, on both sides
                @test all(c[10, :] .== ys[10, 1]) && all(g[10, :] .== ys[10, 1])
                @test all(c[11, :] .== ys[11, Nλ]) && all(g[11, :] .== ys[11, Nλ])
                # the in-place path recomputes the same rows with the same function
                @test cpu_inplace(ys, v_los, v_mic)[out, :] == c[out, :]
            end

            @testset "one kernel for every row" begin
                c = FT.convolve_wavelength_axis(xs, ys, 0.0, 3e3)
                g = gpu(ys, 0.0, 3e3)
                @test relmax(c, g) < 1e-12
                @test cpu_inplace(ys, 0.0, 3e3) == c
            end

            @testset "uniform opacity stays uniform" begin
                u = ones(Natm, Nλ)
                c = FT.convolve_wavelength_axis(xs, u, v_los, v_mic)
                g = gpu(u, v_los, v_mic)
                @test maximum(abs.(c[out, :] .- 1)) < 1e-13
                @test maximum(abs.(g[out, :] .- 1)) < 1e-13
            end
        end
    end
end
