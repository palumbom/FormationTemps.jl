using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using BenchmarkTools
using Printf, Statistics
# output directory
datadir = joinpath(FT.moddir, "benchmarks", "data")
!isdir(datadir) && mkpath(datadir)

# ── setup ──────────────────────────────────────────────────────────────────────
# Fe I 6301 & 6302 lines
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]

# wavelength grid
Δλ = 0.0025
buffer = 2.0
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=Δλ)

# atmosphere + absorption
A_X = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs = atm_gpu.zs
Natm = length(zs)
Nλ = length(λs_korg)
Npad = 512

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# GPU memory — Float64
cmem64 = FT.ConvolutionMemory(Nλ, Natm, Npad; T=Float64)
cmem_mac64 = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad; T=Float64)
gpu_mem = FT.GPUMemory(collect(λs_korg), atm_gpu)

# GPU memory — Float32
cmem32 = FT.ConvolutionMemory(Nλ, Natm, Npad; T=Float32)
cmem_mac32 = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad; T=Float32)

# velocities — Float64 (vector, for GPU per-row path)
v_los_rot64 = CUDA.zeros(Float64, Natm)
v_mic64 = CUDA.zeros(Float64, Natm) .+ 850.0

# velocities — Float32 (vector, for GPU per-row path)
v_los_rot32 = CUDA.zeros(Float32, Natm)
v_mic32 = CUDA.zeros(Float32, Natm) .+ Float32(850.0)

# scalar v_mic for production-matching CPU in-place path
v_mic_scalar = 850.0

# vector v_mic / v_los for per-row CPU dispatch comparison
v_los_vec = fill(0.0, Natm)
v_mic_vec = fill(850.0, Natm)

# CPU tile workspace (matches production in-place path)
cpu_ws = FT.CPUTileWorkspace(Float64, Natm, Nλ)

# separate workspace for macro in-place benchmark (avoids state coupling with micro)
macro_ws = FT.CPUTileWorkspace(Float64, Natm, Nλ)

# pre-allocate GPU arrays for microturbulence (avoid H2D transfer in loop)
λs_gpu64 = CuArray(collect(Float64, λs_korg))
αs_gpu64 = CuArray(αs)
λs_gpu32 = CuArray(collect(Float32, λs_korg))
αs_gpu32 = CuArray(Float32.(αs))

# get a spectrum to convolve
cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem64, v_mic64)
tbc64 = Array(cfunc_flux_stationary.cfunc_dt)
tbc32 = Float32.(tbc64)
λs_korg_f32 = Float32.(collect(λs_korg))

# broadening params
vsini = 2100.0
u1 = 0.4
u2 = 0.26
ζ_rt = 3500.0

println("Convolution benchmark: Nλ=", Nλ, ", Natm=", Natm, ", Npad=", Npad)
println()

# ── timing helpers ─────────────────────────────────────────────────────────────
# BenchmarkTools handles warmup, GC fencing, and result consumption automatically.
# For GPU we use CUDA.@sync inside the benchmark to ensure kernel completion.
iqr_ms(trial) = (quantile(trial.times, 0.75) - quantile(trial.times, 0.25)) / 2e6

bench_cpu(f) = @benchmark $f()
bench_gpu(f) = @benchmark CUDA.@sync $f()

# ── benchmark each kernel ──────────────────────────────────────────────────────
kernels = String[]
cpu_median_ms = Float64[]
gpu64_median_ms = Float64[]
gpu32_median_ms = Float64[]
cpu_iqr_ms = Float64[]
gpu64_iqr_ms = Float64[]
gpu32_iqr_ms = Float64[]

function record!(label, cpu_trial, gpu64_trial, gpu32_trial)
    c = median(cpu_trial).time / 1e6
    g64 = median(gpu64_trial).time / 1e6
    g32 = median(gpu32_trial).time / 1e6
    push!(kernels, label)
    push!(cpu_median_ms, c)
    push!(gpu64_median_ms, g64)
    push!(gpu32_median_ms, g32)
    push!(cpu_iqr_ms, iqr_ms(cpu_trial))
    push!(gpu64_iqr_ms, iqr_ms(gpu64_trial))
    push!(gpu32_iqr_ms, iqr_ms(gpu32_trial))
    @printf("  CPU: %.3f ms  GPU64: %.3f ms (%.0f×)  GPU32: %.3f ms (%.0f×)\n",
            c, g64, c / g64, g32, c / g32)
end

# microturbulence — scalar dispatch (production disk-integration path)
println("Micro (scalar)...")
record!("Micro (scalar)",
    bench_cpu(() -> FT._convolve_micro_inplace!(cpu_ws.αs_broad, λs_korg, αs, 0.0, v_mic_scalar, cpu_ws)),
    bench_gpu(() -> FT.convolve_wavelength_axis_gpu(cmem64, λs_gpu64, αs_gpu64, v_los_rot64, v_mic64)),
    bench_gpu(() -> FT.convolve_wavelength_axis_gpu(cmem32, λs_gpu32, αs_gpu32, v_los_rot32, v_mic32)))

# microturbulence — per-row dispatch (per-layer v_mic path)
println("Micro (per-row)...")
record!("Micro (per-row)",
    bench_cpu(() -> FT._convolve_micro_inplace!(cpu_ws.αs_broad, λs_korg, αs, v_los_vec, v_mic_vec, cpu_ws)),
    bench_gpu(() -> FT.convolve_wavelength_axis_gpu(cmem64, λs_gpu64, αs_gpu64, v_los_rot64, v_mic64)),
    bench_gpu(() -> FT.convolve_wavelength_axis_gpu(cmem32, λs_gpu32, αs_gpu32, v_los_rot32, v_mic32)))

# gray rotation
println("Gray rotation...")
record!("Gray rotation",
    bench_cpu(() -> FT.convolve_gray_rotation(λs_korg, tbc64, vsini, u1)),
    bench_gpu(() -> FT.convolve_gray_rotation_gpu(cmem_mac64, λs_korg, tbc64, vsini, u1)),
    bench_gpu(() -> FT.convolve_gray_rotation_gpu(cmem_mac32, λs_korg_f32, tbc32, Float32(vsini), Float32(u1))))

# isotropic RT macroturbulence
println("Isotropic RT macro...")
record!("Iso. RT macro",
    bench_cpu(() -> FT.convolve_iso_rt_macro(λs_korg, tbc64, ζ_rt)),
    bench_gpu(() -> FT.convolve_iso_rt_macro_gpu(cmem_mac64, λs_korg, tbc64, ζ_rt)),
    bench_gpu(() -> FT.convolve_iso_rt_macro_gpu(cmem_mac32, λs_korg_f32, tbc32, Float32(ζ_rt))))

# anisotropic RT macroturbulence — in-place (production disk-integration path)
println("Aniso. RT (in-place)...")
record!("Aniso. RT (in-place)",
    bench_cpu(() -> FT._convolve_macro_inplace!(macro_ws.macro_out, λs_korg, tbc64, ζ_rt, 0.9, macro_ws)),
    bench_gpu(() -> FT.convolve_rt_macro_gpu(cmem_mac64, λs_korg, tbc64, ζ_rt, 0.9)),
    bench_gpu(() -> FT.convolve_rt_macro_gpu(cmem_mac32, λs_korg_f32, tbc32, Float32(ζ_rt), Float32(0.9))))

# anisotropic RT macroturbulence — allocating (standalone API)
println("Aniso. RT (alloc.)...")
record!("Aniso. RT (alloc.)",
    bench_cpu(() -> FT.convolve_rt_macro(λs_korg, tbc64, ζ_rt, 0.9)),
    bench_gpu(() -> FT.convolve_rt_macro_gpu(cmem_mac64, λs_korg, tbc64, ζ_rt, 0.9)),
    bench_gpu(() -> FT.convolve_rt_macro_gpu(cmem_mac32, λs_korg_f32, tbc32, Float32(ζ_rt), Float32(0.9))))

# Hirano rotation+macro
println("Hirano rot+macro...")
record!("Hirano rot+macro",
    bench_cpu(() -> FT.convolve_hirano_rotmacro(λs_korg, tbc64, vsini, ζ_rt, u1, u2)),
    bench_gpu(() -> FT.convolve_hirano_rotmacro_gpu(cmem_mac64, λs_korg, tbc64, vsini, ζ_rt, u1, u2)),
    bench_gpu(() -> FT.convolve_hirano_rotmacro_gpu(cmem_mac32, λs_korg_f32, tbc32,
        Float32(vsini), Float32(ζ_rt), Float32(u1), Float32(u2))))

# ── save data ─────────────────────────────────────────────────────────────────
open(joinpath(datadir, "convolution_timings.csv"), "w") do io
    println(io, "kernel,cpu_median_ms,cpu_iqr_ms,gpu64_median_ms,gpu64_iqr_ms,gpu32_median_ms,gpu32_iqr_ms")
    for i in eachindex(kernels)
        @printf(io, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                kernels[i], cpu_median_ms[i], cpu_iqr_ms[i],
                gpu64_median_ms[i], gpu64_iqr_ms[i],
                gpu32_median_ms[i], gpu32_iqr_ms[i])
    end
end

open(joinpath(datadir, "convolution_meta.csv"), "w") do io
    println(io, "Nlambda,Natm,Npad")
    @printf(io, "%d,%d,%d\n", Nλ, Natm, Npad)
end
println("\nData written to: ", joinpath(datadir, "convolution_timings.csv"))

println()
println("DONE")
