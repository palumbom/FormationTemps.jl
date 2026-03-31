using FormationTemps; FT = FormationTemps
using Korg
using CUDA
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
idx1 = findfirst(x -> x * 1e8 >= 6301, wls)
idx2 = findfirst(x -> x * 1e8 >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

wls = [l.wl * 1e8 for l in linelist]

# wavelength grid (Δλ matches all other benchmark scripts)
Δλ = 0.005
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
μ_v_rot64 = CUDA.zeros(Float64, Natm)
σ_v_mic64 = CUDA.zeros(Float64, Natm) .+ 850.0

# velocities — Float32 (vector, for GPU per-row path)
μ_v_rot32 = CUDA.zeros(Float32, Natm)
σ_v_mic32 = CUDA.zeros(Float32, Natm) .+ Float32(850.0)

# scalar σ_v for production-matching CPU in-place path
σ_v_scalar = 850.0

# CPU tile workspace (matches production in-place path)
cpu_ws = FT.CPUTileWorkspace(Float64, Natm, Nλ)

# pre-allocate GPU arrays for microturbulence (avoid H2D transfer in loop)
λs_gpu64 = CuArray(collect(Float64, λs_korg))
αs_gpu64 = CuArray(αs)
λs_gpu32 = CuArray(collect(Float32, λs_korg))
αs_gpu32 = CuArray(Float32.(αs))

# get a spectrum to convolve
cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem64, σ_v_mic64)
tbc64 = Array(cfunc_flux_stationary.cfunc_dt)
tbc32 = Float32.(tbc64)
λs_korg_f32 = Float32.(collect(λs_korg))

# broadening params
vsini = 2100.0
u1 = 0.4
u2 = 0.26
ζ_rt = 3500.0

const N_REPEAT = 20

println("Convolution benchmark: Nλ=", Nλ, ", Natm=", Natm, ", Npad=", Npad,
        ", N_REPEAT=", N_REPEAT)
println()

# ── timing helper ──────────────────────────────────────────────────────────────
function iqr(x)
    q = quantile(x, [0.25, 0.75])
    return (q[2] - q[1]) / 2.0
end

function time_cpu(f; n_repeat=N_REPEAT)
    f()  # warmup
    return [(@elapsed f()) for _ in 1:n_repeat]
end

function time_gpu(f; n_repeat=N_REPEAT)
    f(); CUDA.synchronize()  # warmup
    times = zeros(n_repeat)
    for r in 1:n_repeat
        CUDA.synchronize()
        times[r] = CUDA.@elapsed f()
    end
    return times
end

# ── benchmark each kernel ──────────────────────────────────────────────────────
kernels = String[]
cpu_median_ms = Float64[]
gpu64_median_ms = Float64[]
gpu32_median_ms = Float64[]
cpu_iqr_ms = Float64[]
gpu64_iqr_ms = Float64[]
gpu32_iqr_ms = Float64[]

function record!(label, cpu_times, gpu64_times, gpu32_times)
    cpu_ms = cpu_times .* 1000
    g64_ms = gpu64_times .* 1000
    g32_ms = gpu32_times .* 1000
    push!(kernels, label)
    push!(cpu_median_ms, median(cpu_ms))
    push!(gpu64_median_ms, median(g64_ms))
    push!(gpu32_median_ms, median(g32_ms))
    push!(cpu_iqr_ms, iqr(cpu_ms))
    push!(gpu64_iqr_ms, iqr(g64_ms))
    push!(gpu32_iqr_ms, iqr(g32_ms))
    sp64 = median(cpu_ms) / median(g64_ms)
    sp32 = median(cpu_ms) / median(g32_ms)
    @printf("  CPU: %.3f ms  GPU64: %.3f ms (%.0f×)  GPU32: %.3f ms (%.0f×)\n",
            median(cpu_ms), median(g64_ms), sp64, median(g32_ms), sp32)
end

# microturbulence (CPU uses production in-place scalar-σ path)
println("Microturbulence...")
record!("Microturbulence",
    time_cpu() do
        FT._convolve_micro_inplace!(cpu_ws.αs_broad, λs_korg, αs, 0.0, σ_v_scalar, cpu_ws)
    end,
    time_gpu() do
        FT.convolve_wavelength_axis_gpu(cmem64, λs_gpu64, αs_gpu64, μ_v_rot64, σ_v_mic64)
    end,
    time_gpu() do
        FT.convolve_wavelength_axis_gpu(cmem32, λs_gpu32, αs_gpu32, μ_v_rot32, σ_v_mic32)
    end)

# gray rotation
println("Gray rotation...")
record!("Gray rotation",
    time_cpu() do
        FT.convolve_gray_rotation(λs_korg, tbc64, vsini, u1)
    end,
    time_gpu() do
        FT.convolve_gray_rotation_gpu(cmem_mac64, λs_korg, tbc64, vsini, u1)
    end,
    time_gpu() do
        FT.convolve_gray_rotation_gpu(cmem_mac32, λs_korg_f32, tbc32, Float32(vsini), Float32(u1))
    end)

# isotropic RT macroturbulence
println("Isotropic RT macro...")
record!("Iso. RT macro",
    time_cpu() do
        FT.convolve_iso_rt_macro(λs_korg, tbc64, ζ_rt)
    end,
    time_gpu() do
        FT.convolve_iso_rt_macro_gpu(cmem_mac64, λs_korg, tbc64, ζ_rt)
    end,
    time_gpu() do
        FT.convolve_iso_rt_macro_gpu(cmem_mac32, λs_korg_f32, tbc32, Float32(ζ_rt))
    end)

# anisotropic RT macroturbulence
println("Anisotropic RT macro...")
record!("Aniso. RT macro",
    time_cpu() do
        FT.convolve_rt_macro(λs_korg, tbc64, ζ_rt, 0.9)
    end,
    time_gpu() do
        FT.convolve_rt_macro_gpu(cmem_mac64, λs_korg, tbc64, ζ_rt, 0.9)
    end,
    time_gpu() do
        FT.convolve_rt_macro_gpu(cmem_mac32, λs_korg_f32, tbc32, Float32(ζ_rt), Float32(0.9))
    end)

# Hirano rotation+macro
println("Hirano rot+macro...")
record!("Hirano rot+macro",
    time_cpu() do
        FT.convolve_hirano_rotmacro(λs_korg, tbc64, vsini, ζ_rt, u1, u2)
    end,
    time_gpu() do
        FT.convolve_hirano_rotmacro_gpu(cmem_mac64, λs_korg, tbc64, vsini, ζ_rt, u1, u2)
    end,
    time_gpu() do
        FT.convolve_hirano_rotmacro_gpu(cmem_mac32, λs_korg_f32, tbc32,
            Float32(vsini), Float32(ζ_rt), Float32(u1), Float32(u2))
    end)

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
