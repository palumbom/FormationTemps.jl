using Revise
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Printf, Statistics
using PythonPlot; plt = PythonPlot.pyplot
plt.style.use(joinpath(FT.moddir, "fig.mplstyle"))

# output directories
plotdir = joinpath(FT.moddir, "docs", "src", "static")
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

# wavelength grid
buffer = 0.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.005)

# atmosphere + absorption
A_X = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs = atm_gpu.zs
Natm = length(zs)
Nλ = length(λs_korg)
Npad = 2400

αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# GPU memory
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, Natm)
σ_v_mic = CUDA.zeros(Float64, Natm) .+ 1200.0

# get a spectrum to convolve
cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
tbc = Array(cfunc_flux_stationary.cfunc_dt)

# broadening params
vsini = 4200.0
u1 = 0.4
u2 = 0.26
ζ_rt = 1200.0

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
gpu_median_ms = Float64[]
cpu_iqr_ms = Float64[]
gpu_iqr_ms = Float64[]

function record!(label, cpu_times, gpu_times)
    cpu_ms = cpu_times .* 1000
    gpu_ms = gpu_times .* 1000
    push!(kernels, label)
    push!(cpu_median_ms, median(cpu_ms))
    push!(gpu_median_ms, median(gpu_ms))
    push!(cpu_iqr_ms, iqr(cpu_ms))
    push!(gpu_iqr_ms, iqr(gpu_ms))
    speedup = median(cpu_ms) / median(gpu_ms)
    @printf("  CPU: %.3f ± %.3f ms  GPU: %.3f ± %.3f ms  (%.0f×)\n",
            median(cpu_ms), iqr(cpu_ms), median(gpu_ms), iqr(gpu_ms), speedup)
end

# microturbulence
println("Microturbulence...")
record!("Microturbulence",
    time_cpu() do
        FT.convolve_wavelength_axis(λs_korg, αs, Array(μ_v_rot), Array(σ_v_mic))
    end,
    time_gpu() do
        FT.convolve_wavelength_axis_gpu(cmem, CuArray(λs_korg), CuArray(αs), μ_v_rot, σ_v_mic)
    end)

# gray rotation
println("Gray rotation...")
record!("Gray rotation",
    time_cpu() do
        FT.convolve_gray_rotation(λs_korg, tbc, vsini, u1)
    end,
    time_gpu() do
        FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, tbc, vsini, u1)
    end)

# isotropic RT macroturbulence
println("Isotropic RT macro...")
record!("Iso. RT macro",
    time_cpu() do
        FT.convolve_iso_rt_macro(λs_korg, tbc, ζ_rt)
    end,
    time_gpu() do
        FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt)
    end)

# anisotropic RT macroturbulence
println("Anisotropic RT macro...")
record!("Aniso. RT macro",
    time_cpu() do
        FT.convolve_rt_macro(λs_korg, tbc, ζ_rt, 0.9)
    end,
    time_gpu() do
        FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt, 0.9)
    end)

# Hirano rotation+macro
println("Hirano rot+macro...")
record!("Hirano rot+macro",
    time_cpu() do
        FT.convolve_hirano_rotmacro(λs_korg, tbc, vsini, ζ_rt, u1, u2)
    end,
    time_gpu() do
        FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini, ζ_rt, u1, u2)
    end)

# ── save data ─────────────────────────────────────────────────────────────────
open(joinpath(datadir, "convolution_timings.csv"), "w") do io
    println(io, "kernel,cpu_median_ms,cpu_iqr_ms,gpu_median_ms,gpu_iqr_ms,speedup")
    for i in eachindex(kernels)
        @printf(io, "%s,%.4f,%.4f,%.4f,%.4f,%.1f\n",
                kernels[i], cpu_median_ms[i], cpu_iqr_ms[i],
                gpu_median_ms[i], gpu_iqr_ms[i],
                cpu_median_ms[i] / gpu_median_ms[i])
    end
end
println("\nData written to: ", joinpath(datadir, "convolution_timings.csv"))

# ── plot ──────────────────────────────────────────────────────────────────────
plt.ioff()

fig, ax = plt.subplots(figsize=(8, 4))
x = 0:length(kernels)-1
w = 0.35
bars_cpu = ax.bar(x .- w / 2, cpu_median_ms, w,
                  yerr=cpu_iqr_ms, capsize=3, ecolor="#333333",
                  label="{\\rm CPU}", color="#56B4E9", edgecolor="none")
bars_gpu = ax.bar(x .+ w / 2, gpu_median_ms, w,
                  yerr=gpu_iqr_ms, capsize=3, ecolor="#333333",
                  label="{\\rm GPU}", color="#D55E00", edgecolor="none")

# speedup annotations with square brackets
for i in eachindex(kernels)
    speedup = cpu_median_ms[i] / gpu_median_ms[i]
    y_top = max(cpu_median_ms[i] + cpu_iqr_ms[i], gpu_median_ms[i] + gpu_iqr_ms[i])
    xi = i - 1

    # square bracket encompassing both bars
    x_left = xi - w
    x_right = xi + w
    y_brace = y_top * 1.15
    tick_h = y_brace * 0.08
    ax.plot([x_left, x_left, x_right, x_right],
            [y_brace - tick_h, y_brace, y_brace, y_brace - tick_h],
            color="#333333", lw=1.0, clip_on=false)

    # speedup text above bracket
    ax.text(xi, y_brace * 1.08, @sprintf("\$\\sim %.0f\\times\$", speedup),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
end

ax.set_xticks(collect(x))
ax.set_xticklabels(["{\\rm " * k * "}" for k in kernels], rotation=20, ha="right")
ax.set_ylabel("{\\rm Time [ms]}")
ax.set_title(@sprintf("{\\rm Convolution kernel timings (}\$N_\\lambda\${\\rm =%d, %d trials)}", Nλ, N_REPEAT))
ax.legend()
ax.set_yscale("log")
ax.grid(false)

# add vertical buffer so annotations aren't clipped by the top spine
y_max_data = maximum(cpu_median_ms .+ cpu_iqr_ms)
ax.set_ylim(nothing, y_max_data * 5.0)

fig.tight_layout()
fig.savefig(joinpath(plotdir, "benchmark_convolutions.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
println("Saved: benchmark_convolutions.png")

println()
println("DONE")
