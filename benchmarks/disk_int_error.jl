using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Printf
using ProgressMeter

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport
np = pyimport("numpy")
plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
plt.ioff()

py(x::AbstractVector) = np.asarray(collect(Float64, x))

const plotdir = joinpath(FT.moddir, "docs", "src", "static")
mkpath(plotdir)

# consistent palette (matches plot_benchmarks.jl)
const COL_MEAN = "#2A96D1"  # dark blue
const COL_MAX  = "#D55E00"  # orange

# ── linelist setup (Fe I 6301/6302) ──────────────────────────────────────────
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs    = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls      = [l.wl for l in linelist]
idx1     = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2     = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])
wls      = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]

λs_korg = range(first(wls) - 1.0, last(wls) + 1.0, step=0.01)

A_X     = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs      = atm_gpu.zs

αs      = zeros(length(zs), length(λs_korg))
αs_cont = zeros(length(zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

Nλ   = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem    = FT.ConvolutionMemory(Nλ, Natm, Npad)
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

v_los = CUDA.zeros(Float64, length(zs))
v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

# ── reference: direct flux (no disk integration) ────────────────────────────
println("Computing reference flux (direct, no disk integration)...")
cfunc_flux_ref = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, v_mic)
flux_ref       = Array(FT.get_flux(cfunc_flux_ref))

# ── disk integration at several resolutions ──────────────────────────────────
Nϕ_vals = [16, 32, 64, 128, 256, 512]

mean_pct_error = zeros(length(Nϕ_vals))
max_pct_error  = zeros(length(Nϕ_vals))

ρstar = 1.0
istar = 90.0
v0    = 0.0

for j in eachindex(Nϕ_vals)
    println(@sprintf("  Nϕ = %d ...", Nϕ_vals[j]))
    μs, dA, z_rot, _ = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ_vals[j])

    idx       = findall(x -> x > zero(eltype(μs)), Array(μs))
    μs_cpu    = view(Array(μs), idx)
    dA_cpu    = view(Array(dA), idx)

    flux_disk = CUDA.zeros(Float64, length(λs_korg))
    @showprogress for i in eachindex(μs_cpu)
        cfunc_i = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], v_los, v_mic)
        flux_disk .+= FT.get_intensity(cfunc_i) .* dA_cpu[i]
    end

    mean_pct_error[j] = mean(abs.(100.0 .* (flux_ref .- Array(flux_disk)) ./ flux_ref))
    max_pct_error[j]  = maximum(abs.(100.0 .* (flux_ref .- Array(flux_disk)) ./ flux_ref))
end

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(py(Nϕ_vals), py(mean_pct_error), "o-", color=COL_MEAN, lw=2, ms=7,
        label="{\\rm Mean abs.\\ error}")
ax.plot(py(Nϕ_vals), py(max_pct_error), "s--", color=COL_MAX, lw=2, ms=7,
        label="{\\rm Max abs.\\ error}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("\$N_\\phi\$")
ax.set_ylabel("{\\rm Percent error vs.\\ direct flux}")
ax.set_title(@sprintf("{\\rm Disk integration convergence (}\$N_\\lambda\${\\rm =%d)}", Nλ))
ax.legend()

ticker = PythonPlot.pyimport("matplotlib.ticker")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(false)
ax.grid(true, which="major", color="#DDDDDD", lw=0.5)
ax.grid(true, which="minor", color="#EEEEEE", lw=0.3)

fig.tight_layout()
fig.savefig(joinpath(plotdir, "disk_int_convergence.png"), dpi=150, bbox_inches="tight")
plt.close()
println("Saved: disk_int_convergence.png")

# ── summary ──────────────────────────────────────────────────────────────────
println()
println("=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
for j in eachindex(Nϕ_vals)
    @printf("  Nϕ = %4d :  mean = %.4f %%   max = %.4f %%\n",
            Nϕ_vals[j], mean_pct_error[j], max_pct_error[j])
end
println()
println("DONE")
