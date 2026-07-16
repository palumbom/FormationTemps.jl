# Benchmark: accuracy & speed of the disk-integration methods behind
# `calc_formation_temp`, on CPU and GPU.
#
#   method = :disk        explicit tile-based disk integration (the reference)
#          = :quadrature  ring-by-ring μ-quadrature (supplement)
#          = :hirano      analytic rotation+macro convolution
#
# Speed: wall-clock of the full `calc_formation_temp` call (shared atmosphere +
#   Korg chemistry setup is common to every method, so timing differences reflect
#   the integration/broadening step). Reports the minimum of `n_time` runs after a
#   warmup call (which absorbs compilation).
# Accuracy: max/mean interior formation-temperature and flux difference vs a
#   high-resolution tiling reference.
#
# Run:  julia --project=. -t auto scripts/benchmark_disk_methods.jl

using FormationTemps; const FT = FormationTemps
using Korg
using Statistics
using Printf

# ── configuration ────────────────────────────────────────────────────────────
const Teff, logg, Fe_H = 5777.0, 4.44, 0.0
const vsini, ζ_RT, ξ   = 5000.0, 3400.0, 850.0   # m/s
const istar            = 60.0                     # degrees
const α₂, α₄           = 0.0, 0.0                 # rigid; set >0 to benchmark diff. rotation
const u1, u2           = 0.43, 0.31               # limb darkening (Hirano only)
const Δλ               = 0.01
const Nμ, N_az         = 16, 256                  # quadrature resolution
const n_time           = 5                        # timing repeats (report minimum)

const have_gpu = FT.GPU_DEFAULT
const Nϕ_bench = 96                               # tiling resolution in the timed runs
const Nϕ_ref   = have_gpu ? 192 : 128             # high-res tiling reference (accuracy)

# representative linelist chunk (grid follows the line extremes ± 2 Å buffer)
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16300]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT,
                    v_micro=ξ, istar=istar, α₂=α₂, α₄=α₄)

# ── helpers ──────────────────────────────────────────────────────────────────
function runit(method::Symbol, use_gpu::Bool; Nϕ::Int=Nϕ_bench)
    common = (Δλ=Δλ, use_gpu=use_gpu, method=method, showprogress=false, ne_warn_thresh=Inf)
    if method === :hirano
        calc_formation_temp(star, linelist; common..., u1=u1, u2=u2, Nϕ=Nϕ)
    elseif method === :quadrature
        calc_formation_temp(star, linelist; common..., Nμ=Nμ, N_az=N_az)
    else # :disk
        calc_formation_temp(star, linelist; common..., Nϕ=Nϕ)
    end
end

# minimum wall-clock over n runs, after one warmup (GPU calls return host arrays,
# so @elapsed already includes device synchronization)
function bench(f; n::Int=n_time)
    f()
    minimum(@elapsed(f()) for _ in 1:n)
end

function errs(r, ref, interior)
    ft = abs.(r.form_temps[interior] .- ref.form_temps[interior])
    fx = abs.(r.flux .- ref.flux)
    return (maximum(ft), mean(ft), maximum(fx))
end

# ── reference & problem size ─────────────────────────────────────────────────
println("Problem: Teff=$Teff logg=$logg vsini=$(vsini/1000) km/s istar=$(istar)° " *
        "α=($α₂,$α₄)  |  $(length(linelist)) lines")
ref = runit(:disk, have_gpu; Nϕ=Nϕ_ref)
Nλ = length(ref.wavs)
λ0 = mean(ref.wavs)
edge = ceil(Int, max(vsini, ζ_RT) * 3 / (FT.c_ms * Δλ / λ0)) + 10
interior = (edge + 1):(Nλ - edge)
@printf("Grid: Nλ=%d (interior %d)   tiling ref: Nϕ=%d (%s)   quad: Nμ=%d, N_az=%d\n\n",
        Nλ, length(interior), Nϕ_ref, have_gpu ? "GPU" : "CPU", Nμ, N_az)

# ── run configs ──────────────────────────────────────────────────────────────
configs = [(:disk, false), (:quadrature, false), (:hirano, false)]
have_gpu && append!(configs, [(:disk, true), (:quadrature, true), (:hirano, true)])

# time + score each config once
results = map(configs) do (method, use_gpu)
    r = runit(method, use_gpu)
    t = bench(() -> runit(method, use_gpu))
    (method=method, use_gpu=use_gpu, t=t, errs=errs(r, ref, interior))
end

# speedup baseline: :disk on CPU
t_baseline = results[findfirst(x -> x.method === :disk && !x.use_gpu, results)].t

println(rpad("method", 12), rpad("device", 7), rpad("time [ms]", 12),
        rpad("speedup", 10), rpad("formT max/mean [K]", 22), "flux max")
println("-"^75)
for x in results
    ftmax, ftmean, fxmax = x.errs
    @printf("%-12s%-7s%-12.1f%-10s%-22s%.2e\n",
            String(x.method), x.use_gpu ? "GPU" : "CPU", 1e3 * x.t,
            @sprintf("%.1f×", t_baseline / x.t),
            @sprintf("%.2f / %.3f", ftmax, ftmean), fxmax)
end

println("\nNotes:")
println(" • Accuracy is measured vs the high-res tiling reference (Nϕ=$Nϕ_ref).")
println(" • :hirano uses a PARAMETRIC limb-darkening law (u1=$u1, u2=$u2) and a")
println("   shift-invariant kernel, so its formT difference vs tiling is a physical")
println("   model difference, not just numerics — expected to exceed :quadrature's.")
println(" • :disk at Nϕ=$Nϕ_bench shows the tiling method's own convergence vs the reference.")
println(" • Speedup is relative to :disk on CPU. Set α₂,α₄ > 0 to benchmark differential rotation.")
