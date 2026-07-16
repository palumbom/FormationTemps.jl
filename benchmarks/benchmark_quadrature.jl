# Robust quadrature benchmark: accuracy & speed sweeps for method=:quadrature.
#
# Sweeps (each writes a CSV to benchmarks/data/):
#   • Nμ  — μ-quadrature nodes: accuracy (vs high-res tiling) + time  → quadrature_nmu.csv
#   • N_az — ring azimuth samples: accuracy + time                    → quadrature_naz.csv
#   • vsini — accuracy of :quadrature and :hirano vs tiling           → quadrature_vsini.csv
#   • Nλ   — wall-time of :disk/:quadrature/:hirano × CPU/GPU          → quadrature_scaling.csv
#
# Accuracy is always max/mean interior formation-temperature (+ flux) difference vs an
# explicit high-resolution tiling reference. Timing is the minimum of n_time @elapsed
# runs after a warmup (GPU calls return host arrays, so @elapsed includes device sync).
#
# Run:  julia --project=. -t auto benchmarks/benchmark_quadrature.jl

using FormationTemps; const FT = FormationTemps
using Korg, Statistics, Printf

const PROJECT_DIR = dirname(@__DIR__)
const DATADIR = joinpath(PROJECT_DIR, "benchmarks", "data")
!isdir(DATADIR) && mkpath(DATADIR)

const have_gpu = FT.GPU_DEFAULT
const Nϕ_ref = have_gpu ? 192 : 128     # high-res tiling reference (accuracy)
const ref_gpu = have_gpu                # compute the reference on GPU if available (fast)

# fixed stellar params
const Teff, logg, Fe_H = 5777.0, 4.44, 0.0
const ζ_RT, ξ = 3400.0, 850.0
const istar = 60.0
const Δλ0 = 0.01
const u1, u2 = 0.43, 0.31

load_linelist(range) = begin
    ll = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[range]
    [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in ll]
end

mkstar(; vsini, α₂=0.0, α₄=0.0) = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H,
    vsini=vsini, v_macro=ζ_RT, v_micro=ξ, istar=istar, α₂=α₂, α₄=α₄)

function runit(star, ll, method, ug; Δλ=Δλ0, Nϕ=64, Nμ=16, N_az=256)
    common = (Δλ=Δλ, use_gpu=ug, method=method, showprogress=false, ne_warn_thresh=Inf)
    if method === :hirano
        calc_formation_temp(star, ll; common..., u1=u1, u2=u2, Nϕ=Nϕ)
    elseif method === :quadrature
        calc_formation_temp(star, ll; common..., Nμ=Nμ, N_az=N_az)
    else
        calc_formation_temp(star, ll; common..., Nϕ=Nϕ)
    end
end

function bench(f; n=5)
    f()                                   # warmup (compilation / FFT plans)
    minimum(@elapsed(f()) for _ in 1:n)
end

function interior(wavs, vmax; Δλ=Δλ0)
    λ0 = mean(wavs); nλ = length(wavs)
    edge = ceil(Int, vmax * 3 / (FT.c_ms * Δλ / λ0)) + 10
    (edge + 1):(nλ - edge)
end

ferr(r, ref, I) = (maximum(abs.(r.form_temps[I] .- ref.form_temps[I])),
                   mean(abs.(r.form_temps[I] .- ref.form_temps[I])),
                   maximum(abs.(r.flux[I] .- ref.flux[I])))

function writecsv(name, header, rows, fmt)
    f = Printf.Format(fmt)                # runtime format (fmt is not a literal here)
    open(joinpath(DATADIR, name), "w") do io
        println(io, header)
        for r in rows
            Printf.format(io, f, r...)
        end
    end
    println("  wrote ", name)
end

# ── sweeps ───────────────────────────────────────────────────────────────────
function sweep_nodes()
    ll = load_linelist(16000:16010)
    star = mkstar(vsini=15000.0)
    ref = runit(star, ll, :disk, ref_gpu; Nϕ=Nϕ_ref)
    I = interior(ref.wavs, max(15000.0, ζ_RT))

    println("Nμ sweep (vsini=15 km/s, N_az=256):")
    nmu_rows = map((4, 6, 8, 12, 16, 24, 32)) do Nμ
        f() = runit(star, ll, :quadrature, false; Nμ=Nμ, N_az=256)
        e = ferr(f(), ref, I); t = bench(f)
        @printf("  Nμ=%2d  formT max=%.3f mean=%.4f K  t=%.0f ms\n", Nμ, e[1], e[2], 1e3t)
        (Nμ, e[1], e[2], e[3], 1e3t)
    end
    writecsv("quadrature_nmu.csv", "Nmu,formT_max,formT_mean,flux_max,time_ms",
             nmu_rows, "%d,%.5f,%.5f,%.3e,%.3f\n")

    println("N_az sweep (vsini=15 km/s, Nμ=16):")
    naz_rows = map((32, 64, 128, 256, 512)) do N_az
        f() = runit(star, ll, :quadrature, false; Nμ=16, N_az=N_az)
        e = ferr(f(), ref, I); t = bench(f)
        @printf("  N_az=%3d  formT max=%.3f mean=%.4f K  t=%.0f ms\n", N_az, e[1], e[2], 1e3t)
        (N_az, e[1], e[2], e[3], 1e3t)
    end
    writecsv("quadrature_naz.csv", "Naz,formT_max,formT_mean,flux_max,time_ms",
             naz_rows, "%d,%.5f,%.5f,%.3e,%.3f\n")
end

function sweep_vsini()
    ll = load_linelist(16000:16010)
    println("vsini sweep (accuracy vs tiling; Nμ=16, N_az=256):")
    rows = []
    for vk in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
        star = mkstar(vsini=1000vk)
        ref = runit(star, ll, :disk, ref_gpu; Nϕ=Nϕ_ref)
        I = interior(ref.wavs, max(1000vk, ζ_RT))
        for m in (:quadrature, :hirano)
            e = ferr(runit(star, ll, m, false), ref, I)
            push!(rows, (vk, String(m), e[1], e[2], e[3]))
            @printf("  vsini=%4.0f km/s  %-11s formT max=%.3f K\n", vk, m, e[1])
        end
    end
    writecsv("quadrature_vsini.csv", "vsini_kms,method,formT_max,formT_mean,flux_max",
             rows, "%.1f,%s,%.5f,%.5f,%.3e\n")
end

function sweep_scaling()
    ll = load_linelist(16000:16010)
    star = mkstar(vsini=5000.0)
    println("Nλ scaling (time; :disk/:quadrature/:hirano × device):")
    rows = []
    for dλ in (0.05, 0.02, 0.01, 0.005, 0.0025)
        Nλ = length(runit(star, ll, :quadrature, false; Δλ=dλ).wavs)
        configs = [(:disk, false), (:quadrature, false), (:hirano, false)]
        have_gpu && append!(configs, [(:disk, true), (:quadrature, true), (:hirano, true)])
        for (m, ug) in configs
            t = bench(() -> runit(star, ll, m, ug; Δλ=dλ, Nϕ=64); n=3)
            push!(rows, (String(m), ug ? "gpu" : "cpu", Nλ, 1e3t))
            @printf("  Nλ=%5d  %-11s %s: %.1f ms\n", Nλ, m, ug ? "gpu" : "cpu", 1e3t)
        end
    end
    writecsv("quadrature_scaling.csv", "method,device,Nlambda,time_ms",
             rows, "%s,%s,%d,%.3f\n")
end

# ── run ──────────────────────────────────────────────────────────────────────
println("="^64)
println("QUADRATURE BENCHMARK   (GPU ", have_gpu ? "available" : "unavailable", ")")
println("="^64)
sweep_nodes()
sweep_vsini()
sweep_scaling()
println("\nData written to ", DATADIR)
println("Plot with:  julia --project=. benchmarks/plot_quadrature.jl")
