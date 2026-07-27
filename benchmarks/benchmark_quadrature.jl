# Accuracy & speed sweeps for method=:quadrature.
#
# Sweeps (each writes a CSV to benchmarks/data/):
#   • Nμ    — μ-quadrature nodes: accuracy (vs high-res tiling) + time → quadrature_nmu.csv
#   • N_az  — ring azimuth arcs: accuracy + time                       → quadrature_naz.csv
#   • Nϕ_ref — agreement vs the reference's own resolution             → quadrature_ref.csv
#   • Δλ    — agreement vs wavelength sampling (ring-kernel quantization) → quadrature_grid.csv
#   • vsini — accuracy of :quadrature and :hirano vs tiling            → quadrature_vsini.csv
#   • Nλ    — wall-time of :disk/:quadrature/:hirano × CPU/GPU         → quadrature_scaling.csv
#
# Provenance for every run is written to quadrature_meta.csv; a CSV without a matching meta
# row cannot be attributed to a machine and should be regenerated.
#
# Accuracy is the max/mean interior formation-temperature (+ flux) difference against an
# explicit tiling reference, and is deterministic -- it reproduces bit-identically between
# runs, so it is measured with a single call per configuration.
#
# Timing is not deterministic, so it is measured round-robin: see bench_roundrobin.
#
# Run:  julia --project=. -t auto benchmarks/benchmark_quadrature.jl [n_rounds]

using FormationTemps; const FT = FormationTemps
using Korg, Statistics, Printf, CUDA

const PROJECT_DIR = dirname(@__DIR__)
const DATADIR = joinpath(PROJECT_DIR, "benchmarks", "data")
!isdir(DATADIR) && mkpath(DATADIR)

const have_gpu = FT.GPU_DEFAULT
const Nϕ_ref = have_gpu ? 192 : 128     # high-res tiling reference (accuracy)
const ref_gpu = have_gpu                # compute the reference on GPU if available (fast)
const N_ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 7

# fixed stellar params
const Teff, logg, Fe_H = 5777.0, 4.44, 0.0
const ζ_RT, ξ = 3500.0, 850.0   # matches the canonical star in the other benchmark scripts
const istar = 60.0
const Δλ0 = 0.01
const u1, u2 = 0.43, 0.31

load_linelist(range) = begin
    ll = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[range]
    [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in ll]
end

mkstar(; vsini, α₂=0.0, α₄=0.0, istar_deg=istar) = StellarProps(Teff=Teff, logg=logg,
    Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ, istar=istar_deg, α₂=α₂, α₄=α₄)

# defaults here MUST track the `calc_formation_temp` defaults: the sweeps below are read as
# "what a user gets", so a helper that quietly overrides a default misreports the package.
function runit(star, ll, method, ug; Δλ=Δλ0, Nϕ=128, Nμ=32, N_az=256)
    common = (Δλ=Δλ, use_gpu=ug, method=method, showprogress=false, ne_warn_thresh=Inf)
    if method === :hirano
        calc_formation_temp(star, ll; common..., u1=u1, u2=u2, Nϕ=Nϕ)
    elseif method === :quadrature
        calc_formation_temp(star, ll; common..., Nμ=Nμ, N_az=N_az)
    else
        calc_formation_temp(star, ll; common..., Nϕ=Nϕ)
    end
end

"""
    bench_roundrobin(fs; n_rounds) -> [(min, median, max)]

Time each thunk once per round and repeat, rather than taking all samples for one
configuration before moving to the next. Thermal drift and transient contention (a co-tenant
on the GPU, another job on the node) then perturb every configuration alike instead of
penalizing whichever one happened to run during the episode. Consecutive-sample timing of
sub-100 ms GPU calls has been observed to swing 2.6x between runs this way.

The spread is returned, not just the minimum: two configurations whose intervals overlap are
not resolved by the measurement and must not be reported as different.
"""
function bench_roundrobin(fs; n_rounds=N_ROUNDS)
    for f in fs
        f()                               # warmup: compilation, FFT plans, device init
    end
    ts = [Float64[] for _ in fs]
    for _ in 1:n_rounds
        for (i, f) in enumerate(fs)
            push!(ts[i], @elapsed f())    # GPU calls return host arrays, so this syncs
        end
    end
    return [(minimum(t), median(t), maximum(t)) for t in ts]
end

function interior(wavs, vmax; Δλ=Δλ0)
    λ0 = mean(wavs); nλ = length(wavs)
    edge = ceil(Int, vmax * 3 / (FT.c_ms * Δλ / λ0)) + 10
    (edge + 1):(nλ - edge)
end

ferr(r, ref, I) = (maximum(abs.(r.form_temps[I] .- ref.form_temps[I])),
                   mean(abs.(r.form_temps[I] .- ref.form_temps[I])),
                   maximum(abs.(r.flux[I] .- ref.flux[I])))

"""
    adaptive_arcs(vsini, λ0; Δλ)

Arc count that `_ring_kernel_diffrot!` chooses on its own for the widest ring (`r_k = 1`):
32 arcs per velocity pixel of kernel support. `N_az` is a *floor* under this, so it only
influences the kernel when `N_az` exceeds this value -- which is what the N_az sweep has to
straddle to be measuring anything.
"""
adaptive_arcs(vsini, λ0; Δλ=Δλ0) = ceil(Int, 32 * 2 * vsini / (FT.c_ms * Δλ / λ0))

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

function write_meta()
    commit = try
        readchomp(`git -C $PROJECT_DIR rev-parse --short HEAD`)
    catch
        "unknown"
    end
    gpu = have_gpu ? CUDA.name(CUDA.device()) : "none"
    rows = [("host", gethostname()),
            ("git_commit", commit),
            ("julia_version", string(VERSION)),
            ("formationtemps_version", string(pkgversion(FormationTemps))),
            ("julia_threads", string(Threads.nthreads())),
            ("gpu", gpu),
            ("n_rounds", string(N_ROUNDS)),
            ("Nphi_ref", string(Nϕ_ref)),
            ("Nphi_default", "128"),
            ("Nmu_default", "32"),
            ("Naz_default", "256"),
            ("star", "Teff=$Teff logg=$logg Fe_H=$Fe_H zeta=$ζ_RT xi=$ξ"),
            ("dlambda_nominal", string(Δλ0))]
    writecsv("quadrature_meta.csv", "key,value", rows, "%s,%s\n")
end

# ── sweeps ───────────────────────────────────────────────────────────────────
function sweep_nmu()
    ll = load_linelist(16000:16010)
    star = mkstar(vsini=15000.0)
    ref = runit(star, ll, :disk, ref_gpu; Nϕ=Nϕ_ref)
    I = interior(ref.wavs, max(15000.0, ζ_RT))

    Nμs = (4, 6, 8, 12, 16, 24, 32, 48, 64)
    fs = [() -> runit(star, ll, :quadrature, false; Nμ=Nμ, N_az=256) for Nμ in Nμs]

    println("Nμ sweep (vsini=15 km/s, N_az=256, ref Nϕ=$Nϕ_ref):")
    errs = [ferr(f(), ref, I) for f in fs]
    ts = bench_roundrobin(fs)
    rows = map(eachindex(Nμs)) do i
        e, t = errs[i], ts[i]
        @printf("  Nμ=%2d  formT max=%.3f mean=%.4f K  t=%.0f ms (%.0f-%.0f)\n",
                Nμs[i], e[1], e[2], 1e3t[2], 1e3t[1], 1e3t[3])
        (Nμs[i], e[1], e[2], e[3], 1e3t[1], 1e3t[2], 1e3t[3])
    end
    writecsv("quadrature_nmu.csv",
             "Nmu,formT_max,formT_mean,flux_max,time_min_ms,time_med_ms,time_max_ms",
             rows, "%d,%.5f,%.5f,%.3e,%.3f,%.3f,%.3f\n")
end

function sweep_naz()
    ll = load_linelist(16000:16010)
    # α₂ != 0 so _ring_doppler_kernel takes the azimuth-sampled branch at all; for solid body
    # it evaluates the arcsine CDF analytically and never reads N_az.
    vsini = 15000.0
    star = mkstar(vsini=vsini, α₂=0.2)
    ref = runit(star, ll, :disk, ref_gpu; Nϕ=Nϕ_ref)
    I = interior(ref.wavs, max(vsini, ζ_RT))

    # N_az only binds above the adaptive arc count, so the sweep has to reach past it. Sweeping
    # the nominal default range measures nothing at this vsini: it is entirely below the floor.
    floor_arcs = adaptive_arcs(vsini, mean(ref.wavs))
    N_azs = (256, 1024, 2048, 4096, 8192, 16384)
    n_binding = count(>(floor_arcs), N_azs)
    @assert n_binding >= 3 "N_az sweep does not straddle the adaptive arc count " *
                           "($floor_arcs): only $n_binding of $(length(N_azs)) values bind. " *
                           "Extend N_azs upward or lower vsini, else this sweep is inert."

    println("N_az sweep (vsini=15 km/s, α₂=0.2, Nμ=32; adaptive floor = $floor_arcs arcs):")
    fs = [() -> runit(star, ll, :quadrature, false; Nμ=32, N_az=N_az) for N_az in N_azs]
    errs = [ferr(f(), ref, I) for f in fs]
    ts = bench_roundrobin(fs)
    rows = map(eachindex(N_azs)) do i
        e, t = errs[i], ts[i]
        binds = N_azs[i] > floor_arcs
        @printf("  N_az=%6d %-9s formT max=%.3f mean=%.4f K  t=%.0f ms\n",
                N_azs[i], binds ? "(binding)" : "(floored)", e[1], e[2], 1e3t[2])
        (N_azs[i], binds ? 1 : 0, e[1], e[2], e[3], 1e3t[1], 1e3t[2], 1e3t[3])
    end
    writecsv("quadrature_naz.csv",
             "Naz,binding,formT_max,formT_mean,flux_max,time_min_ms,time_med_ms,time_max_ms",
             rows, "%d,%d,%.5f,%.5f,%.3e,%.3f,%.3f,%.3f\n")
end

"""
    sweep_reference()

Vary the resolution of the *reference* at fixed `Nμ`. The Nμ sweep plateaus above Nμ≈32, and
this distinguishes the two available explanations: if the plateau is the μ-quadrature's own
error it is independent of `Nϕ_ref`, and if it is the floor at which the two methods agree
given the tiling's discretization it falls as `Nϕ_ref` rises. The conclusion drawn from the Nμ
sweep depends on which holds, so it is measured rather than assumed.
"""
function sweep_reference()
    ll = load_linelist(16000:16010)
    star = mkstar(vsini=15000.0)
    Nϕs = have_gpu ? (64, 96, 128, 192, 256, 384) : (64, 96, 128, 192)

    println("Reference-resolution sweep (vsini=15 km/s; Nμ ∈ {32, 64}):")
    rows = []
    for Nϕ in Nϕs
        ref = runit(star, ll, :disk, ref_gpu; Nϕ=Nϕ)
        I = interior(ref.wavs, max(15000.0, ζ_RT))
        for Nμ in (32, 64)
            e = ferr(runit(star, ll, :quadrature, false; Nμ=Nμ, N_az=256), ref, I)
            push!(rows, (Nϕ, Nμ, e[1], e[2], e[3]))
            @printf("  Nϕ_ref=%4d  Nμ=%2d  formT max=%.3f mean=%.4f K\n", Nϕ, Nμ, e[1], e[2])
        end
    end
    writecsv("quadrature_ref.csv", "Nphi_ref,Nmu,formT_max,formT_mean,flux_max",
             rows, "%d,%d,%.5f,%.5f,%.3e\n")
end

"""
    sweep_grid()

Vary the wavelength sampling at fixed `Nμ`. The reference sweep showed the `:quadrature`
residual converging to ~0.27 K rather than to zero as `Nϕ_ref` rises, and getting slightly
*worse* at `Nμ=64` than at `Nμ=32` — so the residual is set by neither the reference's tiling
nor the μ-quadrature. The remaining candidate is the ring kernel's bin positions, which are
quantized to `Δv = c·Δλ/λ`; if that is the mechanism the residual falls with `Δλ`.

The reference is recomputed on each grid, so both methods are compared on the same sampling.

`:hirano` is swept alongside to separate two mechanisms at the coarse end. At Δλ = 0.04 Å a
solar Fe line spans only a few pixels, so the *spectrum* is under-sampled and not just the ring
kernel. Comparing both methods against the same coarse reference says whether a coarse-grid
blowup is specific to `:quadrature`'s kernel quantization or generic to any method on an
under-sampled grid -- which is the difference between warning users off coarse grids with
`:quadrature` and warning them off coarse grids at all.

Also note: a finer grid means more pixels for the max to range over, which pushes the max the
other way, so read the trend rather than the exact ratio.
"""
function sweep_grid()
    ll = load_linelist(16000:16010)
    star = mkstar(vsini=15000.0)
    println("Wavelength-sampling sweep (vsini=15 km/s, Nμ=32, ref Nϕ=$Nϕ_ref):")
    rows = []
    for dλ in (0.04, 0.02, 0.01, 0.005, 0.0025)
        ref = runit(star, ll, :disk, ref_gpu; Δλ=dλ, Nϕ=Nϕ_ref)
        I = interior(ref.wavs, max(15000.0, ζ_RT); Δλ=dλ)
        Δv = FT.c_ms * dλ / mean(ref.wavs)
        width_px = 2 * 15000.0 / Δv          # ring-kernel full width, in wavelength pixels
        for m in (:quadrature, :hirano)
            e = ferr(runit(star, ll, m, false; Δλ=dλ, Nμ=32, N_az=256), ref, I)
            push!(rows, (dλ, Δv, width_px, length(ref.wavs), String(m), e[1], e[2], e[3]))
            @printf("  Δλ=%.4f Å (Δv=%6.1f m/s, %5.1f px, Nλ=%5d)  %-11s formT max=%8.3f mean=%.4f K\n",
                    dλ, Δv, width_px, length(ref.wavs), m, e[1], e[2])
        end
    end
    writecsv("quadrature_grid.csv",
             "dlambda_A,dv_ms,kernel_width_px,Nlambda,method,formT_max,formT_mean,flux_max",
             rows, "%.4f,%.2f,%.2f,%d,%s,%.5f,%.5f,%.3e\n")
end

function sweep_vsini()
    ll = load_linelist(16000:16010)
    println("vsini sweep (accuracy vs tiling; default Nμ=32, N_az=256):")
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
    # canonical benchmark star (vsini=2100, istar=90), so these timings are comparable to
    # benchmark_nlambda.jl, which plots the same axis for :disk alone
    star = mkstar(vsini=2100.0, istar_deg=90.0)
    println("Nλ scaling (time; :disk/:quadrature/:hirano × device; Nϕ=128, $N_ROUNDS rounds):")
    rows = []
    for dλ in (0.05, 0.02, 0.01, 0.005, 0.0025)
        Nλ = length(runit(star, ll, :quadrature, false; Δλ=dλ).wavs)
        configs = [(:disk, false), (:quadrature, false), (:hirano, false)]
        have_gpu && append!(configs, [(:disk, true), (:quadrature, true), (:hirano, true)])
        # all configs at this Nλ timed in one round-robin, so they share any drift
        fs = [() -> runit(star, ll, m, ug; Δλ=dλ, Nϕ=128) for (m, ug) in configs]
        ts = bench_roundrobin(fs)
        for (i, (m, ug)) in enumerate(configs)
            t = ts[i]
            push!(rows, (String(m), ug ? "gpu" : "cpu", Nλ, 1e3t[1], 1e3t[2], 1e3t[3]))
            @printf("  Nλ=%5d  %-11s %s: %7.1f ms  (min %.1f, max %.1f)\n",
                    Nλ, m, ug ? "gpu" : "cpu", 1e3t[2], 1e3t[1], 1e3t[3])
        end
    end
    writecsv("quadrature_scaling.csv",
             "method,device,Nlambda,time_min_ms,time_med_ms,time_max_ms",
             rows, "%s,%s,%d,%.3f,%.3f,%.3f\n")
end

# ── run ──────────────────────────────────────────────────────────────────────
println("="^64)
println("QUADRATURE BENCHMARK   (GPU ", have_gpu ? "available" : "unavailable", ")")
println("  threads = ", Threads.nthreads(), ", timing rounds = ", N_ROUNDS)
println("="^64)
write_meta()
sweep_nmu()
sweep_naz()
sweep_reference()
sweep_grid()
sweep_vsini()
sweep_scaling()
println("\nData written to ", DATADIR)
println("Plot with:  julia --project=. benchmarks/plot_quadrature.jl")
