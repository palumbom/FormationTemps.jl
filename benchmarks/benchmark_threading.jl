"""
CPU threading scaling benchmark for disk integration.

Measures end-to-end wall-clock time for `calc_formation_temp` with `convolve=false`
(numerical disk integration) as a function of the number of Julia threads. Since
Julia's thread count is set at startup, this script spawns separate Julia processes
for each thread count and collects timings.

Usage:
    julia --project=. benchmarks/benchmark_threading.jl [max_threads]

    max_threads  — upper bound on thread counts to test (default: number of physical cores).

Output:
    benchmarks/data/threading_scaling.csv
    docs/src/static/benchmark_threading.png  (if PythonPlot available)
"""

using Printf, Statistics, DelimitedFiles

# ── configuration ─────────────────────────────────────────────────────────────
const PROJECT_DIR = dirname(@__DIR__)
const DATADIR = joinpath(PROJECT_DIR, "benchmarks", "data")
!isdir(DATADIR) && mkpath(DATADIR)

# thread counts to benchmark
n_physical = Sys.CPU_THREADS ÷ 2  # assume hyperthreading
max_threads = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : n_physical
thread_counts = unique(sort([1; 2; 4; 8; filter(t -> t <= max_threads, [12, 16, 24, 32, 48, 64])]))
thread_counts = filter(t -> t <= max_threads, thread_counts)

# benchmark parameters
Nϕ = 128
Δλ = 0.01
n_repeat = 8

println("="^60)
println("THREADING SCALING BENCHMARK")
println("="^60)
println("  Max threads: ", max_threads)
println("  Thread counts: ", thread_counts)
println("  Nϕ = ", Nϕ, ", Δλ = ", Δλ, ", repeats = ", n_repeat)
println()

# ── worker script (written to tempfile, run by each subprocess) ──────────────
worker_code = """
using FormationTemps; FT = FormationTemps
using Korg, Statistics

linelist_full = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist_full = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist_full]
specs = [string(l.species) for l in linelist_full]
linelist_fe = linelist_full[specs .== "Fe I"]
wls_all = [l.wl * 1e8 for l in linelist_fe]
idx_start = findfirst(x -> x >= 6298.0, wls_all)
idx_end = findfirst(x -> x >= 6304.0, wls_all)
linelist = linelist_fe[idx_start:idx_end]

star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0,
                    v_macro=3500.0, v_micro=850.0)

Nphi = parse(Int, ARGS[1])
dlambda = parse(Float64, ARGS[2])
n_repeat = parse(Int, ARGS[3])

# warmup
calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=16,
                    use_gpu=false, showprogress=false, ne_warn_thresh=Inf)

times = zeros(n_repeat)
for r in 1:n_repeat
    times[r] = @elapsed calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=Nphi,
                                             use_gpu=false, showprogress=false,
                                             ne_warn_thresh=Inf)
end

println("RESULT:", Threads.nthreads(), ",", median(times), ",", minimum(times), ",", maximum(times))
"""

worker_file = tempname() * ".jl"
write(worker_file, worker_code)

# ── run benchmarks ────────────────────────────────────────────────────────────
results = Dict{Int, NamedTuple{(:median_s, :min_s, :max_s), Tuple{Float64, Float64, Float64}}}()

for nt in thread_counts
    print(@sprintf("  %2d threads: ", nt))
    flush(stdout)

    cmd = pipeline(`julia --project=$PROJECT_DIR -t $nt $worker_file $Nϕ $Δλ $n_repeat`,
                   stderr=devnull)
    output = read(cmd, String)

    # parse RESULT line
    for line in split(output, '\n')
        if startswith(line, "RESULT:")
            parts = split(line[8:end], ',')
            actual_nt = parse(Int, parts[1])
            med = parse(Float64, parts[2])
            mn = parse(Float64, parts[3])
            mx = parse(Float64, parts[4])
            results[nt] = (median_s=med, min_s=mn, max_s=mx)
            @printf("%.2f s (min %.2f, max %.2f)\n", med, mn, mx)
        end
    end

    if !haskey(results, nt)
        println("FAILED — no result line found")
        println("  Output: ", first(output, 500))
    end
end

rm(worker_file, force=true)

# ── compute scaling metrics ───────────────────────────────────────────────────
println()
if haskey(results, 1)
    t1 = results[1].median_s
    println("Scaling relative to 1 thread ($(round(t1, digits=2)) s):")
    for nt in sort(collect(keys(results)))
        t = results[nt].median_s
        speedup = t1 / t
        efficiency = speedup / nt * 100
        @printf("  %2d threads: %.2f s  (%.1f× speedup, %.0f%% efficiency)\n",
                nt, t, speedup, efficiency)
    end
end

# ── save data ─────────────────────────────────────────────────────────────────
open(joinpath(DATADIR, "threading_scaling.csv"), "w") do io
    println(io, "threads,median_s,min_s,max_s,speedup,efficiency_pct")
    t1 = haskey(results, 1) ? results[1].median_s : NaN
    for nt in sort(collect(keys(results)))
        r = results[nt]
        speedup = t1 / r.median_s
        efficiency = speedup / nt * 100
        @printf(io, "%d,%.4f,%.4f,%.4f,%.2f,%.1f\n",
                nt, r.median_s, r.min_s, r.max_s, speedup, efficiency)
    end
end
println("\nData written to: ", joinpath(DATADIR, "threading_scaling.csv"))

println()
println("="^60)
println("DONE")
println("="^60)
