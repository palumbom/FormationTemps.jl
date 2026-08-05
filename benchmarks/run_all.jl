using Printf

const PROJECT_DIR = dirname(@__DIR__)
const BENCH_DIR   = joinpath(PROJECT_DIR, "benchmarks")
const STATIC_DIR  = joinpath(PROJECT_DIR, "docs", "src", "static")
const max_threads = length(ARGS) >= 1 ? ARGS[1] : string(Sys.CPU_THREADS ÷ 2)

# Spawn the julia that is running this script, not whatever `julia` resolves to on PATH.
# Under juliaup those can be different channels, and the package requires 1.12+.
const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())

# Figures this suite owns, and the script responsible for each. Used for the freshness
# report: a plot script stamps a new mtime on every figure it writes even when the
# underlying data did not regenerate, so a green run is not on its own evidence that a
# figure reflects current code.
const OWNED_FIGURES = [
    ("benchmark_pertile.png",       "benchmark_disk_integration.jl → plot_benchmarks.jl"),
    ("benchmark_convolutions.png",  "benchmark_convolutions.jl → plot_benchmarks.jl"),
    ("benchmark_threading.png",     "benchmark_threading.jl → plot_benchmarks.jl"),
    ("benchmark_nlambda.png",       "benchmark_nlambda.jl → plot_benchmarks.jl"),
    ("benchmark_nphi.png",          "benchmark_nphi.jl → plot_benchmarks.jl"),
    ("quadrature_convergence.png",  "benchmark_quadrature.jl → plot_quadrature.jl"),
    ("quadrature_grid.png",         "benchmark_quadrature.jl → plot_quadrature.jl"),
    ("quadrature_scaling.png",      "benchmark_quadrature.jl → plot_quadrature.jl"),
    ("quadrature_vsini.png",        "benchmark_quadrature.jl → plot_quadrature.jl"),
    ("disk_int_convergence.png",    "disk_int_error.jl"),
    ("gpu_precision_convolve.png",  "gpu_precision_comparison.jl"),
    ("gpu_precision_diskint.png",   "gpu_precision_comparison.jl"),
]

function run_script(path; args=String[], threads=nothing)
    name = basename(path)
    println()
    println("=" ^ 70)
    println("  Running: $name")
    println("=" ^ 70)
    println()

    cmd_parts = [JULIA, "--startup-file=no", "--project=$PROJECT_DIR"]
    if threads !== nothing
        push!(cmd_parts, "-t", string(threads))
    end
    push!(cmd_parts, path)
    append!(cmd_parts, args)

    # ignorestatus: without it `run` throws on a nonzero exit and the first failing script
    # takes the whole suite down, leaving every later script unrun and this function's
    # failure branch unreachable.
    local proc
    t = @elapsed begin
        proc = run(ignorestatus(Cmd(cmd_parts)), wait=true)
    end

    if proc.exitcode != 0
        printstyled("  FAILED: $name (exit code $(proc.exitcode))\n", color=:red)
        return false
    end
    @printf("  Completed: %s (%.1f s)\n", name, t)
    return true
end

# ── single-threaded benchmarks (these time per-tile / per-kernel work, so extra
#    threads would change what is being measured; -t 1 is explicit so an inherited
#    JULIA_NUM_THREADS cannot silently change it) ──────────────────────────────
scripts_single = [
    joinpath(BENCH_DIR, "benchmark_convolutions.jl"),
    joinpath(BENCH_DIR, "benchmark_disk_integration.jl"),
    joinpath(BENCH_DIR, "benchmark_memory.jl"),      # console report only, no artifacts
]

# ── multithreaded benchmarks (read no ARGS; need threads from the parent) ─────
scripts_threaded = [
    joinpath(BENCH_DIR, "benchmark_quadrature.jl"),      # quadrature_*.png
    joinpath(BENCH_DIR, "disk_int_error.jl"),            # disk_int_convergence.png
    joinpath(BENCH_DIR, "gpu_precision_comparison.jl"),  # gpu_precision_*.png
]

# ── subprocess benchmarks (set their own thread counts) ───────────────────────
scripts_subprocess = [
    (joinpath(BENCH_DIR, "benchmark_threading.jl"), [max_threads]),
    (joinpath(BENCH_DIR, "benchmark_nlambda.jl"),   [max_threads]),
    (joinpath(BENCH_DIR, "benchmark_nphi.jl"),      [max_threads]),
]

println("=" ^ 70)
println("  FormationTemps.jl — Full Benchmark Suite")
println("  julia       = $JULIA")
println("  max_threads = $max_threads")
println("=" ^ 70)

t_start = time()
failed = String[]

for path in scripts_single
    run_script(path; threads=1) || push!(failed, basename(path))
end

for path in scripts_threaded
    run_script(path; threads=max_threads) || push!(failed, basename(path))
end

for (path, args) in scripts_subprocess
    run_script(path; args=args) || push!(failed, basename(path))
end

# generate plots
println()
println("=" ^ 70)
println("  Generating plots")
println("=" ^ 70)
run_script(joinpath(BENCH_DIR, "plot_benchmarks.jl")) || push!(failed, "plot_benchmarks.jl")
run_script(joinpath(BENCH_DIR, "plot_quadrature.jl")) || push!(failed, "plot_quadrature.jl")

# ── freshness report ─────────────────────────────────────────────────────────
println()
println("=" ^ 70)
println("  Figure freshness")
println("=" ^ 70)
stale = String[]
for (fig, owner) in OWNED_FIGURES
    p = joinpath(STATIC_DIR, fig)
    if !isfile(p)
        printstyled(@sprintf("  MISSING  %-30s %s\n", fig, owner), color=:red)
        push!(stale, fig)
    elseif mtime(p) < t_start
        printstyled(@sprintf("  STALE    %-30s %s\n", fig, owner), color=:yellow)
        push!(stale, fig)
    else
        @printf("  fresh    %-30s\n", fig)
    end
end

# summary
println()
println("=" ^ 70)
if isempty(failed)
    printstyled("  All benchmarks completed successfully.\n", color=:green)
else
    printstyled("  Failed: " * join(failed, ", ") * "\n", color=:red)
end
if !isempty(stale)
    printstyled("  Not regenerated by this run: " * join(stale, ", ") * "\n", color=:yellow)
    printstyled("  Those figures still show older data; do not read their timestamps as provenance.\n",
                color=:yellow)
end
println("=" ^ 70)

exit(isempty(failed) && isempty(stale) ? 0 : 1)
