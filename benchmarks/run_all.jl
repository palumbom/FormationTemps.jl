using Printf

const PROJECT_DIR = dirname(@__DIR__)
const BENCH_DIR   = joinpath(PROJECT_DIR, "benchmarks")
const max_threads = length(ARGS) >= 1 ? ARGS[1] : string(Sys.CPU_THREADS ÷ 2)

function run_script(path; args=String[], threads=nothing)
    name = basename(path)
    println()
    println("=" ^ 70)
    println("  Running: $name")
    println("=" ^ 70)
    println()

    cmd_parts = ["julia", "--startup-file=no", "--project=$PROJECT_DIR"]
    if threads !== nothing
        push!(cmd_parts, "-t", string(threads))
    end
    push!(cmd_parts, path)
    append!(cmd_parts, args)

    t = @elapsed begin
        proc = run(Cmd(cmd_parts), wait=true)
    end

    if proc.exitcode != 0
        printstyled("  FAILED: $name (exit code $(proc.exitcode))\n", color=:red)
        return false
    end
    @printf("  Completed: %s (%.1f s)\n", name, t)
    return true
end

# ── single-threaded benchmarks (these time per-tile / per-kernel work, so extra
#    threads would change what is being measured) ─────────────────────────────
scripts_inprocess = [
    joinpath(BENCH_DIR, "benchmark_convolutions.jl"),
    joinpath(BENCH_DIR, "benchmark_disk_integration.jl"),
    joinpath(BENCH_DIR, "benchmark_memory.jl"),
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
println("  max_threads = $max_threads")
println("=" ^ 70)

failed = String[]

# run in-process benchmarks as subprocesses too (clean state, avoid Revise issues)
for path in scripts_inprocess
    ok = run_script(path)
    ok || push!(failed, basename(path))
end

for path in scripts_threaded
    ok = run_script(path; threads=max_threads)
    ok || push!(failed, basename(path))
end

for (path, args) in scripts_subprocess
    ok = run_script(path; args=args)
    ok || push!(failed, basename(path))
end

# generate plots
println()
println("=" ^ 70)
println("  Generating plots")
println("=" ^ 70)
run_script(joinpath(BENCH_DIR, "plot_benchmarks.jl")) || push!(failed, "plot_benchmarks.jl")
run_script(joinpath(BENCH_DIR, "plot_quadrature.jl")) || push!(failed, "plot_quadrature.jl")

# summary
println()
println("=" ^ 70)
if isempty(failed)
    printstyled("  All benchmarks completed successfully.\n", color=:green)
else
    printstyled("  Failed: ", join(failed, ", "), "\n", color=:red)
end
println("=" ^ 70)
