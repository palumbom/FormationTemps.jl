using Printf, Statistics

# ── configuration ─────────────────────────────────────────────────────────────
const PROJECT_DIR = dirname(@__DIR__)
const DATADIR = joinpath(PROJECT_DIR, "benchmarks", "data")
!isdir(DATADIR) && mkpath(DATADIR)

n_physical = Sys.CPU_THREADS ÷ 2
max_threads = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : n_physical

Δλ_values = [0.1, 0.05, 0.025, 0.01, 0.005]
Nϕ = 128
n_repeat = 8

println("="^60)
println("Nλ SCALING BENCHMARK")
println("="^60)
cpu_thread_counts = unique(sort([1; max_threads > 8 ? [8, max_threads] : [max_threads]]))

println("  Δλ values:   ", Δλ_values)
println("  Nϕ = ", Nϕ, ", repeats = ", n_repeat)
println("  CPU threads: ", cpu_thread_counts)
println()

# ── worker script (CPU) ──────────────────────────────────────────────────────
cpu_worker_code = """
using FormationTemps; FT = FormationTemps
using Korg, Statistics

# Fe I 6301 & 6302 lines
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * 1e8 >= 6301, wls)
idx2 = findfirst(x -> x * 1e8 >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0,
                    v_macro=3500.0, v_micro=850.0)

Nphi = parse(Int, ARGS[1])
dlambda = parse(Float64, ARGS[2])
n_repeat = parse(Int, ARGS[3])

# warmup with coarse grid
calc_formation_temp(star, linelist; Δλ=0.1, Nϕ=16,
                    use_gpu=false, showprogress=false, ne_warn_thresh=Inf)

times = zeros(n_repeat)
Nlambda = 0
for r in 1:n_repeat
    times[r] = @elapsed begin
        res = calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=Nphi,
                                   use_gpu=false, showprogress=false,
                                   ne_warn_thresh=Inf)
        global Nlambda = length(res.wavs)
    end
end

println("RESULT:", Threads.nthreads(), ",", dlambda, ",", Nlambda, ",",
        median(times), ",", minimum(times), ",", maximum(times))
"""

# ── worker script (GPU) ──────────────────────────────────────────────────────
gpu_worker_code = """
using FormationTemps; FT = FormationTemps
using Korg, Statistics, CUDA

if !CUDA.functional()
    println("RESULT:gpu_" * ARGS[4] * ",", ARGS[2], ",0,NaN,NaN,NaN")
    exit()
end

# Fe I 6301 & 6302 lines
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * 1e8 >= 6301, wls)
idx2 = findfirst(x -> x * 1e8 >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0,
                    v_macro=3500.0, v_micro=850.0)

Nphi = parse(Int, ARGS[1])
dlambda = parse(Float64, ARGS[2])
n_repeat = parse(Int, ARGS[3])
precision_str = ARGS[4]  # "float64" or "float32"
gpu_prec = precision_str == "float32" ? Float32 : Float64

# warmup
calc_formation_temp(star, linelist; Δλ=0.1, Nϕ=16,
                    use_gpu=true, gpu_precision=gpu_prec,
                    showprogress=false, ne_warn_thresh=Inf)
CUDA.synchronize()

Nlambda = 0
times = zeros(n_repeat)
for r in 1:n_repeat
    CUDA.synchronize()
    times[r] = @elapsed begin
        res = calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=Nphi,
                                   use_gpu=true, gpu_precision=gpu_prec,
                                   showprogress=false, ne_warn_thresh=Inf)
        global Nlambda = length(res.wavs)
    end
    CUDA.synchronize()
end

println("RESULT:gpu_", precision_str, ",", dlambda, ",", Nlambda, ",",
        median(times), ",", minimum(times), ",", maximum(times))
"""

cpu_file = tempname() * "_cpu.jl"
gpu_file = tempname() * "_gpu.jl"
write(cpu_file, cpu_worker_code)
write(gpu_file, gpu_worker_code)

# ── helpers ───────────────────────────────────────────────────────────────────
function parse_result(output)
    for line in split(output, '\n')
        if startswith(line, "RESULT:")
            parts = split(line[8:end], ',')
            return (threads=parts[1],
                    dlambda=parse(Float64, parts[2]),
                    Nlambda=parse(Int, parts[3]),
                    median_s=parse(Float64, parts[4]),
                    min_s=parse(Float64, parts[5]),
                    max_s=parse(Float64, parts[6]))
        end
    end
    return nothing
end

function run_worker(cmd)
    errfile = tempname()
    try
        output = read(pipeline(cmd, stderr=errfile), String)
        rm(errfile, force=true)
        return output, nothing
    catch e
        errmsg = isfile(errfile) ? read(errfile, String) : ""
        rm(errfile, force=true)
        return "", errmsg
    end
end

# ── run benchmarks ────────────────────────────────────────────────────────────
# columns: backend, threads, dlambda, Nlambda, median_s, min_s, max_s
results = NamedTuple{(:backend, :threads, :dlambda, :Nlambda, :median_s, :min_s, :max_s),
                      Tuple{String, Int, Float64, Int, Float64, Float64, Float64}}[]

for dλ in Δλ_values
    # CPU runs at each thread count
    for nt in cpu_thread_counts
        print(@sprintf("  CPU %2dT  Δλ=%.4f: ", nt, dλ)); flush(stdout)
        output, errmsg = run_worker(`julia --startup-file=no --project=$PROJECT_DIR -t $nt $cpu_file $Nϕ $dλ $n_repeat`)
        if errmsg !== nothing
            println("FAILED")
            !isempty(errmsg) && println("  stderr: ", first(errmsg, 500))
            continue
        end
        r = parse_result(output)
        if r !== nothing
            push!(results, (backend="cpu", threads=nt, dlambda=dλ, Nlambda=r.Nlambda,
                            median_s=r.median_s, min_s=r.min_s, max_s=r.max_s))
            @printf("Nλ=%5d  %.2f s\n", r.Nlambda, r.median_s)
        else
            println("FAILED — no result line")
            println("  stdout: ", first(output, 500))
        end
    end

    # GPU Float64 and Float32
    for prec in ["float64", "float32"]
        label = prec == "float64" ? "GPU64" : "GPU32"
        print(@sprintf("  %-6s  Δλ=%.4f: ", label, dλ)); flush(stdout)
        output, errmsg = run_worker(`julia --startup-file=no --project=$PROJECT_DIR $gpu_file $Nϕ $dλ $n_repeat $prec`)
        if errmsg !== nothing
            println("skipped (error)")
            !isempty(errmsg) && println("  stderr: ", first(errmsg, 500))
        else
            r = parse_result(output)
            if r !== nothing && !isnan(r.median_s)
                push!(results, (backend="gpu_" * prec, threads=1, dlambda=dλ, Nlambda=r.Nlambda,
                                median_s=r.median_s, min_s=r.min_s, max_s=r.max_s))
                @printf("Nλ=%5d  %.2f s\n", r.Nlambda, r.median_s)
            else
                println("skipped (no GPU)")
            end
        end
    end

    println()
end

rm(cpu_file, force=true)
rm(gpu_file, force=true)

# ── save data ─────────────────────────────────────────────────────────────────
open(joinpath(DATADIR, "nlambda_scaling.csv"), "w") do io
    println(io, "backend,threads,dlambda,Nlambda,median_s,min_s,max_s")
    for r in results
        @printf(io, "%s,%d,%.4f,%d,%.4f,%.4f,%.4f\n",
                r.backend, r.threads, r.dlambda, r.Nlambda, r.median_s, r.min_s, r.max_s)
    end
end
println("Data written to: ", joinpath(DATADIR, "nlambda_scaling.csv"))

println()
println("="^60)
println("DONE")
println("="^60)
