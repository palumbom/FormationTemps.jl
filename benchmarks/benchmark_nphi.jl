using Printf, Statistics

# ── configuration ─────────────────────────────────────────────────────────────
const PROJECT_DIR = dirname(@__DIR__)
const DATADIR = joinpath(PROJECT_DIR, "benchmarks", "data")
!isdir(DATADIR) && mkpath(DATADIR)

n_physical = Sys.CPU_THREADS ÷ 2
max_threads = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : n_physical

Nϕ_values = [16, 32, 64, 128, 256]
Δλ = 0.0025
n_repeat = 8

println("="^60)
println("Nϕ SCALING BENCHMARK")
println("="^60)
cpu_thread_counts = unique(sort([1; max_threads > 8 ? [8, max_threads] : [max_threads]]))

println("  Nϕ values:   ", Nϕ_values)
println("  Δλ = ", Δλ, ", repeats = ", n_repeat)
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
idx1 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0,
                    v_macro=3500.0, v_micro=850.0)

Nphi = parse(Int, ARGS[1])
dlambda = parse(Float64, ARGS[2])
n_repeat = parse(Int, ARGS[3])

# warmup at target Nϕ to cover FFTW plan creation and JIT
calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=Nphi,
                    use_gpu=false, showprogress=false, ne_warn_thresh=Inf)

# time compute_alpha! separately (fixed cost independent of Nϕ)
wls_ang = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
λs_korg = range(first(wls_ang) - 2.0, last(wls_ang) + 2.0, step=dlambda)
atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(star.Teff, star.logg, star.A_X))
Natm = length(atm_cpu.zs)
Nλ = length(λs_korg)
αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
alpha_time = @elapsed FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg),
                                        linelist, atm_cpu, star.A_X;
                                        α_ref_out=α_ref, ne_warn_thresh=Inf)

# count tiles for this Nϕ
μs, dA, z_rot = FT.calc_stellar_grid_cpu(star.ρstar, star.istar, star.vsini, Nphi)
Ntiles = count(x -> x > 0.0, μs)

times = zeros(n_repeat)
for r in 1:n_repeat
    times[r] = @elapsed begin
        calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=Nphi,
                             use_gpu=false, showprogress=false,
                             ne_warn_thresh=Inf)
    end
end

println("RESULT:", Threads.nthreads(), ",", Nphi, ",", Nλ, ",", Ntiles, ",",
        median(times), ",", minimum(times), ",", maximum(times), ",", alpha_time)
"""

# ── worker script (GPU) ──────────────────────────────────────────────────────
gpu_worker_code = """
using FormationTemps; FT = FormationTemps
using Korg, Statistics, CUDA

if !CUDA.functional()
    println("RESULT:gpu_" * ARGS[4] * ",", ARGS[1], ",0,0,NaN,NaN,NaN,NaN")
    exit()
end

# Fe I 6301 & 6302 lines
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2 = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0,
                    v_macro=3500.0, v_micro=850.0)

Nphi = parse(Int, ARGS[1])
dlambda = parse(Float64, ARGS[2])
n_repeat = parse(Int, ARGS[3])
precision_str = ARGS[4]  # "float64" or "float32"
gpu_prec = precision_str == "float32" ? Float32 : Float64

# warmup at target Nϕ to cover cuFFT plan creation and JIT
calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=Nphi,
                    use_gpu=true, gpu_precision=gpu_prec,
                    showprogress=false, ne_warn_thresh=Inf)
CUDA.synchronize()

# time compute_alpha! separately (fixed cost independent of Nϕ)
wls_ang = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
λs_korg = range(first(wls_ang) - 2.0, last(wls_ang) + 2.0, step=dlambda)
korg_atm = Korg.interpolate_marcs(star.Teff, star.logg, star.A_X)
atm_f64 = FT.AtmosphereGPU(korg_atm; T=Float64)
Natm = length(atm_f64.zs)
Nλ = length(λs_korg)
αs = zeros(Natm, Nλ)
αs_cont = zeros(Natm, Nλ)
α_ref = zeros(Natm)
alpha_time = @elapsed FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg),
                                        linelist, atm_f64, star.A_X;
                                        α_ref_out=α_ref, ne_warn_thresh=Inf)

# count tiles for this Nϕ
μs, dA, z_rot = FT.calc_stellar_grid_cpu(star.ρstar, star.istar, star.vsini, Nphi)
Ntiles = count(x -> x > 0.0, μs)

times = zeros(n_repeat)
for r in 1:n_repeat
    CUDA.synchronize()
    times[r] = @elapsed begin
        res = calc_formation_temp(star, linelist; Δλ=dlambda, Nϕ=Nphi,
                                   use_gpu=true, gpu_precision=gpu_prec,
                                   showprogress=false, ne_warn_thresh=Inf)
    end
    CUDA.synchronize()
end

println("RESULT:gpu_", precision_str, ",", Nphi, ",", Nλ, ",", Ntiles, ",",
        median(times), ",", minimum(times), ",", maximum(times), ",", alpha_time)
"""

cpu_file = tempname() * "_cpu_nphi.jl"
gpu_file = tempname() * "_gpu_nphi.jl"
write(cpu_file, cpu_worker_code)
write(gpu_file, gpu_worker_code)

# ── helpers ───────────────────────────────────────────────────────────────────
function parse_result(output)
    for line in split(output, '\n')
        if startswith(line, "RESULT:")
            parts = split(line[8:end], ',')
            return (threads=parts[1],
                    Nphi=parse(Int, parts[2]),
                    Nlambda=parse(Int, parts[3]),
                    Ntiles=parse(Int, parts[4]),
                    median_s=parse(Float64, parts[5]),
                    min_s=parse(Float64, parts[6]),
                    max_s=parse(Float64, parts[7]),
                    alpha_s=parse(Float64, parts[8]))
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
# columns: backend, threads, Nphi, Nlambda, Ntiles, median_s, min_s, max_s, alpha_s
results = NamedTuple{(:backend, :threads, :Nphi, :Nlambda, :Ntiles, :median_s, :min_s, :max_s, :alpha_s),
                      Tuple{String, Int, Int, Int, Int, Float64, Float64, Float64, Float64}}[]

for Nϕ in Nϕ_values
    # CPU runs at each thread count
    for nt in cpu_thread_counts
        print(@sprintf("  CPU %2dT  Nϕ=%3d: ", nt, Nϕ)); flush(stdout)
        output, errmsg = run_worker(`julia --startup-file=no --project=$PROJECT_DIR -t $nt $cpu_file $Nϕ $Δλ $n_repeat`)
        if errmsg !== nothing
            println("FAILED")
            !isempty(errmsg) && println("  stderr: ", first(errmsg, 500))
            continue
        end
        r = parse_result(output)
        if r !== nothing
            push!(results, (backend="cpu", threads=nt, Nphi=Nϕ, Nlambda=r.Nlambda,
                            Ntiles=r.Ntiles, median_s=r.median_s, min_s=r.min_s,
                            max_s=r.max_s, alpha_s=r.alpha_s))
            @printf("Ntiles=%5d  %.2f s  (alpha=%.2f s)\n", r.Ntiles, r.median_s, r.alpha_s)
        else
            println("FAILED — no result line")
            println("  stdout: ", first(output, 500))
        end
    end

    # GPU Float64 and Float32
    for prec in ["float64", "float32"]
        label = prec == "float64" ? "GPU64" : "GPU32"
        print(@sprintf("  %-6s  Nϕ=%3d: ", label, Nϕ)); flush(stdout)
        output, errmsg = run_worker(`julia --startup-file=no --project=$PROJECT_DIR $gpu_file $Nϕ $Δλ $n_repeat $prec`)
        if errmsg !== nothing
            println("skipped (error)")
            !isempty(errmsg) && println("  stderr: ", first(errmsg, 500))
        else
            r = parse_result(output)
            if r !== nothing && !isnan(r.median_s)
                push!(results, (backend="gpu_" * prec, threads=1, Nphi=Nϕ, Nlambda=r.Nlambda,
                                Ntiles=r.Ntiles, median_s=r.median_s, min_s=r.min_s,
                                max_s=r.max_s, alpha_s=r.alpha_s))
                @printf("Ntiles=%5d  %.2f s  (alpha=%.2f s)\n", r.Ntiles, r.median_s, r.alpha_s)
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
open(joinpath(DATADIR, "nphi_scaling.csv"), "w") do io
    println(io, "backend,threads,Nphi,Nlambda,Ntiles,median_s,min_s,max_s,alpha_s")
    for r in results
        @printf(io, "%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f\n",
                r.backend, r.threads, r.Nphi, r.Nlambda, r.Ntiles,
                r.median_s, r.min_s, r.max_s, r.alpha_s)
    end
end
println("Data written to: ", joinpath(DATADIR, "nphi_scaling.csv"))

println()
println("="^60)
println("DONE")
println("="^60)
