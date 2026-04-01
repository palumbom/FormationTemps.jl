using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Printf, Statistics

# ── helpers ────────────────────────────────────────────────────────────────────
function gpu_used_bytes()
    GC.gc()
    CUDA.reclaim()
    CUDA.synchronize()
    return CUDA.used_memory()
end

function measure_alloc(f, label)
    before = gpu_used_bytes()
    result = f()
    CUDA.synchronize()
    after = gpu_used_bytes()
    delta = after - before
    @printf("  %-45s  %8.2f MB\n", label, delta / 1024^2)
    return result, delta
end

# ── setup ──────────────────────────────────────────────────────────────────────
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
ξ = 850.0
A_X = Korg.format_A_X(Fe_H)

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
wls = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]

Npad = 512

# ── GPU info ───────────────────────────────────────────────────────────────────
dev = CUDA.device()
total_mem = CUDA.totalmem(dev)
@printf("GPU: %s  (%.2f GB total)\n\n", CUDA.name(dev), total_mem / 1024^3)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Per-Δλ sweep: base structs at Float64
# ══════════════════════════════════════════════════════════════════════════════
Δλ_values = [0.1, 0.05, 0.02, 0.01, 0.005]

println("=" ^ 90)
println("Base struct allocations (Float64)")
println("-" ^ 90)
@printf("%-6s  %6s  %6s  %10s  %10s  %10s  %10s\n",
        "Δλ", "Nλ", "Natm", "GPUMemory", "ConvMem", "ConvMac", "Total")
println("-" ^ 90)

for Δλ in Δλ_values
    λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
    Nλ = length(λs_korg)

    atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
    Natm = length(atm_gpu.zs)

    αs = zeros(Natm, Nλ)
    αs_cont = zeros(Natm, Nλ)
    α_ref = zeros(Natm)
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                      α_ref_out=α_ref, ne_warn_thresh=Inf)

    _, mem_gpu = measure_alloc("GPUMemory (Δλ=$Δλ)") do
        FT.GPUMemory(collect(λs_korg), atm_gpu, α_ref)
    end

    cmem_ref, mem_cmem = measure_alloc("ConvMem (Δλ=$Δλ)") do
        FT.ConvolutionMemory(Nλ, Natm, Npad)
    end

    _, mem_cmac = measure_alloc("MacroConvMem (Δλ=$Δλ)") do
        FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)
    end

    total_current = mem_gpu + mem_cmem + mem_cmac

    @printf("%-6.3f  %6d  %6d  %8.2f MB  %8.2f MB  %8.2f MB  %8.2f MB\n",
            Δλ, Nλ, Natm, mem_gpu/1024^2, mem_cmem/1024^2, mem_cmac/1024^2,
            total_current/1024^2)

    gpu_used_bytes()
end

# ══════════════════════════════════════════════════════════════════════════════
# 2. BatchedMicroConvMem: actual allocations at varying B
# ══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 90)
println("BatchedMicroConvMem allocations (Float64, Δλ=0.01)")
println("=" ^ 90)

Δλ_ref = 0.01
λs_ref = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ_ref)
Nλ_ref = length(λs_ref)
atm_ref = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm_ref = length(atm_ref.zs)
Natm1_ref = Natm_ref - 1

@printf("\n%-6s  %8s  %12s  %12s  %12s\n",
        "B", "B*Natm", "BatchMicro", "Work arrays", "Total")
println("-" ^ 60)

for B in [1, 4, 8, 16, 32, 64]
    _, mem_bcmem = measure_alloc("BatchedMicroConvMem B=$B") do
        FT.BatchedMicroConvMem(Nλ_ref, Natm_ref, B, Npad)
    end

    # work arrays: τs_batch + cfdt_batch (what convenience.jl allocates per stream)
    bpe = sizeof(Float64)
    mem_work = B * Natm_ref * Nλ_ref * bpe +      # τs_batch
               B * Natm1_ref * Nλ_ref * bpe        # cfdt_batch

    total = mem_bcmem + mem_work

    @printf("%-6d  %8d  %10.2f MB  %10.2f MB  %10.2f MB\n",
            B, B * Natm_ref, mem_bcmem/1024^2, mem_work/1024^2, total/1024^2)
    gpu_used_bytes()
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Float32 vs Float64 comparison
# ══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 90)
println("Float32 vs Float64 memory comparison (Δλ=0.01, B=16)")
println("=" ^ 90)

B_cmp = 16

for (label, G) in [("Float64", Float64), ("Float32", Float32)]
    println("\n  $label:")

    _, mem_bcmem = measure_alloc("  BatchedMicroConvMem") do
        FT.BatchedMicroConvMem(Nλ_ref, Natm_ref, B_cmp, Npad; T=G)
    end

    _, mem_cmem = measure_alloc("  ConvolutionMemory") do
        FT.ConvolutionMemory(Nλ_ref, Natm_ref, Npad; T=G)
    end

    _, mem_cmac = measure_alloc("  MacroConvolutionMemory") do
        FT.MacroConvolutionMemory(Nλ_ref, Natm1_ref, Npad; T=G)
    end

    bpe = sizeof(G)
    mem_work = B_cmp * Natm_ref * Nλ_ref * bpe +
               B_cmp * Natm1_ref * Nλ_ref * bpe

    total = mem_bcmem + mem_cmem + mem_cmac + mem_work
    @printf("    Work arrays (τs+cfdt):            %8.2f MB\n", mem_work/1024^2)
    @printf("    Total (1 stream):                 %8.2f MB\n", total/1024^2)
    @printf("    Total (2 streams, dual):          %8.2f MB\n", (2*total)/1024^2)
    gpu_used_bytes()
end

# ── free memory budget ─────────────────────────────────────────────────────────
gpu_used_bytes()
free_mem = CUDA.available_memory()
@printf("\nFree GPU memory:  %.2f GB / %.2f GB\n",
        free_mem / 1024^3, total_mem / 1024^3)
