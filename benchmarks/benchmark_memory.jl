#!/usr/bin/env julia
# Measure actual GPU memory costs of current structs and project Phase 3 batched costs.
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Printf

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
    @printf("  %-40s  %8.2f MB\n", label, delta / 1024^2)
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
wls = [l.wl * 1e8 for l in linelist]

Npad = 512

# ── GPU info ───────────────────────────────────────────────────────────────────
dev = CUDA.device()
total_mem = CUDA.totalmem(dev)
@printf("GPU: %s  (%.2f GB total)\n\n", CUDA.name(dev), total_mem / 1024^3)

# ── sweep over representative Δλ values ────────────────────────────────────────
Δλ_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

println("=" ^ 90)
@printf("%-6s  %6s  %6s  %10s  %10s  %10s  %10s  %10s\n",
        "Δλ", "Nλ", "Natm", "GPUMemory", "ConvMem", "ConvMac", "cuFFT L",
        "Total/now")
println("-" ^ 90)

for Δλ in Δλ_values
    λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
    Nλ = length(λs_korg)

    atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
    Natm = length(atm_gpu.zs)

    # compute α_ref for anchored tau
    αs = zeros(Natm, Nλ)
    αs_cont = zeros(Natm, Nλ)
    α_ref = zeros(Natm)
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X;
                      α_ref_out=α_ref, vmic_ref_cms=ξ * 100.0, ne_warn_thresh=Inf)

    # measure GPUMemory
    _, mem_gpu = measure_alloc("GPUMemory (Δλ=$Δλ)") do
        FT.GPUMemory(λs_korg, atm_gpu, α_ref)
    end

    # measure ConvolutionMemory (micro)
    cmem_ref, mem_cmem = measure_alloc("ConvMem micro (Δλ=$Δλ)") do
        FT.ConvolutionMemory(Nλ, Natm, Npad)
    end

    # measure MacroConvolutionMemory
    _, mem_cmac = measure_alloc("MacroConvMem (Δλ=$Δλ)") do
        FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)
    end

    L = cmem_ref.L
    total_current = mem_gpu + mem_cmem + mem_cmac

    @printf("%-6.3f  %6d  %6d  %8.2f MB  %8.2f MB  %8.2f MB  %10d  %8.2f MB\n",
            Δλ, Nλ, Natm, mem_gpu/1024^2, mem_cmem/1024^2, mem_cmac/1024^2,
            L, total_current/1024^2)

    # free everything
    gpu_used_bytes()  # GC + reclaim side-effect
end

# ── Phase 3 projections ───────────────────────────────────────────────────────
println("\n" * "=" ^ 90)
println("Phase 3 projections: batched ConvolutionMemory (B*Natm rows)")
println("=" ^ 90)

# use a representative case
Δλ_ref = 0.01
λs_ref = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ_ref)
Nλ_ref = length(λs_ref)
atm_ref = FT.AtmosphereGPU(Korg.interpolate_marcs(Teff, logg, A_X))
Natm_ref = length(atm_ref.zs)

# measure actual ConvolutionMemory at increasing Natm to capture cuFFT plan scaling
@printf("\n%-6s  %8s  %10s  %12s  %12s\n",
        "B", "B*Natm", "ConvMem", "cuFFT plan", "Total batch")
println("-" ^ 60)

for B in [1, 4, 8, 16, 32, 64]
    Natm_batch = B * Natm_ref
    _, mem_batch = measure_alloc("ConvMem B=$B") do
        FT.ConvolutionMemory(Nλ_ref, Natm_batch, Npad)
    end
    gpu_used_bytes()  # GC + reclaim side-effect

    # BatchedGPUMemory arrays (manual estimate: αs + τs + cfunc_dt, all (B,Natm,Nλ))
    bytes_per_elem = sizeof(Float64)
    mem_αs = B * Natm_ref * Nλ_ref * bytes_per_elem
    mem_τs = B * Natm_ref * Nλ_ref * bytes_per_elem
    mem_cfdt = B * (Natm_ref - 1) * Nλ_ref * bytes_per_elem
    mem_arrays = mem_αs + mem_τs + mem_cfdt

    total = mem_batch + mem_arrays

    @printf("%-6d  %8d  %8.2f MB  %10s  %10.2f MB\n",
            B, Natm_batch, mem_batch/1024^2, "—", total/1024^2)
end

# ── free memory budget ─────────────────────────────────────────────────────────
gpu_used_bytes()  # GC + reclaim
free_mem = CUDA.available_memory()
@printf("\nFree GPU memory:  %.2f GB / %.2f GB\n",
        free_mem / 1024^3, total_mem / 1024^3)

# ── per-Δλ batched projections ─────────────────────────────────────────────────
println("\n" * "=" ^ 90)
println("Full per-Δλ projections (B=32, including 2x for dual-stream total+cont)")
println("=" ^ 90)
B = 32

@printf("\n%-6s  %6s  %12s  %12s  %12s  %10s\n",
        "Δλ", "Nλ", "2x BatchMem", "2x ConvMem", "Total", "% of GPU")
println("-" ^ 70)

for Δλ in Δλ_values
    λs = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)
    Nλ = length(λs)
    Natm = Natm_ref
    Natm_batch = B * Natm

    # array costs (x2 for total + continuum)
    bpe = sizeof(Float64)
    mem_arrays = 2 * (B * Natm * Nλ * bpe + B * Natm * Nλ * bpe + B * (Natm-1) * Nλ * bpe)

    # ConvMem costs (x2 for total + continuum micro; +2 for macro)
    # estimate from L scaling: measure one and scale
    _, mem_cmem_one = measure_alloc("proj ConvMem B=$B Δλ=$Δλ") do
        FT.ConvolutionMemory(Nλ, Natm_batch, Npad)
    end
    gpu_used_bytes()  # GC + reclaim side-effect

    _, mem_cmac_one = measure_alloc("proj ConvMac B=$B Δλ=$Δλ") do
        FT.ConvolutionMemory(Nλ, B * (Natm - 1), Npad)
    end
    gpu_used_bytes()  # GC + reclaim side-effect

    mem_conv_total = 2 * mem_cmem_one + 2 * mem_cmac_one

    total = mem_arrays + mem_conv_total
    pct = 100.0 * total / total_mem

    @printf("%-6.3f  %6d  %10.1f MB  %10.1f MB  %10.1f MB  %8.1f%%\n",
            Δλ, Nλ, mem_arrays/1024^2, mem_conv_total/1024^2, total/1024^2, pct)
end
