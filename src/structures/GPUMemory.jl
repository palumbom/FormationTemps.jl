struct GPUMemory{T<:AF}
    λs::CA{T,1}
    αs::CA{T,2}
    τs::CA{T,2}
    cfunc::CA{T,2}
    flux::CA{T,2}
    # Bezier work arrays (used when use_anchored=false)
    tau_ds::CA{T,1}
    tau_alphaC::CA{T,1}
    # Anchored τ constants (used when use_anchored=true)
    log_τ_ref::CA{T,1}     # log(τ_ref), constant across tiles
    ifactor_base::CA{T,1}  # τ_ref / α_ref; divided by μ_i per-tile in kernel
    use_anchored::Bool
end

"""
    GPUMemory(λs_cpu, atm)

Allocate GPU working memory using the Bezier τ integrator (backward-compatible default).
"""
function GPUMemory(λs_cpu::AA{T,1}, atm::AtmosphereGPU) where T
    Nλ   = length(λs_cpu)
    Natm = length(atm.zs)

    λs         = CuArray{T}(λs_cpu)
    αs         = CUDA.zeros(T, Natm, Nλ)
    τs         = CUDA.zeros(T, Natm, Nλ)
    cfunc      = CUDA.zeros(T, Natm - 1, Nλ)
    flux       = CUDA.zeros(T, Natm - 1, Nλ)
    tau_ds     = CUDA.zeros(T, Natm - 1)
    tau_alphaC = CUDA.zeros(T, Natm)
    log_τ_ref    = CUDA.zeros(T, Natm)
    ifactor_base = CUDA.zeros(T, Natm)

    CUDA.synchronize()
    return GPUMemory(λs, αs, τs, cfunc, flux, tau_ds, tau_alphaC,
                     log_τ_ref, ifactor_base, false)
end

"""
    GPUMemory(λs_cpu, atm, α_ref_cpu)

Allocate GPU working memory using the anchored τ integrator.  `α_ref_cpu` is the
continuum absorption coefficient at the MARCS reference wavelength for each layer —
`αs_cont[:, end]` from a prior `compute_alpha!` call.
"""
function GPUMemory(λs_cpu::AA{T,1}, atm::AtmosphereGPU, α_ref_cpu::AA{T,1}) where T
    Nλ   = length(λs_cpu)
    Natm = length(atm.zs)

    λs           = CuArray{T}(λs_cpu)
    αs           = CUDA.zeros(T, Natm, Nλ)
    τs           = CUDA.zeros(T, Natm, Nλ)
    cfunc        = CUDA.zeros(T, Natm - 1, Nλ)
    flux         = CUDA.zeros(T, Natm - 1, Nλ)
    tau_ds       = CUDA.zeros(T, Natm - 1)
    tau_alphaC   = CUDA.zeros(T, Natm)
    log_τ_ref    = CuArray{T}(log.(atm.τs))
    ifactor_base = CuArray{T}(atm.τs ./ α_ref_cpu)

    CUDA.synchronize()
    return GPUMemory(λs, αs, τs, cfunc, flux, tau_ds, tau_alphaC,
                     log_τ_ref, ifactor_base, true)
end
