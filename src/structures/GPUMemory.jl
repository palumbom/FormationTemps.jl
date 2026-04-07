"""
    GPUMemory{T<:AbstractFloat}

Pre-allocated GPU working arrays for the radiative transfer computation.

Fields:
- `λs`: Wavelength grid on the device (Å), length `Nλ`.
- `αs`, `τs`: Absorption coefficient and optical depth arrays, shape `(Natm, Nλ)`.
- `cfunc`, `cfunc_dt`: Contribution function and dτ-weighted variant, shape `(Natm-1, Nλ)`.
- `tau_ds`, `tau_alphaC`: Bézier τ-integration geometry work arrays; used when `use_anchored=false`.
- `log_τ_ref`, `ifactor_base`: Anchored τ-integration constants (`log τ_ref` and `τ_ref / α_ref`);
  populated and used when `use_anchored=true`.
- `use_anchored`: `true` for the anchored d(log τ) integrator; `false` for the Bézier integrator.
- `v_los_zeros`: Pre-allocated zero vector (length `Natm`) for stationary-frame flux calculations.
  Used instead of destructively zeroing `atm.v_los`.

See also: [`GPUMemory(λs, atm)`](@ref), [`GPUMemory(λs, atm, α_ref)`](@ref)
"""
struct GPUMemory{T<:AF}
    λs::CA{T,1}
    αs::CA{T,2}
    τs::CA{T,2}
    cfunc::CA{T,2}
    cfunc_dt::CA{T,2}      # cfunc .* Δτ, pre-allocated to avoid per-tile allocation
    # Bezier work arrays (used when use_anchored=false)
    tau_ds::CA{T,1}
    tau_alphaC::CA{T,1}
    # Anchored τ constants (used when use_anchored=true)
    log_τ_ref::CA{T,1}     # log(τ_ref), constant across tiles
    ifactor_base::CA{T,1}  # τ_ref / α_ref; divided by μ_i per-tile in kernel
    use_anchored::Bool
    # stationary-frame flux path (avoids destructive fill! on atm.v_los)
    v_los_zeros::CA{T,1}
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
    cfunc_dt   = CUDA.zeros(T, Natm - 1, Nλ)
    tau_ds     = CUDA.zeros(T, Natm - 1)
    tau_alphaC = CUDA.zeros(T, Natm)
    log_τ_ref    = CUDA.zeros(T, Natm)
    ifactor_base = CUDA.zeros(T, Natm)

    v_los_zeros = CUDA.zeros(T, Natm)

    return GPUMemory(λs, αs, τs, cfunc, cfunc_dt, tau_ds, tau_alphaC,
                     log_τ_ref, ifactor_base, false, v_los_zeros)
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
    cfunc_dt     = CUDA.zeros(T, Natm - 1, Nλ)
    tau_ds       = CUDA.zeros(T, Natm - 1)
    tau_alphaC   = CUDA.zeros(T, Natm)
    log_τ_ref    = CuArray{T}(log.(atm.τs))
    ifactor_base = CuArray{T}(atm.τs ./ α_ref_cpu)

    v_los_zeros = CUDA.zeros(T, Natm)

    return GPUMemory(λs, αs, τs, cfunc, cfunc_dt, tau_ds, tau_alphaC,
                     log_τ_ref, ifactor_base, true, v_los_zeros)
end
