"""
    Atmosphere{T}

Abstract atmosphere type used by formation temperature calculations.
"""
abstract type Atmosphere{T<:AF} end

# TODO set up Atmosphere constructor

"""
    get_τs(atm)

Return the optical depth grid as a standard `Array`.
"""
get_τs(atm::Atmosphere) = Array(atm.τs)

"""
    get_zs(atm)

Return the height grid as a standard `Array`.
"""
get_zs(atm::Atmosphere) = Array(atm.zs)

"""
    get_Ts(atm)

Return the temperature grid as a standard `Array`.
"""
get_Ts(atm::Atmosphere) = Array(atm.Ts)

"""
    AtmosphereGPU(atm_korg)

Build a GPU-backed atmosphere from a Korg atmosphere object.
"""
mutable struct AtmosphereGPU{T<:AF} <: Atmosphere{T}
    Natm::Int
    τs::AA{T,1}
    zs::AA{T,1}
    Ts::AA{T,1}
    nₑ::AA{T,1}
    nd::AA{T,1}
    reference_wavelength::T  # cm; MARCS reference wavelength for τ_ref (typically 5000 Å)

    zs_gpu::AA{T,1}
    Ts_gpu::AA{T,1}
    nd_gpu::CA{T,1}
    vx::CA{T,1}
    vy::CA{T,1}
    vz::CA{T,1}
    σ_v::CA{T,1}
    μ_v::CA{T,1}
end

"""
    AtmosphereGPU(atm_korg)

Construct an `AtmosphereGPU` with thermodynamic fields from Korg and velocity
fields allocated on the GPU.
"""
function AtmosphereGPU(atm_korg)
    # resample onto uniform log τ grid
    atm_korg = _resample_log_tau(atm_korg)

    # Korg atmosphere parameters
    τs  = Korg.get_tau_refs(atm_korg)
    zs  = Korg.get_zs(atm_korg)
    Ts  = Korg.get_temps(atm_korg)
    ne  = Korg.get_electron_number_densities(atm_korg)
    nd  = Korg.get_number_densities(atm_korg)
    ref_wl = atm_korg.reference_wavelength  # cm

    # allocate on gpu
    Natm = length(zs)
    zs_gpu = CuArray{Float64}(zs)
    Ts_gpu = CuArray{Float64}(Ts)
    nd_gpu = CuArray{Float64}(nd)
    vx = CUDA.zeros(Float64, Natm)
    vy = CUDA.zeros(Float64, Natm)
    vz = CUDA.zeros(Float64, Natm)
    σ_v = CUDA.zeros(Float64, Natm)
    μ_v = CUDA.zeros(Float64, Natm)

    return AtmosphereGPU(Natm, τs, zs, Ts, ne, nd, ref_wl, zs_gpu, Ts_gpu, nd_gpu, vx, vy, vz, σ_v, μ_v)
end

"""
    AtmosphereCPU(atm_korg)

Build a CPU-backed atmosphere from a Korg atmosphere object.
"""
mutable struct AtmosphereCPU{T<:AF} <: Atmosphere{T}
    Natm::Int
    τs::AA{T,1}
    zs::AA{T,1}
    Ts::AA{T,1}
    nₑ::AA{T,1}
    nd::AA{T,1}
    reference_wavelength::T  # cm; MARCS reference wavelength for τ_ref (typically 5000 Å)

    vx::AA{T,1}
    vy::AA{T,1}
    vz::AA{T,1}
    σ_v::AA{T,1}
    μ_v::AA{T,1}
end

"""
    AtmosphereCPU(atm_korg)

Construct an `AtmosphereCPU` with thermodynamic and velocity fields on the CPU.
"""
function AtmosphereCPU(atm_korg)
    # resample onto uniform log τ grid
    atm_korg = _resample_log_tau(atm_korg)

    # Korg atmosphere parameters
    τs     = Korg.get_tau_refs(atm_korg)
    zs     = Korg.get_zs(atm_korg)
    Ts     = Korg.get_temps(atm_korg)
    ne     = Korg.get_electron_number_densities(atm_korg)
    nd     = Korg.get_number_densities(atm_korg)
    ref_wl = atm_korg.reference_wavelength  # cm

    Natm = length(zs)
    vx = zeros(Float64, Natm)
    vy = zeros(Float64, Natm)
    vz = zeros(Float64, Natm)
    σ_v = zeros(Float64, Natm)
    μ_v = zeros(Float64, Natm)

    return AtmosphereCPU(Natm, τs, zs, Ts, ne, nd, ref_wl, vx, vy, vz, σ_v, μ_v)
end

"""
    _resample_log_tau(atm_korg; n_layers=length(get_tau_refs(atm_korg)))

Resample a Korg model atmosphere onto a uniform grid in log(τ_ref), returning a new atmosphere
of the same type.

`interpolate_marcs` drops layers where the interpolated grid has NaN values (via `nanmask`),
which produces a non-uniform log-τ spacing that contaminates the anchored τ integration scheme.
This function detects and removes that discontinuity by re-interpolating all thermodynamic
quantities onto a uniform log-τ grid.

By default `n_layers` matches the input layer count, so the function is a pure spacing fix.
Pass an explicit value to upsample or downsample.
"""
function _resample_log_tau(atm_korg; n_layers::Int=length(Korg.get_tau_refs(atm_korg)))
    τ_ref = Korg.get_tau_refs(atm_korg)
    zs    = Korg.get_zs(atm_korg)
    Ts    = Korg.get_temps(atm_korg)
    ne    = Korg.get_electron_number_densities(atm_korg)
    nd    = Korg.get_number_densities(atm_korg)

    log_τ      = log.(τ_ref)
    Δlog_τ     = diff(log_τ)
    step_ratio = maximum(Δlog_τ) / minimum(Δlog_τ)
    if step_ratio > 1.1
        n_in = length(τ_ref)
        suffix = n_layers == n_in ? "" : " → $n_layers layers"
        @warn "_resample_log_tau: non-uniform log-τ spacing (max/min step = " *
              "$(round(step_ratio, digits=2))×, $n_in layers$suffix). " *
              "Resampling atmosphere to uniform grid."
    end

    log_τ_new = collect(range(first(log_τ), last(log_τ), length=n_layers))

    itp_z  = Korg.CubicSplines.CubicSpline(log_τ, zs)
    itp_T  = Korg.CubicSplines.CubicSpline(log_τ, Ts)
    itp_ne = Korg.CubicSplines.CubicSpline(log_τ, ne)
    itp_nd = Korg.CubicSplines.CubicSpline(log_τ, nd)

    τs_new = exp.(log_τ_new)
    zs_new = itp_z.(log_τ_new)
    Ts_new = itp_T.(log_τ_new)
    ne_new = itp_ne.(log_τ_new)
    nd_new = itp_nd.(log_τ_new)

    ref_wl = atm_korg.reference_wavelength
    if atm_korg isa Korg.PlanarAtmosphere
        ls = [Korg.PlanarAtmosphereLayer(τs_new[i], zs_new[i], Ts_new[i], ne_new[i], nd_new[i])
              for i in eachindex(τs_new)]
        return Korg.PlanarAtmosphere(ls, ref_wl)
    else
        R  = atm_korg.R
        ls = [Korg.ShellAtmosphereLayer(τs_new[i], zs_new[i], Ts_new[i], ne_new[i], nd_new[i])
              for i in eachindex(τs_new)]
        return Korg.ShellAtmosphere(ls, R, ref_wl)
    end
end
