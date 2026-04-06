"""
    Atmosphere{T}

Abstract atmosphere type used by formation temperature calculations.
"""
abstract type Atmosphere{T<:AF} end

"""
    Atmosphere(::Type{A}, atm_korg) where A<:Atmosphere

Type-dispatched factory constructor. Returns a concrete `A` from a Korg model atmosphere.

    Atmosphere(AtmosphereGPU, atm_korg)   # → AtmosphereGPU
    Atmosphere(AtmosphereCPU, atm_korg)   # → AtmosphereCPU
"""
Atmosphere(::Type{A}, atm_korg) where {A<:Atmosphere} = A(atm_korg)

"""
    get_τs(atm)

Return the optical depth reference grid as a standard `Array`. Returns an empty vector
when the atmosphere was constructed from a model that does not supply `tau_ref` (in
that case the Bézier τ integrator is used and `atm.τs` is empty).
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
    get_nd(atm)

Return the number density grid as a standard `Array`.
"""
get_nd(atm::Atmosphere) = Array(atm.nd)


"""
    _extract_korg_fields(atm_korg)

Resample a Korg model atmosphere onto a uniform log-τ grid, then extract thermodynamic
fields into a `NamedTuple`. Shared by both `AtmosphereCPU` and `AtmosphereGPU` constructors.
"""
function _extract_korg_fields(atm_korg)
    atm_korg = _resample_log_tau(atm_korg)

    τs = try
        Korg.get_tau_refs(atm_korg)
    catch
        Float64[]
    end
    zs     = Korg.get_zs(atm_korg)
    Ts     = Korg.get_temps(atm_korg)
    nₑ     = Korg.get_electron_number_densities(atm_korg)
    nd     = Korg.get_number_densities(atm_korg)
    ref_wl = try
        atm_korg.reference_wavelength  # cm
    catch
        5e-5  # default MARCS reference wavelength (5000 Å in cm)
    end

    return (; τs, zs, Ts, nₑ, nd, ref_wl, Natm=length(zs))
end

"""
    AtmosphereGPU{T<:AbstractFloat} <: Atmosphere{T}

GPU-backed atmosphere wrapping a Korg MARCS model.

Fields:
- `Natm`: Number of atmosphere layers.
- `τs`: Reference optical depth grid (length `Natm`); empty if `tau_ref` is unavailable.
- `zs`, `Ts`: Height (cm) and temperature (K) grids on the CPU.
- `nₑ`, `nd`: Electron and total number density grids on the CPU.
- `reference_wavelength`: MARCS reference wavelength for `τ_ref` (cm; typically 5000 Å).
- `zs_gpu`, `Ts_gpu`, `nd_gpu`: Height, temperature, and number density on the device.
- `vx`, `vy`, `vz`: Per-layer velocity components on the device (m/s). Initialized to zero
  by the convenience constructor; populated by downstream packages from MHD simulation data.
- `σ_v`, `μ_v`: Per-layer microturbulent speed and mean line-of-sight velocity on the device (m/s).

See also: [`AtmosphereGPU(atm_korg)`](@ref)
"""
struct AtmosphereGPU{T<:AF} <: Atmosphere{T}
    Natm::Int
    τs::Vector{T}
    zs::Vector{T}
    Ts::Vector{T}
    nₑ::Vector{T}
    nd::Vector{T}
    reference_wavelength::T  # cm; MARCS reference wavelength for τ_ref (typically 5000 Å)

    zs_gpu::CuVector{T}
    Ts_gpu::CuVector{T}
    nd_gpu::CuVector{T}
    vx::CuVector{T}
    vy::CuVector{T}
    vz::CuVector{T}
    σ_v::CuVector{T}
    μ_v::CuVector{T}
end

"""
    AtmosphereGPU(atm_korg; T=Float64)

Construct an `AtmosphereGPU{T}` with thermodynamic fields from Korg and velocity
fields allocated on the GPU. Pass `T=Float32` for single-precision GPU arrays.

Korg always returns Float64 data; the constructor converts all fields (CPU and GPU)
to type `T`.

The input atmosphere is first resampled onto a uniform log-τ grid to remove
non-uniform layer spacing introduced by `Korg.interpolate_marcs`. If the model
does not supply `tau_ref` (e.g., some non-MARCS grids), resampling is skipped and
`atm.τs` is set to an empty vector — the Bézier τ integrator is used automatically
downstream in that case.
"""
function AtmosphereGPU(atm_korg; T::Type{<:AF}=Float64)
    f = _extract_korg_fields(atm_korg)

    zs_gpu = CuArray{T}(f.zs)
    Ts_gpu = CuArray{T}(f.Ts)
    nd_gpu = CuArray{T}(f.nd)
    vx     = CUDA.zeros(T, f.Natm)
    vy     = CUDA.zeros(T, f.Natm)
    vz     = CUDA.zeros(T, f.Natm)
    σ_v    = CUDA.zeros(T, f.Natm)
    μ_v    = CUDA.zeros(T, f.Natm)

    return AtmosphereGPU(f.Natm, T.(f.τs), T.(f.zs), T.(f.Ts), T.(f.nₑ), T.(f.nd),
                         T(f.ref_wl), zs_gpu, Ts_gpu, nd_gpu, vx, vy, vz, σ_v, μ_v)
end

"""
    AtmosphereCPU{T<:AbstractFloat} <: Atmosphere{T}

CPU-backed atmosphere wrapping a Korg MARCS model.

Fields:
- `Natm`: Number of atmosphere layers.
- `τs`: Reference optical depth grid (length `Natm`); empty if `tau_ref` is unavailable.
- `zs`, `Ts`: Height (cm) and temperature (K) grids.
- `nₑ`, `nd`: Electron and total number density grids.
- `reference_wavelength`: MARCS reference wavelength for `τ_ref` (cm; typically 5000 Å).
- `vx`, `vy`, `vz`: Per-layer velocity components (m/s).
- `σ_v`, `μ_v`: Per-layer microturbulent speed and mean line-of-sight velocity (m/s).

See also: [`AtmosphereCPU(atm_korg)`](@ref)
"""
struct AtmosphereCPU{T<:AF} <: Atmosphere{T}
    Natm::Int
    τs::Vector{T}
    zs::Vector{T}
    Ts::Vector{T}
    nₑ::Vector{T}
    nd::Vector{T}
    reference_wavelength::T  # cm; MARCS reference wavelength for τ_ref (typically 5000 Å)

    vx::Vector{T}
    vy::Vector{T}
    vz::Vector{T}
    σ_v::Vector{T}
    μ_v::Vector{T}
end

"""
    AtmosphereCPU(atm_korg)

Construct an `AtmosphereCPU` with thermodynamic and velocity fields on the CPU.

The input atmosphere is first resampled onto a uniform log-τ grid to remove
non-uniform layer spacing introduced by `Korg.interpolate_marcs`. If the model
does not supply `tau_ref` (e.g., some non-MARCS grids), resampling is skipped and
`atm.τs` is set to an empty vector — the Bézier τ integrator is used automatically
downstream in that case.
"""
function AtmosphereCPU(atm_korg)
    f = _extract_korg_fields(atm_korg)

    vx  = zeros(Float64, f.Natm)
    vy  = zeros(Float64, f.Natm)
    vz  = zeros(Float64, f.Natm)
    σ_v = zeros(Float64, f.Natm)
    μ_v = zeros(Float64, f.Natm)

    return AtmosphereCPU(f.Natm, f.τs, f.zs, f.Ts, f.nₑ, f.nd, f.ref_wl,
                         vx, vy, vz, σ_v, μ_v)
end

"""
    _resample_log_tau(atm_korg; n_layers=-1)

Resample a Korg model atmosphere onto a uniform grid in log(τ_ref), returning a new atmosphere
of the same type. Returns `atm_korg` unchanged if tau_ref is unavailable.

`interpolate_marcs` drops layers where the interpolated grid has NaN values (via `nanmask`),
which produces a non-uniform log-τ spacing that contaminates the anchored τ integration scheme.
This function detects and removes that discontinuity by re-interpolating all thermodynamic
quantities onto a uniform log-τ grid.

By default `n_layers` matches the input layer count, so the function is a pure spacing fix.
Pass an explicit value to upsample or downsample.
"""
function _resample_log_tau(atm_korg; n_layers::Int=-1)
    τ_ref = try
        Korg.get_tau_refs(atm_korg)
    catch
        return atm_korg  # no tau_ref available; cannot resample
    end
    isempty(τ_ref) && return atm_korg
    any(isnan, τ_ref) && return atm_korg

    zs    = Korg.get_zs(atm_korg)
    Ts    = Korg.get_temps(atm_korg)
    ne    = Korg.get_electron_number_densities(atm_korg)
    nd    = Korg.get_number_densities(atm_korg)

    n_layers_eff = n_layers < 0 ? length(τ_ref) : n_layers
    log_τ      = log.(τ_ref)
    Δlog_τ     = diff(log_τ)
    step_ratio = maximum(Δlog_τ) / minimum(Δlog_τ)
    if step_ratio > 1.1
        n_in = length(τ_ref)
        suffix = n_layers_eff == n_in ? "" : " → $n_layers_eff layers"
        @warn "_resample_log_tau: non-uniform log-τ spacing (max/min step = " *
              "$(round(step_ratio, digits=2))×, $n_in layers$suffix). " *
              "Resampling atmosphere to uniform d log-τ grid." maxlog=1
    end

    log_τ_new = collect(range(first(log_τ), last(log_τ), length=n_layers_eff))

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
