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
    atm_korg = _resample_log_tau(atm_korg)
    # Korg atmosphere parameters
    τs = Korg.get_tau_refs(atm_korg)
    zs = Korg.get_zs(atm_korg)
    Ts = Korg.get_temps(atm_korg)
    ne = Korg.get_electron_number_densities(atm_korg)
    nd = Korg.get_number_densities(atm_korg)

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

    return AtmosphereGPU(Natm, τs, zs, Ts, ne, nd, zs_gpu, Ts_gpu, nd_gpu, vx, vy, vz, σ_v, μ_v)
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
    atm_korg = _resample_log_tau(atm_korg)
    # Korg atmosphere parameters
    τs = Korg.get_tau_refs(atm_korg)
    zs = Korg.get_zs(atm_korg)
    Ts = Korg.get_temps(atm_korg)
    ne = Korg.get_electron_number_densities(atm_korg)
    nd = Korg.get_number_densities(atm_korg)

    # allocate on gpu
    Natm = length(zs)
    vx = zeros(Float64, Natm)
    vy = zeros(Float64, Natm)
    vz = zeros(Float64, Natm)
    σ_v = zeros(Float64, Natm)
    μ_v = zeros(Float64, Natm)

    return AtmosphereCPU(Natm, τs, zs, Ts, ne, nd, vx, vy, vz, σ_v, μ_v)
end

"""
    _resample_log_tau(atm_korg; n_layers=56)

Resample a Korg model atmosphere onto a uniform grid in log(τ_ref), returning a new atmosphere
of the same type.

`interpolate_marcs` drops layers where the interpolated grid has NaN values (via `nanmask`),
which produces a non-uniform log-τ spacing that contaminates the anchored τ integration scheme.
This function detects and removes that discontinuity by re-interpolating all thermodynamic
quantities onto a uniform log-τ grid.
"""
function _resample_log_tau(atm_korg; n_layers::Int=56)
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
        # @warn "_resample_log_tau: non-uniform log-τ spacing (max/min step = " *
        #       "$(round(step_ratio, digits=2))×, $n_in layers → $n_layers). " *
        #       "Resampling to uniform grid. Expected for MARCS grid-interpolated atmospheres " *
        #       "(nanmask drops outer layers)."
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

"""
    get_marcs_atm(Teff, logg, A_X; n_layers=56)

Interpolate a MARCS atmosphere from Korg and return it with `n_layers` layers.
"""
function get_marcs_atm(Teff::T, logg::T, A_X::AA{T,1}; n_layers::Int=56) where T<:AF
    # get the model atmosphere
    marcs_atm = Korg.interpolate_marcs(Teff, logg, A_X)
    τ_500 = Korg.get_tau_refs(marcs_atm)
    zs = Korg.get_zs(marcs_atm)
    Ts = Korg.get_temps(marcs_atm)
    ne = Korg.get_electron_number_densities(marcs_atm)
    nd = Korg.get_number_densities(marcs_atm)

    # interpolate in zs
    itp_τs = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(τ_500))
    itp_Ts = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(Ts))
    itp_ne = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(ne))
    itp_nd = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(nd))

    zs_new = range(last(zs), first(zs), length=n_layers)
    τs_new = reverse(itp_τs.(zs_new))
    Ts_new = reverse(itp_Ts.(zs_new))
    ne_new = reverse(itp_ne.(zs_new))
    nd_new = reverse(itp_nd.(zs_new))
    zs_new = reverse(collect(zs_new))

    ls = Array{Korg.PlanarAtmosphereLayer{Float64, Float64, Float64, Float64, Float64}}(undef, length(zs_new))
    for i in eachindex(zs_new)
        ls[i] = Korg.PlanarAtmosphereLayer(τs_new[i], zs_new[i], Ts_new[i], ne_new[i], nd_new[i])
    end
    return Korg.PlanarAtmosphere(ls, 5000.0 / 1e8)
end
