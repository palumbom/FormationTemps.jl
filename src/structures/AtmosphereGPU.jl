mutable struct AtmosphereGPU{T<:AF}
    Natm::Int
    τs::AA{T,1}
    zs::AA{T,1}
    Ts::AA{T,1}
    nₑ::AA{T,1}
    nd::AA{T,1}

    zs_gpu::AA{T,1}
    Ts_gpu::AA{T,1}
    vx::CA{T,1}
    vy::CA{T,1}
    vz::CA{T,1}
    σ_v::CA{T,1}
    μ_v::CA{T,1}
end

function AtmosphereGPU(atm_korg)
    # Korg atmosphere parameters
    τs = Korg.get_tau_5000s(atm_korg)
    zs = Korg.get_zs(atm_korg)
    Ts = Korg.get_temps(atm_korg)
    ne = Korg.get_electron_number_densities(atm_korg)
    nd = Korg.get_number_densities(atm_korg)

    # allocate on gpu 
    Natm = length(zs)
    zs_gpu = CuArray{Float64}(zs)
    Ts_gpu = CuArray{Float64}(Ts)
    vx = CUDA.zeros(Float64, Natm)
    vy = CUDA.zeros(Float64, Natm)
    vz = CUDA.zeros(Float64, Natm)
    σ_v = CUDA.zeros(Float64, Natm)
    μ_v = CUDA.zeros(Float64, Natm)

    return AtmosphereGPU(Natm, τs, zs, Ts, ne, nd, zs_gpu, Ts_gpu, vx, vy, vz, σ_v, μ_v)
end