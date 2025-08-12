struct GPUMemory{T<:AF}
    λs::CA{T,1}
    αs::CA{T,2}
    τs::CA{T,2}
    cfunc::CA{T,2}
    flux::CA{T,2}
end

function GPUMemory(λs_cpu::AA{T,1}, atm::AtmosphereGPU) where T
    # get dims 
    Nλ = length(λs_cpu)
    Natm = length(atm.zs)

    # allocate arrays
    λs = CuArray{T}(λs_cpu)
    αs = CUDA.zeros(T, Natm, Nλ)
    τs = CUDA.zeros(T, Natm, Nλ)
    cfunc = CUDA.zeros(T, Natm - 1, Nλ)
    flux = CUDA.zeros(T, Natm - 1, Nλ)

    # synchronize and return 
    CUDA.synchronize()
    return GPUMemory(λs, αs, τs, cfunc, flux)
end

function GPUMemory(λs_cpu::AA{T,1}, Natm::Int) where T
    # get dims 
    Nλ = length(λs_cpu)

    # allocate arrays
    λs = CuArray{T}(λs_cpu)
    αs = CUDA.zeros(T, Natm, Nλ)
    τs = CUDA.zeros(T, Natm, Nλ)
    cfunc = CUDA.zeros(T, Natm - 1, Nλ)
    flux = CUDA.zeros(T, Natm - 1, Nλ)

    # synchronize and return 
    CUDA.synchronize()
    return GPUMemory(λs, αs, τs, cfunc, flux)
end