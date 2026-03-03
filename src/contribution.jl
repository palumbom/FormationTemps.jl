E_2(τ) = Korg.RadiativeTransfer.exponential_integral_2(τ)
# E_2(τ) = exponential_integral_2_gpu(τ)

function calc_intensity_quantities(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory, 
                                   cmem::ConvolutionMemory, μ_tile::T, μ_v::CA{T,1}, 
                                   σ_v::CA{T,1}) where T<:AF
    # get contribution function
    calc_intensity_cfunc!(αs_init, atm, mem, cmem, μ_tile, μ_v, σ_v)

    # multiply by differential for cum. cont. & intensity
    cfunc_dt = mem.cfunc .* diff(mem.τs, dims=1)
    return IntensityContFunc(mem.cfunc, cfunc_dt)
end

function calc_flux_quantities(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory, 
                              cmem::ConvolutionMemory, σ_v::CA{T,1}) where T<:AF
    # get contribution function
    calc_flux_cfunc!(αs_init, atm, mem, cmem, σ_v)

    # multiply by differential for cum. cont. & flux
    cfunc_dt = mem.cfunc .* diff(mem.τs, dims=1)
    return FluxContFunc(mem.cfunc, cfunc_dt)
end

function calc_intensity_cfunc!(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory, 
                               cmem::ConvolutionMemory, μ_tile::T, μ_v::CA{T,1}, 
                               σ_v::CA{T,1}) where T<:AF
    # perturb the alphas
    αs_gpu = CuArray{Float64}(αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, αs_gpu, μ_v, σ_v)
    CUDA.synchronize()

    # compute taus
    ts = 512 
    bs = cld(cmem.Nλ, ts)
    @cusync @cuda threads=ts blocks=bs calc_tau!(μ_tile, atm.zs_gpu, αs_gpu, mem.τs)

    # compute the contribution function
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cusync @cuda threads=ts blocks=bs calc_intensity_cfunc!(μ_tile, atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    return nothing
end

function calc_flux_cfunc!(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory, 
                          cmem::ConvolutionMemory, σ_v::CA{T,1}) where T<:AF
    # move alphas to GPU
    μ_v = CUDA.zeros(T, length(σ_v))
    αs_gpu = CuArray{Float64}(αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, αs_gpu, μ_v, σ_v)
    CUDA.synchronize()

    # compute taus
    ts = 512 
    bs = cld(cmem.Nλ, ts)
    @cusync @cuda threads=ts blocks=bs calc_tau!(1.0, atm.zs_gpu, αs_gpu, mem.τs)

    # compute the contribution function
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cusync @cuda threads=ts blocks=bs calc_flux_cfunc!(atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    return nothing
end

function calc_intensity_cfunc!(μ_i::T, Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM) where T<:AF
    # thread indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    # Gauss-Legendre two-point abscissa constant
    one_over_sqrt3 = 1.0 / sqrt(3.0)

    # loop over lambda
    for j in idx:sdx:length(λs)
        # convert to cm
        λ_cm = λs[j] * 1e-8

        # loop over atmosphere
        for k in idy:sdy:length(Ts)-1
            # endpoints in τ-space
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            # Gauss nodes
            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            # linear T interp wrt τ
            dT = Ts[k+1] - Ts[k]
            inv_dτ = 1.0 / Δτ
            T1 = Ts[k] + dT * ((τp1 - τ0) * inv_dτ)
            T2 = Ts[k] + dT * ((τp2 - τ0) * inv_dτ)

            # evaluate integrand f = B(T,λ) * exp(-τ) 
            f1 = blackbody_gpu(T1, λ_cm) * exp(-τp1)
            f2 = blackbody_gpu(T2, λ_cm) * exp(-τp2)

            # store contribution
            @inbounds cfunc[k, j] = 0.5 * (f1 + f2) * 1e-8
        end
    end
    return nothing
end

function calc_flux_cfunc!(Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM)
    # thread indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    # Gauss-Legendre two-point abscissa constant
    one_over_sqrt3 = 1.0 / sqrt(3.0)

    # loop over lambda
    for j in idx:sdx:length(λs)
        λ_cm = λs[j] * 1e-8

        # loop over atmosphere
        for k in idy:sdy:length(Ts)-1
            # endpoints in τ-space
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            # Gauss nodes
            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            # linear T interp wrt τ
            dT = Ts[k+1] - Ts[k]
            inv_dτ = 1.0 / Δτ
            T1 = Ts[k] + dT * ((τp1 - τ0) * inv_dτ)
            T2 = Ts[k] + dT * ((τp2 - τ0) * inv_dτ)

            # evaluate integrand f = B(T,λ) * E_2(τ)  
            f1 = blackbody_gpu(T1, λ_cm) * E_2(τp1)
            f2 = blackbody_gpu(T2, λ_cm) * E_2(τp2)

            # store contribution
            @inbounds cfunc[k, j] = 0.5 * (f1 + f2) * 1e-8
        end
    end
    return nothing
end

function calc_intensity_cfunc_cpu!(cfunc::AA{T,2}, Ts::AA{T,1}, λs::AA{T,1},
                                   τs::AA{T,2}) where {T<:AF}
    Natm = length(Ts)
    one_over_sqrt3 = one(T) / sqrt(T(3))
    @inbounds for j in 1:length(λs)
        λ_cm = λs[j] * T(1e-8)
        for k in 1:Natm-1
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            dT = Ts[k+1] - Ts[k]
            inv_dτ = one(T) / Δτ
            T1 = Ts[k] + dT * ((τp1 - τ0) * inv_dτ)
            T2 = Ts[k] + dT * ((τp2 - τ0) * inv_dτ)

            f1 = Korg.blackbody(T1, λ_cm) * exp(-τp1)
            f2 = Korg.blackbody(T2, λ_cm) * exp(-τp2)
            @inbounds cfunc[k, j] = 0.5 * (f1 + f2) * T(1e-8)
        end
    end
    return nothing
end

function calc_flux_cfunc_cpu!(cfunc::AA{T,2}, Ts::AA{T,1}, λs::AA{T,1},
                              τs::AA{T,2}) where {T<:AF}
    Natm = length(Ts)
    one_over_sqrt3 = one(T) / sqrt(T(3))
    E2 = Korg.RadiativeTransfer.exponential_integral_2
    @inbounds for j in 1:length(λs)
        λ_cm = λs[j] * T(1e-8)
        for k in 1:Natm-1
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            dT = Ts[k+1] - Ts[k]
            inv_dτ = one(T) / Δτ
            T1 = Ts[k] + dT * ((τp1 - τ0) * inv_dτ)
            T2 = Ts[k] + dT * ((τp2 - τ0) * inv_dτ)

            f1 = Korg.blackbody(T1, λ_cm) * E2(τp1)
            f2 = Korg.blackbody(T2, λ_cm) * E2(τp2)
            @inbounds cfunc[k, j] = 0.5 * (f1 + f2) * T(1e-8)
        end
    end
    return nothing
end
