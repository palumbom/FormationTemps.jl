function calc_intensity_cfunc(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory, 
                              cmem::ConvolutionMemory, μ_tile::T, μ_v::CA{T,1}, σ_v::CA{T,1}) where T<:AF
    # perturb the alphas
    # αs_gpu = CuArray{Float64}(αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, αs_init, μ_v, σ_v)

    # compute taus
    ts = 512 
    bs = cld(cmem.Nλ, ts)
    @cuda threads=ts blocks=bs calc_tau!(μ_tile, atm.zs_gpu, αs_gpu, mem.τs)
    CUDA.synchronize()

    # compute the contribution function
    # ts = (32, 32)
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_intensity_cfunc!(μ_tile, atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    CUDA.synchronize()
    return Array(mem.cfunc)
end

function calc_flux_cfunc(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory, 
                                         cmem::ConvolutionMemory, σ_v::CA{T,1}) where T<:AF
    # move alphas to GPU
    # αs_gpu = CuArray{Float64}(αs_init)
    μ_v = CUDA.zeros(T, length(σ_v))
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, αs_init, μ_v, σ_v)

    # compute taus
    ts = 512 
    bs = cld(cmem.Nλ, ts)
    @cuda threads=ts blocks=bs calc_tau!(1.0, atm.zs_gpu, αs_gpu, mem.τs)
    CUDA.synchronize()

    # compute the contribution function
    # ts = (32, 32)
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_flux_cfunc!(atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    CUDA.synchronize()
    return Array(mem.cfunc)
end

function calc_intensity_cfunc!(μ_i::T, Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM) where T<:AF
 # thread indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    # Gauss-Legendre two-point abscissa constant
    one_over_sqrt3 = 1.0 / sqrt(3.0)

    for j in idx:sdx:length(λs)
        # convert to cm
        λ_cm = λs[j] * 1e-8

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

            # two-point Gauss weight = Δτ/2
            @inbounds cfunc[k, j] = (f1 + f2) * (Δτ * 0.5)
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

    for j in idx:sdx:length(λs)
        λ_cm = λs[j] * 1e-8  # Angstrom → cm

        for k in idy:sdy:length(Ts)-1
            # Interval endpoints
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            # Gauss nodes
            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            # Linear interp of T(τ)
            dT = Ts[k+1] - Ts[k]
            inv_dτ = 1.0 / Δτ
            T1 = Ts[k] + dT * ((τp1 - τ0) * inv_dτ)
            T2 = Ts[k] + dT * ((τp2 - τ0) * inv_dτ)

            # Evaluate integrand: B(T) * E2(τ)
            f1 = blackbody_gpu(T1, λ_cm) * Korg.RadiativeTransfer.exponential_integral_2(τp1) # SpecialFunctions.expint(2, τp1)
            f2 = blackbody_gpu(T2, λ_cm) * Korg.RadiativeTransfer.exponential_integral_2(τp2) # SpecialFunctions.expint(2, τp2)

            @inbounds cfunc[k, j] = 2π * (f1 + f2) * (Δτ * 0.5)
        end
    end
    return nothing
end

function calc_intensity_cfunc_cpu(μ::T, Ts::AA{T,1}, λs::AA{T,1}, τs::AA{T,2}) where {T<:AF}
    # get dims, preallocate
    Natm, Nλ = size(τs)
    one_over_sqrt3 = 1.0 / sqrt(3.0)
    cfunc = zeros(Natm - 1, Nλ)

    # loop over wavelength
    for j in 1:Nλ
        # convert to cm
        λ_cm = λs[j] * 1e-8

        # loop over layers of atmospbere
        for k in 1:Natm-1
            # endpoints in τ-space
            τ0 = τs[k, j]
            τ1 = τs[k+1, j]
            Δτ = τ1 - τ0
            τ_mid = 0.5 * (τ0 + τ1)

            # Gauss–Legendre nodes
            τp1 = τ_mid - 0.5 * Δτ * one_over_sqrt3
            τp2 = τ_mid + 0.5 * Δτ * one_over_sqrt3

            # linear T interp wrt τ
            dT = Ts[k+1] - Ts[k]
            inv_Δτ = 1.0 / Δτ
            T1 = Ts[k] + dT * ((τp1 - τ0) * inv_Δτ)
            T2 = Ts[k] + dT * ((τp2 - τ0) * inv_Δτ)

            # evaluate integrand f = B(T,λ) * exp(-τ)
            f1 = Korg.blackbody(T1, λ_cm) * exp(-τp1)
            f2 = Korg.blackbody(T2, λ_cm) * exp(-τp2)

            # two-point Gauss weight = Δτ/2
            cfunc[k, j] = (f1 + f2) * (Δτ * 0.5)
        end
    end
    return cfunc
end

