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
    # copy opacities (skip when signal FFT is cached — αs unchanged)
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, μ_v, σ_v)

    # compute taus
    calc_tau_bezier_cached!(μ_tile, atm.zs_gpu, αs_gpu, mem.τs,
                            mem.tau_ds, mem.tau_alphaC)

    # compute the contribution function
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_intensity_cfunc!(μ_tile, atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    return nothing
end

# Like calc_intensity_cfunc! but writes intensity directly (fused cfunc+reduce).
# Does NOT populate mem.cfunc — use calc_intensity_cfunc! if you need the cfunc matrix.
function calc_intensity_direct!(out::CA{T,1}, αs_init::AA{T,2}, atm::AtmosphereGPU{T},
                                mem::GPUMemory, cmem::ConvolutionMemory, μ_tile::T,
                                μ_v::CA{T,1}, σ_v::CA{T,1}) where T<:AF
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, μ_v, σ_v)
    calc_tau_bezier_cached!(μ_tile, atm.zs_gpu, αs_gpu, mem.τs,
                            mem.tau_ds, mem.tau_alphaC)
    cfunc_reduce_intensity!(out, μ_tile, atm.Ts_gpu, mem.λs, mem.τs)
    return nothing
end

function calc_flux_cfunc!(αs_init::AA{T,2}, atm::AtmosphereGPU{T}, mem::GPUMemory,
                         cmem::ConvolutionMemory, σ_v::CA{T,1}) where T<:AF
    # move alphas to reusable buffers and zero mean velocity in-place
    cmem.signal_cached || copyto!(mem.αs, αs_init)
    fill!(atm.μ_v, zero(T))
    αs_gpu = convolve_wavelength_axis_gpu(cmem, mem.λs, mem.αs, atm.μ_v, σ_v)

    # compute taus
    calc_tau_bezier_cached!(1.0, atm.zs_gpu, αs_gpu, mem.τs,
                            mem.tau_ds, mem.tau_alphaC)

    # compute the contribution function
    ts = (32, 16)
    bs = (cld(cmem.Nλ, ts[1]), cld(cmem.Natm, ts[2]))
    @cuda threads=ts blocks=bs calc_flux_cfunc!(atm.Ts_gpu, mem.λs, mem.τs, mem.cfunc)
    return nothing
end

function calc_intensity_cfunc!(μ_i::T, Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM) where T<:AF
    # thread indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    # Gauss-Legendre two-point abscissa constant
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    # loop over lambda
    for j in idx:sdx:length(λs)
        # convert to cm
        λ_cm = λs[j] * T(1e-8)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

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

            # linear T interp at fixed 2-point Gauss nodes
            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            # evaluate integrand f = B(T,λ) * exp(-τ) 
            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * exp(-τp1)
            f2 = B2 * exp(-τp2)

            # store contribution
            @inbounds cfunc[k, j] = T(0.5) * (f1 + f2) * T(1e-8)
        end
    end
    return nothing
end

# Fused cfunc + atmosphere reduction: computes intensity directly without
# materializing the cfunc matrix.  2D thread block (x=wavelength, y=atmosphere
# chunk) with shared-memory reduction across y.
function cfunc_reduce_intensity_kernel!(out::CDV{T}, Ts::CDV, λs::CDV,
                                        τs::CDM, Natm1::Int32,
                                        Nλ::Int32) where T<:AF
    shmem = CuDynamicSharedArray(T, Int(blockDim().x) * Int(blockDim().y))

    tx = threadIdx().x
    ty = threadIdx().y
    bdx = Int32(blockDim().x)
    bdy = Int32(blockDim().y)
    j = tx + bdx * (blockIdx().x - Int32(1))

    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    partial = zero(T)
    if j <= Nλ
        λ_cm = λs[j] * T(1e-8)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

        k = ty
        while k <= Natm1
            @inbounds begin
                τ0 = τs[k, j]
                τ1 = τs[k + Int32(1), j]
                Δτ = τ1 - τ0
                τ_mid = T(0.5) * (τ0 + τ1)

                τp1 = τ_mid - T(0.5) * Δτ * one_over_sqrt3
                τp2 = τ_mid + T(0.5) * Δτ * one_over_sqrt3

                dT = Ts[k + Int32(1)] - Ts[k]
                T1 = Ts[k] + dT * frac1
                T2 = Ts[k] + dT * frac2

                B1 = bb_num / (exp(bb_x / T1) - one(T))
                B2 = bb_num / (exp(bb_x / T2) - one(T))
                f1 = B1 * exp(-τp1)
                f2 = B2 * exp(-τp2)

                cfunc_val = T(0.5) * (f1 + f2) * T(1e-8)
                partial = muladd(cfunc_val, Δτ, partial)
            end
            k += bdy
        end
    end

    # shared memory reduction across atmosphere (y) dimension
    sidx = Int(tx) + (Int(ty) - Int32(1)) * Int(bdx)
    @inbounds shmem[sidx] = partial
    sync_threads()

    s = Int(bdy) >> 1
    while s >= 1
        if Int(ty) <= s
            @inbounds shmem[sidx] += shmem[sidx + s * Int(bdx)]
        end
        sync_threads()
        s >>= 1
    end

    if ty == Int32(1) && j <= Nλ
        @inbounds out[j] = shmem[Int(tx)]
    end
    return nothing
end

# Unfused reduction: out[j] = sum_k cfunc[k,j] * Δτ[k,j].
# Use when cfunc matrix is already computed (e.g., for diagnostics).
function reduce_intensity_kernel!(out::CDV{T}, cfunc::CDM{T}, τs::CDM{T},
                                  Natm1::Int32, Nλ::Int32) where T<:AF
    j = Int32(threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x)
    s = Int32(blockDim().x * gridDim().x)
    while j <= Nλ
        acc = zero(T)
        @inbounds for k in Int32(1):Natm1
            Δτ = τs[k + Int32(1), j] - τs[k, j]
            acc = muladd(cfunc[k, j], Δτ, acc)
        end
        @inbounds out[j] = acc
        j += s
    end
    return nothing
end

function reduce_intensity!(out::CA{T,1}, cfunc::CA{T,2}, τs::CA{T,2}) where T<:AF
    Nλ = Int32(size(cfunc, 2))
    Natm1 = Int32(size(cfunc, 1))
    size(out, 1) == Nλ || error("reduce_intensity!: output length mismatch")
    size(τs, 1) == Natm1 + 1 || error("reduce_intensity!: τs/cfunc atmosphere dimension mismatch")
    size(τs, 2) == Nλ || error("reduce_intensity!: τs/cfunc wavelength dimension mismatch")

    threads = 256
    blocks = cld(Int(Nλ), threads)
    @cuda threads=threads blocks=blocks reduce_intensity_kernel!(out, cfunc, τs, Natm1, Nλ)
    return nothing
end

function cfunc_reduce_intensity!(out::CA{T,1}, μ_i::T, Ts::CA{T,1},
                                  λs::CA{T,1}, τs::CA{T,2}) where T<:AF
    Nλ = Int32(length(λs))
    Natm1 = Int32(size(τs, 1) - 1)
    length(out) == Nλ || error("cfunc_reduce_intensity!: output length mismatch")

    ts = (32, 16)
    bs = (cld(Int(Nλ), ts[1]), 1)
    shmem_bytes = prod(ts) * sizeof(T)
    @cuda threads=ts blocks=bs shmem=shmem_bytes cfunc_reduce_intensity_kernel!(
        out, Ts, λs, τs, Natm1, Nλ)
    return nothing
end

function calc_flux_cfunc!(Ts::CDV, λs::CDV, τs::CDM, cfunc::CDM)
    # thread indices
    idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    sdx = gridDim().x * blockDim().x
    idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
    sdy = gridDim().y * blockDim().y

    # Gauss-Legendre two-point abscissa constant
    T = eltype(Ts)
    one_over_sqrt3 = one(T) / sqrt(T(3))
    frac1 = T(0.5) * (one(T) - one_over_sqrt3)
    frac2 = T(0.5) * (one(T) + one_over_sqrt3)

    # loop over lambda
    for j in idx:sdx:length(λs)
        λ_cm = λs[j] * T(1e-8)
        λ5 = λ_cm * λ_cm * λ_cm * λ_cm * λ_cm
        bb_num = T(2.0) * T(h) * (T(c)^2) / λ5
        bb_x = T(h) * T(c) / (λ_cm * T(kB))

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

            # linear T interp at fixed 2-point Gauss nodes
            dT = Ts[k+1] - Ts[k]
            T1 = Ts[k] + dT * frac1
            T2 = Ts[k] + dT * frac2

            # evaluate integrand f = B(T,λ) * E_2(τ)  
            B1 = bb_num / (exp(bb_x / T1) - one(T))
            B2 = bb_num / (exp(bb_x / T2) - one(T))
            f1 = B1 * E_2(τp1)
            f2 = B2 * E_2(τp2)

            # store contribution
            @inbounds cfunc[k, j] = T(0.5) * (f1 + f2) * T(1e-8)
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
