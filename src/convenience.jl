"""
    calc_formation_temp(star, linelist; use_gpu=true, Δλ=0.01, convolve=false, u1=NaN, u2=NaN, Nϕ=128)

Compute flux formation temperatures for a given star and linelist.

# Examples
```julia-repl
star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0)
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[1:500]
result = calc_formation_temp(star, linelist; Δλ=0.01, convolve=true, u1=0.43, u2=0.31)
```
"""
function calc_formation_temp(star::StellarProps, linelist; use_gpu::Bool=GPU_DEFAULT,
                             Δλ::T=0.01, convolve::Bool=false,
                             u1::T=NaN, u2::T=NaN, Nϕ::Int=128) where T<:AF
    if use_gpu
        form_temps_flux = _calc_formation_temp_gpu(star, linelist; Δλ=Δλ, convolve=convolve, u1=u1, u2=u2,
                                                   Nϕ=Nϕ)
    else
        form_temps_flux = _calc_formation_temp_cpu(star, linelist; Δλ=Δλ, convolve=convolve, u1=u1, u2=u2,
                                                   Nϕ=Nϕ)
    end
    return form_temps_flux
end

function _calc_formation_temp_cpu(star::StellarProps, linelist; Δλ::T=0.01,
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN,
                                  Nϕ::Int=128) where T<:AF
    # get linelist 
    wls = [l.wl * 1e8 for l in linelist]
    λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)

    # get model atmosphere
    atm_cpu = AtmosphereCPU(Korg.interpolate_marcs(star.Teff, star.logg, star.A_X))
    zs = atm_cpu.zs
    Ts = atm_cpu.Ts

    # get the absorption coefficients
    Natm = length(zs)
    Nλ = length(λs_korg)
    αs = zeros(T, Natm, Nλ)
    αs_cont = zeros(T, Natm, Nλ)
    compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_cpu, star.A_X)

    # set microturbulent broadening
    σ_v = fill(star.ξ, Natm)
    μ_v = zeros(T, Natm)

    # get the "stationary" flux
    αs_broad = convolve_wavelength_axis(λs_korg, αs, μ_v, σ_v)
    αs_cont_broad = convolve_wavelength_axis(λs_korg, αs_cont, μ_v, σ_v)

    τs = zeros(T, Natm, Nλ)
    τs_cont = zeros(T, Natm, Nλ)
    calc_tau_cpu!(one(T), zs, αs_broad, τs)
    calc_tau_cpu!(one(T), zs, αs_cont_broad, τs_cont)

    cfunc_flux = zeros(T, Natm - 1, Nλ)
    cfunc_flux_cont = zeros(T, Natm - 1, Nλ)
    calc_flux_cfunc_cpu!(cfunc_flux, Ts, λs_korg, τs)
    calc_flux_cfunc_cpu!(cfunc_flux_cont, Ts, λs_korg, τs_cont)

    cfunc_dt_flux = cfunc_flux .* diff(τs, dims=1)
    cfunc_dt_flux_cont = cfunc_flux_cont .* diff(τs_cont, dims=1)

    # convolution or numerical integration
    if convolve
        @assert !isnan(u1)
        @assert !isnan(u2)
        cfunc_dt_flux = convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux, star.vsini, star.ζ, u1, u2)
        cfunc_dt_flux_cont = convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux_cont, star.vsini, star.ζ, u1, u2)
    else # numerical disk integration
        μs, dA, z_rot = calc_stellar_grid_cpu(star.ρstar, star.istar, star.vsini, Nϕ)
        idx = findall(x -> x .> zero(T), μs)
        μs_cpu = μs[idx]
        dA_cpu = dA[idx]
        z_rot_cpu = z_rot[idx]
        if iszero(star.vsini)
            z_rot_cpu .= 0.0
        end

        flux_integration = zeros(T, Nλ)
        flux_cont_integration = zeros(T, Nλ)
        cfunc_flux_integration = zeros(T, Natm - 1, Nλ)
        cfunc_flux_cont_integration = zeros(T, Natm - 1, Nλ)

        μ_v_rot = zeros(T, Natm)
        τs_int = zeros(T, Natm, Nλ)
        τs_int_cont = zeros(T, Natm, Nλ)
        cfunc_int = zeros(T, Natm - 1, Nλ)
        cfunc_int_cont = zeros(T, Natm - 1, Nλ)

        @showprogress for i in eachindex(μs_cpu)
            μ_tile = μs_cpu[i]
            μ_v_rot .= z_rot_cpu[i] .* c_ms

            αs_broad_i = convolve_wavelength_axis(λs_korg, αs, μ_v_rot, σ_v)
            calc_tau_cpu!(μ_tile, zs, αs_broad_i, τs_int)
            calc_intensity_cfunc_cpu!(cfunc_int, Ts, λs_korg, τs_int)
            cfunc_dt_int = cfunc_int .* diff(τs_int, dims=1)
            cfunc_int_i_mac = convolve_rt_macro(λs_korg, cfunc_dt_int, star.ζ, μ_tile)

            αs_cont_broad_i = convolve_wavelength_axis(λs_korg, αs_cont, μ_v_rot, σ_v)
            calc_tau_cpu!(μ_tile, zs, αs_cont_broad_i, τs_int_cont)
            calc_intensity_cfunc_cpu!(cfunc_int_cont, Ts, λs_korg, τs_int_cont)
            cfunc_dt_int_cont = cfunc_int_cont .* diff(τs_int_cont, dims=1)
            cfunc_int_cont_i_mac = convolve_rt_macro(λs_korg, cfunc_dt_int_cont, star.ζ, μ_tile)

            flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
            flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[i]
            cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]
            cfunc_flux_cont_integration .+= cfunc_int_cont_i_mac .* dA_cpu[i]
        end

        cfunc_dt_flux = cfunc_flux_integration
        cfunc_dt_flux_cont = cfunc_flux_cont_integration
    end

    cum_cfunc_flux = cumsum(cfunc_dt_flux, dims=1)
    cum_cfunc_flux ./= maximum(cum_cfunc_flux, dims=1)

    flux_norm = vec(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))

    form_temps = zeros(T, Nλ)
    mid_temps = elav(Ts)
    for i in eachindex(λs_korg)
        xs = view(cum_cfunc_flux, :, i)
        itp = linear_interp(xs, mid_temps)
        form_temps[i] = itp(0.5)
    end

    cont_func = cfunc_dt_flux
    return FormTempResult(λs_korg, flux_norm, form_temps, cont_func)#, αs_broad
end

function _calc_formation_temp_gpu(star::StellarProps, linelist; Δλ::T=0.01, 
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN,
                                  Nϕ::Int=128) where T<:AF
    # get linelist 
    wls = [l.wl * 1e8 for l in linelist]
    λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)

    # get model atmosphere and move to GPU
    atm_gpu = AtmosphereGPU(Korg.interpolate_marcs(star.Teff, star.logg, star.A_X))

    # get the absorption coefficients
    αs = zeros(length(atm_gpu.zs), length(λs_korg))
    αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
    compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, star.A_X)

    # allocate on device
    gpu_mem = GPUMemory(λs_korg, atm_gpu)

    # allocate memory for convolutions
    Nλ = length(λs_korg)
    Natm = size(αs, 1)
    Npad = 100
    cmem = ConvolutionMemory(Nλ, Natm, Npad)
    cmem_mac = ConvolutionMemory(Nλ, Natm - 1, Npad)

    # set microturbulent broadening
    σ_v = CUDA.zeros(T, length(atm_gpu.zs)) .+ star.ξ

    # get the "stationary" flux
    cfunc_flux_struct = calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
    cfunc_dt_flux = cfunc_flux_struct.cfunc_dt

    # same for the continuum
    cfunc_flux_struct_cont = calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v)
    cfunc_dt_flux_cont = cfunc_flux_struct_cont.cfunc_dt

    # convolution or numerical integration
    if convolve
        @assert !isnan(u1)
        @assert !isnan(u2)
        cfunc_dt_flux = convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux, star.vsini, star.ζ, u1, u2)
        cfunc_dt_flux_cont = convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux_cont, star.vsini, star.ζ, u1, u2)
    else # numerical disk integration
        # get stellar grid
        μs_gpu, dA, z_rot, _ = calc_stellar_grid(star.ρstar, star.istar, star.vsini, Nϕ)
        idx = findall(x -> x .> zero(eltype(μs_gpu)), Array(μs_gpu))
        μs_cpu = Array(μs_gpu)[idx]
        dA_cpu = Array(dA)[idx]
        z_rot_cpu = Array(z_rot)[idx]
        if iszero(star.vsini)
            z_rot_cpu .= 0.0
        end

        # allocate on gpu 
        μ_v_rot = CUDA.zeros(T, Natm)
        flux_integration = CUDA.zeros(T, length(λs_korg))
        flux_cont_integration = CUDA.zeros(T, length(λs_korg))
        cfunc_flux_integration = CUDA.zeros(T, Natm - 1, length(λs_korg))
        cfunc_flux_cont_integration = CUDA.zeros(T, Natm - 1, length(λs_korg))

        # loop over cells on grid
        @showprogress for i in eachindex(μs_cpu)
            μ_tile = μs_cpu[i]
            μ_v_rot .= z_rot_cpu[i] .* c_ms

            cfunc_intensity = calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
            tbc = cfunc_intensity.cfunc_dt
            cfunc_int_i_mac = convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, star.ζ, μ_tile)

            cfunc_intensity_cont = calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
            tbc_cont = cfunc_intensity_cont.cfunc_dt
            cfunc_int_cont_i_mac = convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc_cont, star.ζ, μ_tile)

            flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
            flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[i]
            cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]
            cfunc_flux_cont_integration .+= cfunc_int_cont_i_mac .* dA_cpu[i]
        end

        cfunc_dt_flux = cfunc_flux_integration
        cfunc_dt_flux_cont = cfunc_flux_cont_integration
    end

    # get the normalized cumulative contribution function
    cum_cfunc_flux = Array(cumsum(cfunc_dt_flux, dims=1))
    cum_cfunc_flux ./= maximum(cum_cfunc_flux, dims=1)

    # get the normalized flux
    flux_norm = vec(Array(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1)))

    # loop over wavelength
    form_temps = zeros(length(λs_korg))
    mid_temps = elav(atm_gpu.Ts)
    for i in eachindex(λs_korg)
        xs = view(cum_cfunc_flux, :, i)
        itp = linear_interp(xs, mid_temps)
        form_temps[i] = itp(0.5)
    end

    cont_func = Array(cfunc_dt_flux)
    return FormTempResult(λs_korg, flux_norm, form_temps, cont_func)#, Array(αs_gpu)
end
