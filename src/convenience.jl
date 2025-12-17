"""
    calc_formation_temp()

Compute TODO

# Examples
```julia-repl
TODO
```
"""
function calc_formation_temp(star::StellarProps, linelist; use_gpu::Bool=GPU_DEFAULT, kwargs...)
    if use_gpu
        form_temps_flux = _calc_formation_temp_gpu(star, linelist, kwargs...)
    else
        form_temps_flux = _calc_formation_temp_cpu(star, linelist, kwargs...)
    end
    return form_temps_flux
end

function _calc_formation_temp_cpu(star::StellarProps, linelist; Δλ::T=0.01, convolve::Bool=false) where T<:AF


    return nothing
end

function _calc_formation_temp_gpu(star::StellarProps, linelist; Δλ::T=0.01, 
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN) where T<:AF
    # get linelist 
    wls = [l.wl * 1e8 for l in linelist]
    λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)

    # get model atmosphere
    marcs_atm = Korg.interpolate_marcs(star.Teff, star.logg, star.A_X)
    τ_500 = Korg.get_tau_5000s(marcs_atm)
    zs = Korg.get_zs(marcs_atm)
    Ts = Korg.get_temps(marcs_atm)
    ne = Korg.get_electron_number_densities(marcs_atm)
    nd = Korg.get_number_densities(marcs_atm)

    # move stuff to GPU
    atm_gpu = AtmosphereGPU(marcs_atm)
    zs = atm_gpu.zs
    Ts = atm_gpu.Ts
    τ5000 = atm_gpu.τs

    # get the absorption coefficients
    αs = zeros(length(atm_gpu.zs), length(λs_korg))
    αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, star.A_X)

    # allocate on device
    gpu_mem = GPUMemory(λs_korg, atm_gpu)

    # allocate memory for convolutions
    Nλ = length(λs_korg)
    Natm = size(αs, 1)
    Npad = 100
    cmem = ConvolutionMemory(Nλ, Natm, Npad)

    # set microturbulent broadening
    σ_v = CUDA.zeros(Float64, length(zs)) .+ star.ξ

    # if/else block for convolution or integration
    if convolve # convolution
        @assert !isnan(u1)
        @assert !isnan(u2)

        # get the "stationary" flux
        cfunc_flux_struct = calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
        # cfunc_dt_flux_stationary = Array(cfunc_flux_struct.cfunc_dt)
        cfunc_dt_flux_stationary = cfunc_flux_struct.cfunc_dt

        # same for the continuum
        cfunc_flux_struct_cont = calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v)
        # cfunc_dt_flux_stationary_cont = Array(cfunc_flux_struct_cont.cfunc_dt)
        cfunc_dt_flux_stationary_cont = cfunc_flux_struct_cont.cfunc_dt

        # convolve with hirano kernel
        cfunc_dt_flux_convolution = convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux_stationary, star.vsini, star.ζ, u1, u2)
        cfunc_dt_flux_convolution_cont = convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux_stationary_cont, star.vsini, star.ζ, u1, u2)

        # get the normalized cumulative contribution function 
        cum_cfunc_flux_convolution = Array(cumsum(cfunc_dt_flux_convolution, dims=1))
        cum_cfunc_flux_convolution ./= maximum(cum_cfunc_flux_convolution, dims=1)

        # get the normalized flux
        flux_convolution_norm = sum(cfunc_dt_flux_convolution, dims=1) ./ sum(cfunc_dt_flux_convolution_cont, dims=1)
        flux_convolution_norm = Array(flux_convolution_norm)
    
        # loop over wavelength
        form_temp_convolution = zeros(length(λs_korg))
        for i in eachindex(λs_korg)
            xs = view(cum_cfunc_flux_convolution, :, i)
            itp = FT.linear_interp(xs, elav(Ts))
            form_temp_convolution[i] = itp(0.5)
        end

        # create object for return
        out = FormTempResult(flux_convolution_norm, form_temp_convolution, cont_func)

        return out 
    else # integration 



        return nothing
    end
    return nothing
end

