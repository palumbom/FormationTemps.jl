"""
    calc_formation_temp()

Compute TODO

# Examples
```julia-repl
TODO
```
"""
function calc_formation_temp(μs, linelist, Teff, logg, A_X, v_mic; use_gpu::Bool=GPU_DEFAULT)
    if use_gpu
        form_temps_intensity, form_temps_flux = _calc_formation_temp_gpu(μs, linelist, Teff, logg, A_X, v_mic)
    else
        form_temps_intensity, form_temps_flux = _calc_formation_temp_cpu()
    end
    return form_temps_intensity, form_temps_flux
end

function _calc_formation_temp_cpu()


    return nothing
end

function _calc_formation_temp_gpu(μs, linelist, Teff, logg, A_X, v_mic)
    # get linelist 
    wls = [l.wl * 1e8 for l in linelist]
    λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=0.005)

    # get model atmosphere
    marcs_atm = Korg.interpolate_marcs(Teff, logg, A_X)
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
    compute_alpha!(αs, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

    # allocate on device
    gpu_mem = GPUMemory(λs_korg, atm_gpu)

    # allocate memory for convolutions
    Nλ = length(λs_korg)
    Natm = size(αs, 1)
    Npad = 100
    cmem = ConvolutionMemory(Nλ, Natm, Npad)

    # loop over mus for intensity
    μ_v = CUDA.zeros(Float64, length(zs))
    σ_v = CUDA.zeros(Float64, length(zs)) .+ v_mic
    cfunc_intensities = zeros(length(zs)-1, length(λs_korg), length(μs))
    intensities = zeros(length(λs_korg), length(μs))

    for i in eachindex(μs)
        cfunc_intensities[:,:,i] .= calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
        intensities[:,i] .= dropdims(sum(view(cfunc_intensities,:,:,i), dims=1), dims=1)
    end
    
    # do flux
    cfunc_flux = calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v)
    flux = 2π .* dropdims(sum(cfunc_flux, dims=1), dims=1)

    # now get cumulative contribution functions
    cum_cfuncs_norm = cumsum(cfunc_intensities, dims=1) 
    cum_cfuncs_norm ./= maximum(cum_cfuncs_norm, dims=1)
    cum_cfunc_flux_norm = cumsum(cfunc_flux, dims=1) 
    cum_cfunc_flux_norm ./= maximum(cum_cfunc_flux_norm, dims=1)

    # now compute the formation temperature
    form_temps_intensity = zeros(length(λs_korg), length(μs))
    form_temps_flux = zeros(length(λs_korg))

    for i in eachindex(λs_korg)
        local xs = view(cum_cfunc_flux_norm, :, i)
        local itp = linear_interp(xs, elav(Ts))
        form_temps_flux[i] = itp(0.5)

        for j in eachindex(μs)
            local xs = view(cum_cfuncs_norm, :, i, j)
            local itp = linear_interp(xs, elav(Ts))
            form_temps_intensity[i,j] = itp(0.5)
        end
    end
    return form_temps_intensity, form_temps_flux
end

