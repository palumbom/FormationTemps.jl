"""
    calc_formation_temp(star, linelist; use_gpu=GPU_DEFAULT, Δλ=0.01, convolve=false,
                        minλ=NaN, maxλ=NaN, u1=NaN, u2=NaN, Nϕ=128,
                        showprogress=true, kwargs...)

Compute flux formation temperatures, normalized flux, and flux contribution function for a given `star` and `linelist`.

The wavelength grid is built from the line list (`wl * 1e8`) with padding and step `Δλ`.
Use `minλ`/`maxλ` (Angstrom) to override the default bounds (first/last line ± 2 A).

Returns a `FormTempResult` with fields:
- `wavs`: wavelength grid (Angstrom).
- `flux`: normalized flux (`sum(cfunc_dt_flux) / sum(cfunc_dt_flux_cont)`).
- `form_temps`: formation temperature defined at 50% of the cumulative flux contribution.
- `cont_func`: contribution function, shape `(Natm - 1, Nλ)`.
- `atmosphere`: atmosphere structure used for the calculation.

If `convolve=true`, applies Hirano rotation + macroturbulent convolution using limb-darkening
coefficients `u1` and `u2`. Otherwise, performs numerical disk integration using `Nϕ` latitude
bins. Set `use_gpu=true` to use the GPU implementation when available.
Set `showprogress=false` to suppress the progress bar during disk integration.

The CPU disk integration path (`use_gpu=false, convolve=false`) is parallelized across tiles
using `Threads.@threads`. Launch Julia with multiple threads (e.g. `julia -t auto`) to benefit.
FFTW internal threading is disabled during the tile loop to avoid contention. See
[Parallelization](parallelization.md) for details.

# Examples
```julia-repl
star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0)
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[1:500]
result = calc_formation_temp(star, linelist; Δλ=0.01, convolve=true, u1=0.43, u2=0.31)
```
"""
function calc_formation_temp(star::StellarProps, linelist; use_gpu::Bool=GPU_DEFAULT,
                             Δλ::T=0.01, convolve::Bool=false,
                             minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                             u1::T=NaN, u2::T=NaN, Nϕ::Int=128,
                             kwargs...) where T<:AF
    if use_gpu
        form_temps_flux = _calc_formation_temp_gpu(star, linelist; Δλ=Δλ,
                                                   minλ, maxλ, buffer, convolve=convolve,
                                                   u1=u1, u2=u2, Nϕ=Nϕ, kwargs...)
    else
        form_temps_flux = _calc_formation_temp_cpu(star, linelist; Δλ=Δλ,
                                                   minλ, maxλ, buffer, convolve=convolve,
                                                   u1=u1, u2=u2, Nϕ=Nϕ, kwargs...)
    end
    return form_temps_flux
end

function _calc_formation_temp_cpu(star::StellarProps, linelist; Δλ::T=0.01,
                                  minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN,
                                  Nϕ::Int=128, showprogress::Bool=true,
                                  kwargs...) where T<:AF
    # get linelist
    wls = [l.wl * 1e8 for l in linelist]
    minλ = isnan(minλ) ? first(wls) - buffer : minλ
    maxλ = isnan(maxλ) ? last(wls) + buffer : maxλ
    λs_korg = range(minλ, maxλ, step=Δλ)

    # get model atmosphere
    atm_cpu = AtmosphereCPU(Korg.interpolate_marcs(star.Teff, star.logg, star.A_X))
    zs = atm_cpu.zs
    Ts = atm_cpu.Ts

    # get the absorption coefficients; α_ref filled inline during the chemistry loop
    # (reuses nₑ, n_dict already computed per layer — zero extra solver calls)
    Natm = length(zs)
    Nλ = length(λs_korg)
    αs = zeros(T, Natm, Nλ)
    αs_cont = zeros(T, Natm, Nλ)
    α_ref = zeros(T, Natm)
    compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg),
                   linelist, atm_cpu, star.A_X;
                   α_ref_out=α_ref, vmic_ref_cms=star.ξ * 100.0, kwargs...)

    # set microturbulent broadening
    σ_v = fill(star.ξ, Natm)
    μ_v = zeros(T, Natm)

    # convolve absorption coefficients with microturbulence
    αs_broad = convolve_wavelength_axis(λs_korg, αs, μ_v, σ_v)
    αs_cont_broad = convolve_wavelength_axis(λs_korg, αs_cont, μ_v, σ_v)

    # dispatch between anchored (preferred) and Bezier (fallback when tau_ref unavailable)
    if isempty(atm_cpu.τs)
        _calc_tau_cpu! = (μ_i, αs_in, τs_out) -> calc_tau_bezier_cpu!(μ_i, zs, αs_in, τs_out)
    else
        _calc_tau_cpu! = (μ_i, αs_in, τs_out) -> calc_tau_anchored_cpu!(μ_i, atm_cpu.τs, α_ref, αs_in, τs_out)
    end

    τs = zeros(T, Natm, Nλ)
    τs_cont = zeros(T, Natm, Nλ)
    _calc_tau_cpu!(one(T), αs_broad, τs)
    _calc_tau_cpu!(one(T), αs_cont_broad, τs_cont)

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
        if any(map(!isnan, (u1, u2)))
            @warn "Prescribed limb darkening coefficients are not used in integration method!"
        end

        # get stellar grid
        μs, dA, z_rot = calc_stellar_grid_cpu(star.ρstar, star.istar, star.vsini, Nϕ)
        idx = findall(x -> x .> zero(T), μs)
        μs_cpu = μs[idx]
        dA_cpu = dA[idx]
        z_rot_cpu = z_rot[idx]
        if iszero(star.vsini)
            z_rot_cpu .= 0.0
        end

        # disable FFTW internal threading; we parallelize at the tile level
        prev_fftw_threads = FFTW.get_num_threads()
        FFTW.set_num_threads(1)

        # allocate per-thread workspaces with pre-computed FFT plans
        workspaces = [CPUTileWorkspace(T, Natm, Nλ) for _ in 1:Threads.maxthreadid()]

        # threaded tile loop (in-place convolutions eliminate per-tile allocations)
        Threads.@threads :static for i in eachindex(μs_cpu)
            ws = workspaces[Threads.threadid()]
            μ_tile = μs_cpu[i]
            dA_i = dA_cpu[i]
            ws.μ_v_buf .= z_rot_cpu[i] * c_ms

            # total absorption → macro_out → accumulate immediately
            _convolve_micro_inplace!(ws.αs_broad, λs_korg, αs, ws.μ_v_buf, σ_v, ws)
            _calc_tau_cpu!(μ_tile, ws.αs_broad, ws.τs_int)
            calc_intensity_cfunc_cpu!(ws.cfunc_int, Ts, λs_korg, ws.τs_int)
            @views ws.cfunc_dt_int .= ws.cfunc_int .* (ws.τs_int[2:end, :] .- ws.τs_int[1:end-1, :])
            _convolve_macro_inplace!(ws.macro_out, λs_korg, ws.cfunc_dt_int, star.ζ, μ_tile, ws)
            ws.cfunc_flux_acc .+= ws.macro_out .* dA_i

            # continuum absorption → macro_out → accumulate immediately
            _convolve_micro_inplace!(ws.αs_cont_broad, λs_korg, αs_cont, ws.μ_v_buf, σ_v, ws)
            _calc_tau_cpu!(μ_tile, ws.αs_cont_broad, ws.τs_int_cont)
            calc_intensity_cfunc_cpu!(ws.cfunc_int_cont, Ts, λs_korg, ws.τs_int_cont)
            @views ws.cfunc_dt_int_cont .= ws.cfunc_int_cont .* (ws.τs_int_cont[2:end, :] .- ws.τs_int_cont[1:end-1, :])
            _convolve_macro_inplace!(ws.macro_out, λs_korg, ws.cfunc_dt_int_cont, star.ζ, μ_tile, ws)
            ws.cfunc_flux_cont_acc .+= ws.macro_out .* dA_i
        end

        # reduce per-thread accumulators
        cfunc_dt_flux = sum(ws.cfunc_flux_acc for ws in workspaces)
        cfunc_dt_flux_cont = sum(ws.cfunc_flux_cont_acc for ws in workspaces)

        # restore FFTW threading state
        FFTW.set_num_threads(prev_fftw_threads)
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
    return FormTempResult(λs_korg, flux_norm, form_temps, cont_func, atm_cpu)
end

function _calc_formation_temp_gpu(star::StellarProps, linelist; Δλ::T=0.01,
                                  minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN,
                                  Nϕ::Int=128, showprogress::Bool=true,
                                  kwargs...) where T<:AF
    # get linelist
    wls = [l.wl * 1e8 for l in linelist]
    minλ = isnan(minλ) ? first(wls) - buffer : minλ
    maxλ = isnan(maxλ) ? last(wls) + buffer : maxλ
    λs_korg = range(minλ, maxλ, step=Δλ)

    # get model atmosphere and move to GPU
    atm_gpu = AtmosphereGPU(Korg.interpolate_marcs(star.Teff, star.logg, star.A_X))

    # get the absorption coefficients; α_ref filled inline during the chemistry loop
    # (reuses nₑ, n_dict already computed per layer — zero extra solver calls)
    Natm = length(atm_gpu.zs)
    αs = zeros(Natm, length(λs_korg))
    αs_cont = zeros(Natm, length(λs_korg))
    α_ref = zeros(Natm)
    compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg),
                   linelist, atm_gpu, star.A_X;
                   α_ref_out=α_ref, vmic_ref_cms=star.ξ * 100.0, kwargs...)

    # allocate on device; use anchored τ when tau_ref is available, Bezier otherwise
    _make_gpu_mem = if isempty(atm_gpu.τs)
        () -> GPUMemory(λs_korg, atm_gpu)
    else
        () -> GPUMemory(λs_korg, atm_gpu, α_ref)
    end
    gpu_mem = _make_gpu_mem()
    gpu_mem_cont = _make_gpu_mem()  # separate buffers for dual-stream continuum

    # allocate memory for convolutions
    Nλ = length(λs_korg)
    Natm = size(αs, 1)
    Npad = 512
    cmem = ConvolutionMemory(Nλ, Natm, Npad)
    cmem_cont = ConvolutionMemory(Nλ, Natm, Npad)
    cmem_mac = MacroConvolutionMemory(Nλ, Natm - 1, Npad)
    cmem_mac_cont = MacroConvolutionMemory(Nλ, Natm - 1, Npad)

    # set microturbulent broadening
    σ_v = CUDA.zeros(T, length(atm_gpu.zs)) .+ star.ξ

    # get the "stationary" flux
    cfunc_flux_struct = calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
    cfunc_dt_flux = cfunc_flux_struct.cfunc_dt

    # same for the continuum (separate gpu_mem_cont + cmem_cont for independent buffers)
    cfunc_flux_struct_cont = calc_flux_quantities(αs_cont, atm_gpu, gpu_mem_cont, cmem_cont, σ_v)
    cfunc_dt_flux_cont = cfunc_flux_struct_cont.cfunc_dt

    # convolution or numerical integration
    if convolve
        @assert !isnan(u1)
        @assert !isnan(u2)
        cfunc_dt_flux = copy(convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux, star.vsini, star.ζ, u1, u2))
        cfunc_dt_flux_cont = copy(convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux_cont, star.vsini, star.ζ, u1, u2))
    else # numerical disk integration
        if any(map(!isnan, (u1, u2)))
            @warn "Prescribed limb darkening coefficients are not used in integration method!"
        end

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
        λs_gpu = CuArray{T}(collect(λs_korg))
        μ_v_rot = CUDA.zeros(T, Natm)
        μ_v_rot_cont = CUDA.zeros(T, Natm)  # separate buffer for continuum stream
        flux_integration = CUDA.zeros(T, length(λs_korg))
        flux_cont_integration = CUDA.zeros(T, length(λs_korg))
        cfunc_flux_integration = CUDA.zeros(T, Natm - 1, length(λs_korg))
        cfunc_flux_cont_integration = CUDA.zeros(T, Natm - 1, length(λs_korg))

        # create two CUDA streams for overlapping total/continuum work
        stream_total = CuStream()
        stream_cont = CuStream()

        # prime the signal FFT cache: first tile computes + caches, rest reuse
        cmem.signal_cached = false
        cmem_cont.signal_cached = false
        μ_v_rot .= z_rot_cpu[1] .* c_ms
        calc_intensity_quantities_inplace!(αs, atm_gpu, gpu_mem, cmem, μs_cpu[1], μ_v_rot, σ_v)
        cmem.signal_cached = true
        calc_intensity_quantities_inplace!(αs_cont, atm_gpu, gpu_mem_cont, cmem_cont, μs_cpu[1], μ_v_rot, σ_v)
        cmem_cont.signal_cached = true

        # precompute macro kernel FFTs for unique μ values
        macro_kernel_cache = Dict{T, CuVector{Complex{T}}}()
        if !iszero(star.ζ)
            unique_μ_vals = unique(μs_cpu)
            for μ_val in unique_μ_vals
                macro_kernel_cache[μ_val] = precompute_rt_macro_kernel_ft(cmem_mac, λs_korg, star.ζ, μ_val)
            end
        end

        # loop over cells on grid (total and continuum on separate streams)
        prog = Progress(length(μs_cpu); enabled=showprogress)
        for i in eachindex(μs_cpu)
            μ_tile = μs_cpu[i]
            z_rot_v = z_rot_cpu[i] * c_ms

            # total absorption on stream_total
            CUDA.stream!(stream_total) do
                μ_v_rot .= z_rot_v
                cfunc_intensity = calc_intensity_quantities_inplace!(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
                if iszero(star.ζ)
                    src_total = cfunc_intensity.cfunc_dt
                else
                    src_total = convolve_rt_macro_gpu_cached(cmem_mac, cfunc_intensity.cfunc_dt,
                                                             macro_kernel_cache[μ_tile])
                end
                accumulate_tile!(flux_integration, cfunc_flux_integration, src_total, dA_cpu[i])
            end

            # continuum absorption on stream_cont (overlaps with total)
            CUDA.stream!(stream_cont) do
                μ_v_rot_cont .= z_rot_v
                cfunc_intensity_cont = calc_intensity_quantities_inplace!(αs_cont, atm_gpu, gpu_mem_cont, cmem_cont, μ_tile, μ_v_rot_cont, σ_v)
                if iszero(star.ζ)
                    src_cont = cfunc_intensity_cont.cfunc_dt
                else
                    src_cont = convolve_rt_macro_gpu_cached(cmem_mac_cont, cfunc_intensity_cont.cfunc_dt,
                                                            macro_kernel_cache[μ_tile])
                end
                accumulate_tile!(flux_cont_integration, cfunc_flux_cont_integration, src_cont, dA_cpu[i])
            end

            # sync both streams before next tile (buffers reused)
            CUDA.synchronize(stream_total)
            CUDA.synchronize(stream_cont)
            next!(prog)
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
    return FormTempResult(λs_korg, flux_norm, form_temps, cont_func, atm_gpu)
end
