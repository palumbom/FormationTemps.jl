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
                             Δλ::T=0.01, gpu_precision::Type{<:AF}=Float64,
                             convolve::Bool=false,
                             minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                             u1::T=NaN, u2::T=NaN, Nϕ::Int=128,
                             kwargs...) where T<:AF
    if use_gpu
        form_temps_flux = _calc_formation_temp_gpu(star, linelist; Δλ=Δλ,
                                                   gpu_precision=gpu_precision,
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
                   α_ref_out=α_ref, kwargs...)

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
                                  gpu_precision::Type{<:AF}=Float64,
                                  minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN,
                                  Nϕ::Int=128, showprogress::Bool=true,
                                  kwargs...) where T<:AF
    G = gpu_precision  # shorthand for GPU float type

    # get linelist
    wls = [l.wl * 1e8 for l in linelist]
    minλ = isnan(minλ) ? first(wls) - buffer : minλ
    maxλ = isnan(maxλ) ? last(wls) + buffer : maxλ
    λs_korg = range(minλ, maxλ, step=Δλ)

    # build Float64 atmosphere for Korg (which requires Float64 internally)
    korg_atm = Korg.interpolate_marcs(star.Teff, star.logg, star.A_X)
    atm_f64 = AtmosphereGPU(korg_atm; T=Float64)

    # absorption coefficients at Float64 (Korg requirement)
    Natm = length(atm_f64.zs)
    Nλ = length(λs_korg)
    αs = zeros(Float64, Natm, Nλ)
    αs_cont = zeros(Float64, Natm, Nλ)
    α_ref = zeros(Float64, Natm)
    compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg),
                   linelist, atm_f64, star.A_X;
                   α_ref_out=α_ref, kwargs...)

    # convert to GPU precision; rebuild atmosphere at target type
    if G !== Float64
        atm_gpu = AtmosphereGPU(korg_atm; T=G)
        αs      = G.(αs)
        αs_cont = G.(αs_cont)
        α_ref   = G.(α_ref)
    else
        atm_gpu = atm_f64
    end

    # wavelength array at GPU precision for GPUMemory (which infers T from λs_cpu)
    λs_G = G.(collect(λs_korg))

    # allocate on device; use anchored τ when tau_ref is available, Bezier otherwise
    _make_gpu_mem = if isempty(atm_gpu.τs)
        () -> GPUMemory(λs_G, atm_gpu)
    else
        () -> GPUMemory(λs_G, atm_gpu, α_ref)
    end
    gpu_mem = _make_gpu_mem()
    gpu_mem_cont = _make_gpu_mem()  # separate buffers for dual-stream continuum

    # allocate memory for convolutions
    Natm = size(αs, 1)
    Npad = 512
    cmem = ConvolutionMemory(Nλ, Natm, Npad; T=G)
    cmem_cont = ConvolutionMemory(Nλ, Natm, Npad; T=G)
    cmem_mac = MacroConvolutionMemory(Nλ, Natm - 1, Npad; T=G)
    cmem_mac_cont = MacroConvolutionMemory(Nλ, Natm - 1, Npad; T=G)

    # set microturbulent broadening
    σ_v = CUDA.zeros(G, length(atm_gpu.zs)) .+ G(star.ξ)

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
        cfunc_dt_flux = copy(convolve_hirano_rotmacro_gpu(cmem_mac, λs_G, cfunc_dt_flux, G(star.vsini), G(star.ζ), G(u1), G(u2)))
        cfunc_dt_flux_cont = copy(convolve_hirano_rotmacro_gpu(cmem_mac, λs_G, cfunc_dt_flux_cont, G(star.vsini), G(star.ζ), G(u1), G(u2)))
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
        Ntiles = length(μs_cpu)
        Natm1 = Natm - 1

        # choose batch size: estimate per-tile memory, stay within 70% of free GPU RAM
        use_anchored = gpu_mem.use_anchored
        nfreq = fld(next_fft_friendly_len(Nλ + Npad), 2) + 1
        L_est = next_fft_friendly_len(Nλ + Npad)
        bytes_per_tile = Natm * (L_est * sizeof(G) + nfreq * sizeof(Complex{G}) * 2)
        bytes_work = Natm * Nλ * sizeof(G) + Natm1 * Nλ * sizeof(G)
        # shared signal buffers (ys_gpu + signal_gpu + signal_ft_gpu), paid once per stream
        bytes_shared = Natm * (Nλ * sizeof(G) + L_est * sizeof(G) + nfreq * sizeof(Complex{G}))
        bytes_per_tile_total = 2 * (bytes_per_tile + bytes_work)  # dual-stream, scales with B
        bytes_fixed = 2 * bytes_shared                             # dual-stream, paid once
        avail = CUDA.free_memory()
        budget = Int(floor(avail * 0.5))
        B_mem = max(1, (budget - bytes_fixed) ÷ bytes_per_tile_total)
        B = min(64, Ntiles, B_mem)

        # free GPU memory from prior tests/computations before batch allocation
        GC.gc()
        CUDA.reclaim()

        # allocate batched convolution memories (dual-stream)
        bcmem      = BatchedMicroConvMem(Nλ, Natm, B, Npad; T=G)
        bcmem_cont = BatchedMicroConvMem(Nλ, Natm, B, Npad; T=G)

        # prime signal FFT caches: the forward FFT of the padded absorption signal
        # is tile-independent (only the Doppler filter changes per tile). The priming
        # call writes a throw-away Doppler filter and convolution product for Bcur=1,
        # which are overwritten by the first real batch — only signal_ft_gpu persists.
        μ_v_prime = CUDA.zeros(G, Natm)
        bcmem.signal_cached = false
        convolve_wavelength_axis_batched!(bcmem, λs_G, αs, μ_v_prime, σ_v, 1)
        bcmem.signal_cached = true
        bcmem_cont.signal_cached = false
        convolve_wavelength_axis_batched!(bcmem_cont, λs_G, αs_cont, μ_v_prime, σ_v, 1)
        bcmem_cont.signal_cached = true

        # batched working arrays (dual-stream)
        τs_batch      = CUDA.zeros(G, B * Natm, Nλ)
        τs_batch_cont = CUDA.zeros(G, B * Natm, Nλ)
        cfdt_batch      = CUDA.zeros(G, B * Natm1, Nλ)
        cfdt_batch_cont = CUDA.zeros(G, B * Natm1, Nλ)
        μ_tiles_gpu   = CUDA.zeros(G, B)
        dA_tiles_gpu  = CUDA.zeros(G, B)
        μ_v_batch     = CUDA.zeros(G, B * Natm)
        μ_v_batch_cont = CUDA.zeros(G, B * Natm)

        # Bezier work arrays (only allocated when needed)
        if !use_anchored
            ds_batch      = CUDA.zeros(G, B * Natm)
            alphaC_batch  = CUDA.zeros(G, B * Natm)
            ds_batch_cont = CUDA.zeros(G, B * Natm)
            alphaC_batch_cont = CUDA.zeros(G, B * Natm)
        end

        # accumulators
        flux_integration = CUDA.zeros(G, Nλ)
        flux_cont_integration = CUDA.zeros(G, Nλ)
        cfunc_flux_integration = CUDA.zeros(G, Natm1, Nλ)
        cfunc_flux_cont_integration = CUDA.zeros(G, Natm1, Nλ)

        # tau integration constants (shared across tiles)
        log_τ_ref    = gpu_mem.log_τ_ref
        ifactor_base = gpu_mem.ifactor_base

        # precompute macro kernel FFTs for unique μ values
        macro_kernel_cache = Dict{G, CuVector{Complex{G}}}()
        if !iszero(star.ζ)
            unique_μ_vals = unique(μs_cpu)
            for μ_val in unique_μ_vals
                macro_kernel_cache[G(μ_val)] = precompute_rt_macro_kernel_ft(cmem_mac, λs_G, G(star.ζ), G(μ_val))
            end
        end

        # CUDA streams for overlapping total/continuum
        stream_total = CuStream()
        stream_cont = CuStream()

        # CPU staging buffers for batch parameter upload
        μ_tiles_cpu  = zeros(G, B)
        dA_tiles_cpu = zeros(G, B)
        μ_v_batch_cpu = zeros(G, B * Natm)

        # batched tile loop
        prog = Progress(Ntiles; enabled=showprogress)
        for batch_start in 1:B:Ntiles
            batch_end = min(batch_start + B - 1, Ntiles)
            Bcur = batch_end - batch_start + 1

            # fill batch parameters on CPU, then upload
            for bi in 1:Bcur
                i = batch_start + bi - 1
                μ_tiles_cpu[bi] = μs_cpu[i]
                dA_tiles_cpu[bi] = dA_cpu[i]
                v = z_rot_cpu[i] * c_ms
                for k in 1:Natm
                    μ_v_batch_cpu[(bi - 1) * Natm + k] = v
                end
            end
            copyto!(μ_tiles_gpu, 1, μ_tiles_cpu, 1, Bcur)
            copyto!(dA_tiles_gpu, 1, dA_tiles_cpu, 1, Bcur)
            copyto!(μ_v_batch, 1, μ_v_batch_cpu, 1, Bcur * Natm)
            copyto!(μ_v_batch_cont, 1, μ_v_batch_cpu, 1, Bcur * Natm)
            # ensure uploads visible to both worker streams (default stream syncs under
            # legacy semantics, but explicit sync is safer)
            CUDA.synchronize()

            # total absorption on stream_total
            CUDA.stream!(stream_total) do
                αs_conv = convolve_wavelength_axis_batched!(bcmem, λs_G, αs,
                    μ_v_batch, σ_v, Bcur)
                if use_anchored
                    calc_tau_anchored_batched!(μ_tiles_gpu, log_τ_ref, ifactor_base,
                        αs_conv, τs_batch, Natm, Bcur)
                else
                    calc_tau_bezier_batched!(μ_tiles_gpu, atm_gpu.zs_gpu,
                        αs_conv, τs_batch, ds_batch, alphaC_batch, Natm, Bcur)
                end
                calc_intensity_cfunc_dt_batched!(cfdt_batch, τs_batch,
                    atm_gpu.Ts_gpu, gpu_mem.λs, Natm, Bcur)

                if iszero(star.ζ)
                    accumulate_batch!(flux_integration, cfunc_flux_integration,
                        cfdt_batch, dA_tiles_gpu, Natm1, Bcur)
                else
                    # per-tile macro convolution (Phase 5 will batch this);
                    # accumulate_tile! completes before the next convolve_rt_macro_gpu_cached
                    # overwrites cmem_mac.out_gpu (same-stream serialization)
                    for bi in 1:Bcur
                        i = batch_start + bi - 1
                        tile_cfdt = @view cfdt_batch[(bi-1)*Natm1+1 : bi*Natm1, :]
                        src = convolve_rt_macro_gpu_cached(cmem_mac, tile_cfdt,
                                                           macro_kernel_cache[G(μs_cpu[i])])
                        accumulate_tile!(flux_integration, cfunc_flux_integration,
                            src, G(dA_cpu[i]))
                    end
                end
            end

            # continuum absorption on stream_cont
            CUDA.stream!(stream_cont) do
                αs_conv_c = convolve_wavelength_axis_batched!(bcmem_cont, λs_G, αs_cont,
                    μ_v_batch_cont, σ_v, Bcur)
                if use_anchored
                    calc_tau_anchored_batched!(μ_tiles_gpu, log_τ_ref, ifactor_base,
                        αs_conv_c, τs_batch_cont, Natm, Bcur)
                else
                    calc_tau_bezier_batched!(μ_tiles_gpu, atm_gpu.zs_gpu,
                        αs_conv_c, τs_batch_cont, ds_batch_cont, alphaC_batch_cont, Natm, Bcur)
                end
                calc_intensity_cfunc_dt_batched!(cfdt_batch_cont, τs_batch_cont,
                    atm_gpu.Ts_gpu, gpu_mem.λs, Natm, Bcur)

                if iszero(star.ζ)
                    accumulate_batch!(flux_cont_integration, cfunc_flux_cont_integration,
                        cfdt_batch_cont, dA_tiles_gpu, Natm1, Bcur)
                else
                    for bi in 1:Bcur
                        i = batch_start + bi - 1
                        tile_cfdt_c = @view cfdt_batch_cont[(bi-1)*Natm1+1 : bi*Natm1, :]
                        src_c = convolve_rt_macro_gpu_cached(cmem_mac_cont, tile_cfdt_c,
                                                              macro_kernel_cache[G(μs_cpu[i])])
                        accumulate_tile!(flux_cont_integration, cfunc_flux_cont_integration,
                            src_c, G(dA_cpu[i]))
                    end
                end
            end

            # sync both streams before next batch
            CUDA.synchronize(stream_total)
            CUDA.synchronize(stream_cont)
            for _ in 1:Bcur; next!(prog); end
        end

        cfunc_dt_flux = cfunc_flux_integration
        cfunc_dt_flux_cont = cfunc_flux_cont_integration
    end

    # get the normalized cumulative contribution function
    cum_cfunc_flux = Array(cumsum(cfunc_dt_flux, dims=1))
    cum_cfunc_flux ./= maximum(cum_cfunc_flux, dims=1)

    # get the normalized flux
    flux_norm = G.(vec(Array(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))))

    # loop over wavelength
    form_temps = zeros(G, length(λs_korg))
    mid_temps = elav(atm_gpu.Ts)
    for i in eachindex(λs_korg)
        xs = view(cum_cfunc_flux, :, i)
        itp = linear_interp(xs, mid_temps)
        form_temps[i] = G(itp(G(0.5)))
    end

    cont_func = Array(cfunc_dt_flux)
    return FormTempResult(G.(collect(λs_korg)), flux_norm, form_temps, cont_func, atm_gpu)
end
