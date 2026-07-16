# function barrier: returns a single concrete closure type per branch
function _make_tau_integrator(atm_cpu, zs, α_ref)
    if isempty(atm_cpu.τs)
        return (μ_i, αs_in, τs_out) -> calc_tau_bezier_cpu!(μ_i, zs, αs_in, τs_out)
    else
        return (μ_i, αs_in, τs_out) -> calc_tau_anchored_cpu!(μ_i, atm_cpu.τs, α_ref, αs_in, τs_out)
    end
end

"""
    calc_formation_temp(star, linelist; use_gpu=GPU_DEFAULT, Δλ=0.01,
                        gpu_precision=Float64, convolve=false,
                        minλ=NaN, maxλ=NaN, u1=NaN, u2=NaN, Nϕ=128,
                        showprogress=true, kwargs...)

Compute flux formation temperatures, normalized flux, and flux contribution function for a given `star` and `linelist`.

The wavelength grid is built from the line list (vacuum cm → Å) with padding and step `Δλ`.
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

Pass `gpu_precision=Float32` to run GPU computations at single precision. Absorption
coefficients are always computed at Float64 (a Korg requirement) and converted to the target
precision before GPU upload. This roughly halves GPU memory usage and can improve throughput
on consumer GPUs. The default is `Float64`.

The CPU disk integration path (`use_gpu=false, convolve=false`) is parallelized across tiles
using `Threads.@threads`. Launch Julia with multiple threads (e.g. `julia -t auto`) to benefit.
FFTW internal threading is disabled during the tile loop to avoid contention. See
[Parallelization](parallelization.md) for details.

# Examples
```julia-repl
star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0, vsini=2100.0)
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[1:500]
result = calc_formation_temp(star, linelist; Δλ=0.01, convolve=true, u1=0.43, u2=0.31)

# Float32 GPU:
result32 = calc_formation_temp(star, linelist; Δλ=0.01, gpu_precision=Float32)
```
"""
function calc_formation_temp(star::StellarProps, linelist; use_gpu::Bool=GPU_DEFAULT,
                             Δλ::T=0.01, gpu_precision::Type{<:AF}=Float64,
                             method::Union{Nothing,Symbol}=nothing, convolve::Bool=false,
                             minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                             u1::T=NaN, u2::T=NaN, Nϕ::Int=128,
                             kwargs...) where T<:AF
    # resolve the disk-integration method. `method` (preferred) selects among
    # :disk (explicit tiling), :hirano (analytic convolution), :quadrature
    # (ring-by-ring μ-quadrature). `convolve` is the deprecated boolean alias.
    resolved = if method === nothing
        convolve ? :hirano : :disk
    else
        method
    end
    @assert resolved in (:disk, :hirano, :quadrature) "method must be :disk, :hirano, or :quadrature (got :$resolved)"

    if resolved === :quadrature
        if use_gpu
            return _calc_formation_temp_quadrature_gpu(star, linelist; Δλ=Δλ,
                                                       gpu_precision=gpu_precision,
                                                       minλ, maxλ, buffer, kwargs...)
        else
            return _calc_formation_temp_quadrature_cpu(star, linelist; Δλ=Δλ,
                                                       minλ, maxλ, buffer, kwargs...)
        end
    end

    conv = (resolved === :hirano)
    if use_gpu
        return _calc_formation_temp_gpu(star, linelist; Δλ=Δλ,
                                        gpu_precision=gpu_precision,
                                        minλ, maxλ, buffer, convolve=conv,
                                        u1=u1, u2=u2, Nϕ=Nϕ, kwargs...)
    else
        return _calc_formation_temp_cpu(star, linelist; Δλ=Δλ,
                                        minλ, maxλ, buffer, convolve=conv,
                                        u1=u1, u2=u2, Nϕ=Nϕ, kwargs...)
    end
end

function _calc_formation_temp_cpu(star::StellarProps, linelist; Δλ::T=0.01,
                                  minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN,
                                  Nϕ::Int=128, showprogress::Bool=true,
                                  kwargs...) where T<:AF
    # get linelist
    wls = [l.wl * CM_TO_ANGSTROM for l in linelist]
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

    # populate atmosphere v_mic from stellar params
    star.ξ isa AbstractVector && @assert length(star.ξ) == Natm "v_micro vector length ($(length(star.ξ))) must match Natm ($Natm)"
    if star.ξ isa AbstractVector
        copyto!(atm_cpu.v_mic, T.(star.ξ))
    else
        fill!(atm_cpu.v_mic, star.ξ)
    end

    # dispatch between anchored (preferred) and Bezier (fallback when tau_ref unavailable)
    _calc_tau_cpu! = _make_tau_integrator(atm_cpu, zs, α_ref)

    # convolution or numerical integration
    if convolve
        # stationary (μ=1) flux quantities needed by the Hirano convolution path
        v_los = zeros(T, Natm)
        αs_broad = convolve_wavelength_axis(λs_korg, αs, v_los, atm_cpu.v_mic)
        αs_cont_broad = convolve_wavelength_axis(λs_korg, αs_cont, v_los, atm_cpu.v_mic)

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

        @assert !isnan(u1)
        @assert !isnan(u2)
        cfunc_dt_flux = convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux, star.vsini, star.ζ, u1, u2)
        cfunc_dt_flux_cont = convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux_cont, star.vsini, star.ζ, u1, u2)
    else # numerical disk integration
        if any(map(!isnan, (u1, u2)))
            @warn "Prescribed limb darkening coefficients are not used in integration method!"
        end

        # get stellar grid
        μs, dA, z_rot = calc_stellar_grid_cpu(star.ρstar, star.istar, star.vsini, Nϕ; α₂=star.α₂, α₄=star.α₄)
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
            # additive v_los: atmosphere base + rotation
            ws.v_los_buf .= atm_cpu.v_los .+ T(z_rot_cpu[i] * c_ms)

            # total absorption → macro_out → accumulate immediately
            _convolve_micro_inplace!(ws.αs_broad, λs_korg, αs, ws.v_los_buf, atm_cpu.v_mic, ws)
            _calc_tau_cpu!(μ_tile, ws.αs_broad, ws.τs_int)
            calc_intensity_cfunc_cpu!(ws.cfunc_int, Ts, λs_korg, ws.τs_int)
            @views ws.cfunc_dt_int .= ws.cfunc_int .* (ws.τs_int[2:end, :] .- ws.τs_int[1:end-1, :])
            _convolve_macro_inplace!(ws.macro_out, λs_korg, ws.cfunc_dt_int, star.ζ, μ_tile, ws)
            ws.cfunc_flux_acc .+= ws.macro_out .* dA_i

            # continuum absorption → macro_out → accumulate immediately
            _convolve_micro_inplace!(ws.αs_cont_broad, λs_korg, αs_cont, ws.v_los_buf, atm_cpu.v_mic, ws)
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
    return FormTempResult(collect(λs_korg), flux_norm, form_temps, cont_func, atm_cpu)
end

function _calc_formation_temp_gpu(star::StellarProps, linelist; Δλ::T=0.01,
                                  gpu_precision::Type{<:AF}=Float64,
                                  minλ::T=NaN, maxλ::T=NaN, buffer::T=2.0,
                                  convolve::Bool=false, u1::T=NaN, u2::T=NaN,
                                  Nϕ::Int=128, showprogress::Bool=true,
                                  kwargs...) where T<:AF
    G = gpu_precision  # shorthand for GPU float type

    # get linelist
    wls = [l.wl * CM_TO_ANGSTROM for l in linelist]
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

    Natm = size(αs, 1)
    Npad = 512

    # populate atmosphere v_mic from stellar params
    star.ξ isa AbstractVector && @assert length(star.ξ) == Natm "v_micro vector length ($(length(star.ξ))) must match Natm ($Natm)"
    if star.ξ isa AbstractVector
        copyto!(atm_gpu.v_mic, G.(star.ξ))
    else
        fill!(atm_gpu.v_mic, G(star.ξ))
    end

    # convolution or numerical integration
    if convolve
        # stationary (μ=1) flux quantities needed by the Hirano convolution path
        gpu_mem_cont = _make_gpu_mem()
        cmem = ConvolutionMemory(Nλ, Natm, Npad; T=G)
        cmem_cont = ConvolutionMemory(Nλ, Natm, Npad; T=G)
        cmem_mac = MacroConvolutionMemory(Nλ, Natm - 1, Npad; T=G)

        cfunc_flux_struct = calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, atm_gpu.v_mic)
        cfunc_dt_flux = cfunc_flux_struct.cfunc_dt
        cfunc_flux_struct_cont = calc_flux_quantities(αs_cont, atm_gpu, gpu_mem_cont, cmem_cont, atm_gpu.v_mic)
        cfunc_dt_flux_cont = cfunc_flux_struct_cont.cfunc_dt

        @assert !isnan(u1)
        @assert !isnan(u2)
        cfunc_dt_flux = copy(convolve_hirano_rotmacro_gpu(cmem_mac, λs_G, cfunc_dt_flux, G(star.vsini), G(star.ζ), G(u1), G(u2)))
        cfunc_dt_flux_cont = copy(convolve_hirano_rotmacro_gpu(cmem_mac, λs_G, cfunc_dt_flux_cont, G(star.vsini), G(star.ζ), G(u1), G(u2)))
    else # numerical disk integration
        if any(map(!isnan, (u1, u2)))
            @warn "Prescribed limb darkening coefficients are not used in integration method!"
        end

        # get stellar grid
        μs_gpu, dA, z_rot, _ = calc_stellar_grid(star.ρstar, star.istar, star.vsini, Nϕ; α₂=star.α₂, α₄=star.α₄)
        idx = findall(x -> x .> zero(eltype(μs_gpu)), Array(μs_gpu))
        μs_cpu = Array(μs_gpu)[idx]
        dA_cpu = Array(dA)[idx]
        z_rot_cpu = Array(z_rot)[idx]
        if iszero(star.vsini)
            z_rot_cpu .= 0.0
        end
        Ntiles = length(μs_cpu)
        Natm1 = Natm - 1

        # choose batch size: estimate per-tile memory, stay within 50% of free GPU RAM
        use_anchored = gpu_mem.use_anchored
        nfreq = fld(next_fft_friendly_len(Nλ + Npad), 2) + 1
        L_est = next_fft_friendly_len(Nλ + Npad)
        bpe = sizeof(G)
        bpc = sizeof(Complex{G})

        # per-tile (scales with B): bcmem batched buffers, dual-stream
        #   conv_gpu (Natm×L real) + kernel_ft_gpu (Natm×nfreq complex) +
        #   conv_ft_gpu (Natm×nfreq complex)
        bytes_bcmem_per_tile = Natm * (L_est * bpe + nfreq * bpc * 2)

        # cfdt_batch (Natm1×Nλ real), dual-stream
        bytes_cfdt_per_tile = Natm1 * Nλ * bpe

        # Bézier path also needs τs_batch (Natm×Nλ) + ds/alphaC (2×Natm), per-stream
        bytes_bezier_per_tile = use_anchored ? 0 :
            (Natm * Nλ * bpe + 2 * Natm * bpe)

        bytes_per_tile_total = 2 * (bytes_bcmem_per_tile + bytes_cfdt_per_tile +
                                    bytes_bezier_per_tile)

        # fixed cost (paid once, not scaling with B):
        #   bcmem shared: ys_gpu (Natm×Nλ) + signal_gpu (Natm×L) +
        #                 signal_ft_gpu (Natm×nfreq complex), dual-stream
        bytes_bcmem_shared = 2 * Natm * (Nλ * bpe + L_est * bpe + nfreq * bpc)
        #   accumulators: cfunc_flux + cfunc_comp, dual-stream = 4 × (Natm1×Nλ)
        bytes_accumulators = 4 * Natm1 * Nλ * bpe
        #   tile parameters: μ_tiles (Ntiles) + dA (Ntiles) + v_los (Ntiles×Natm)
        bytes_tile_params = Ntiles * (2 * bpe + Natm * bpe)

        bytes_fixed = bytes_bcmem_shared + bytes_accumulators + bytes_tile_params

        # macro convolution buffers (only when macroturbulence is active)
        if !iszero(star.ζ)
            # per-tile: mac_pad and mac_ft alias bcmem's conv_gpu and conv_ft_gpu,
            # so no additional per-tile memory for them.
            # fixed: acc_ft (complex) + mac_ifft_buf (real) + mac_out (real), dual-stream
            bytes_mac_fixed = 2 * Natm1 * (nfreq * bpc + L_est * bpe + Nλ * bpe)
            bytes_fixed += bytes_mac_fixed
        end

        avail = CUDA.free_memory()
        budget = Int(floor(avail * 0.5))
        B_mem = max(1, (budget - bytes_fixed) ÷ bytes_per_tile_total)
        B = min(64, Ntiles, B_mem)

        # only reclaim GPU memory if we can't reach maximum batch size without it
        if B_mem < 64
            GC.gc()
            CUDA.reclaim()
            avail = CUDA.free_memory()
            budget = Int(floor(avail * 0.5))
            B_mem = max(1, (budget - bytes_fixed) ÷ bytes_per_tile_total)
            B = min(64, Ntiles, B_mem)
        end

        # allocate batched convolution memories (dual-stream)
        bcmem      = BatchedMicroConvMem(Nλ, Natm, B, Npad; T=G)
        bcmem_cont = BatchedMicroConvMem(Nλ, Natm, B, Npad; T=G)

        # prime signal FFT caches: the forward FFT of the padded absorption signal
        # is tile-independent (only the Doppler filter changes per tile). The priming
        # call writes a throw-away Doppler filter and convolution product for Bcur=1,
        # which are overwritten by the first real batch — only signal_ft_gpu persists.
        v_los_prime = CUDA.zeros(G, Natm)
        bcmem.signal_cached = false
        convolve_wavelength_axis_batched!(bcmem, λs_G, αs, v_los_prime, atm_gpu.v_mic, 1)
        bcmem.signal_cached = true
        bcmem_cont.signal_cached = false
        convolve_wavelength_axis_batched!(bcmem_cont, λs_G, αs_cont, v_los_prime, atm_gpu.v_mic, 1)
        bcmem_cont.signal_cached = true

        # batched working arrays (dual-stream)
        cfdt_batch      = CUDA.zeros(G, B * Natm1, Nλ)
        cfdt_batch_cont = CUDA.zeros(G, B * Natm1, Nλ)

        # pre-upload all tile parameters (single H2D transfer replaces per-batch uploads)
        all_μ_tiles_gpu = CuArray(G.(μs_cpu))
        all_dA_tiles_gpu = CuArray(G.(dA_cpu))
        # additive v_los: atmosphere base + rotation per tile
        rot_v_los = repeat(G.(z_rot_cpu .* c_ms), inner=Natm)
        base_v_los = repeat(Array(atm_gpu.v_los), Ntiles)
        all_v_los_gpu = CuArray(rot_v_los .+ base_v_los)

        # Bezier work arrays (only allocated when needed)
        if !use_anchored
            τs_batch      = CUDA.zeros(G, B * Natm, Nλ)
            τs_batch_cont = CUDA.zeros(G, B * Natm, Nλ)
            ds_batch      = CUDA.zeros(G, B * Natm)
            alphaC_batch  = CUDA.zeros(G, B * Natm)
            ds_batch_cont = CUDA.zeros(G, B * Natm)
            alphaC_batch_cont = CUDA.zeros(G, B * Natm)
        end

        # accumulators + Kahan compensation arrays
        cfunc_flux_integration = CUDA.zeros(G, Natm1, Nλ)
        cfunc_flux_cont_integration = CUDA.zeros(G, Natm1, Nλ)
        cfunc_comp = CUDA.zeros(G, Natm1, Nλ)
        cfunc_cont_comp = CUDA.zeros(G, Natm1, Nλ)

        # tau integration constants (shared across tiles)
        log_τ_ref    = gpu_mem.log_τ_ref
        ifactor_base = gpu_mem.ifactor_base

        # precompute macro kernel FFTs for all unique μ values (batched)
        if !iszero(star.ζ)
            L_mac, _, pad_left_mac, _ = _conv_mem_geometry(Nλ, Npad)
            nfreq_mac = fld(L_mac, 2) + 1
            i0_mac = Nλ ÷ 2 + 1

            unique_μ_sorted = sort(unique(G.(μs_cpu)))
            N_unique = length(unique_μ_sorted)
            μ_to_idx = Dict(μ => Int32(i) for (i, μ) in enumerate(unique_μ_sorted))
            v_losals_gpu = CuArray(unique_μ_sorted)

            # evaluate all kernels in DFT layout with one 2D kernel launch
            kbuf_mac = CUDA.zeros(G, N_unique, L_mac)
            ts_kc = (32, 32)
            bs_kc = (cld(Nλ, ts_kc[1]), cld(N_unique, ts_kc[2]))
            @cuda threads=ts_kc blocks=bs_kc compute_rt_macro_dft_layout_2d!(
                kbuf_mac, gpu_mem.λs, v_losals_gpu, Int32(i0_mac), G(star.ζ),
                Int32(Nλ), Int32(L_mac))
            # TODO(zero-sum-guard): unguarded normalization; can produce NaN if
            # kernel underflows. See microturbulence.jl pattern +
            # .claude/CLAUDE.md "Kernel normalization underflow guard".
            kbuf_mac ./= sum(kbuf_mac, dims=2)
            CUDA.unsafe_free!(v_losals_gpu)

            # batched R2C FFT → kernel_cache_flat; free temporary kbuf_mac afterward
            plan_kc = CUDA.CUFFT.plan_rfft(kbuf_mac, 2)
            kernel_cache_flat = CUDA.zeros(Complex{G}, N_unique, nfreq_mac)
            mul!(kernel_cache_flat, plan_kc, kbuf_mac)
            CUDA.unsafe_free!(kbuf_mac)

            # per-tile μ index
            μ_idx_gpu = CuArray(Int32[μ_to_idx[G(μs_cpu[i])] for i in 1:Ntiles])

            # batched macro buffers: alias bcmem's conv_gpu and conv_ft_gpu to avoid
            # separate allocations. mac_pad (B*Natm1, L) fits inside conv_gpu (B*Natm, L)
            # because Natm1 < Natm. They're used sequentially on the same stream:
            # conv_gpu is consumed by the fused τ+cfunc kernel before mac_pad is written.
            # unsafe_wrap creates a non-owning CuArray with different shape over the same
            # flat device memory; cuFFT plans are created on the wrapped arrays.
            @assert B * Natm1 * L_mac <= B * Natm * L_mac  # Natm1 < Natm
            @assert B * Natm1 * nfreq_mac <= B * Natm * nfreq_mac
            mac_pad      = unsafe_wrap(CuArray{G, 2}, pointer(bcmem.conv_gpu), (B * Natm1, L_mac))
            mac_pad_cont = unsafe_wrap(CuArray{G, 2}, pointer(bcmem_cont.conv_gpu), (B * Natm1, L_mac))
            mac_ft      = unsafe_wrap(CuArray{Complex{G}, 2}, pointer(bcmem.conv_ft_gpu), (B * Natm1, nfreq_mac))
            mac_ft_cont = unsafe_wrap(CuArray{Complex{G}, 2}, pointer(bcmem_cont.conv_ft_gpu), (B * Natm1, nfreq_mac))
            plan_mac_fwd      = CUDA.CUFFT.plan_rfft(mac_pad, 2)
            plan_mac_fwd_cont = CUDA.CUFFT.plan_rfft(mac_pad_cont, 2)

            # Fourier-space accumulators (Natm1 × nfreq, summed across all batches)
            acc_ft      = CUDA.zeros(Complex{G}, Natm1, nfreq_mac)
            acc_ft_cont = CUDA.zeros(Complex{G}, Natm1, nfreq_mac)

            # final IFFT buffers (plans created from acc_ft/acc_ft_cont to avoid
            # throwaway CuArray allocations)
            mac_ifft_buf      = CUDA.zeros(G, Natm1, L_mac)
            mac_ifft_buf_cont = CUDA.zeros(G, Natm1, L_mac)
            plan_mac_bwd      = CUDA.CUFFT.plan_irfft(acc_ft, L_mac, 2)
            plan_mac_bwd_cont = CUDA.CUFFT.plan_irfft(acc_ft_cont, L_mac, 2)

            # real-space output after final IFFT + extract
            mac_out      = CUDA.zeros(G, Natm1, Nλ)
            mac_out_cont = CUDA.zeros(G, Natm1, Nλ)
        end

        # CUDA streams for overlapping total/continuum
        stream_total = CuStream()
        stream_cont = CuStream()

        # batched tile loop (tile parameters pre-uploaded; kernels use tile_offset)
        prog = Progress(Ntiles; enabled=showprogress)
        for batch_start in 1:B:Ntiles
            batch_end = min(batch_start + B - 1, Ntiles)
            Bcur = batch_end - batch_start + 1
            tile_offset = batch_start - 1
            BNatm1 = Bcur * Natm1

            # total absorption on stream_total
            CUDA.stream!(stream_total) do
                αs_conv = convolve_wavelength_axis_batched!(bcmem, λs_G, αs,
                    all_v_los_gpu, atm_gpu.v_mic, Bcur; tile_offset=tile_offset)
                if use_anchored
                    calc_tau_cfunc_dt_fused!(cfdt_batch, αs_conv,
                        log_τ_ref, ifactor_base, all_μ_tiles_gpu,
                        atm_gpu.Ts_gpu, gpu_mem.λs, Natm, Bcur;
                        tile_offset=tile_offset)
                else
                    calc_tau_bezier_batched!(all_μ_tiles_gpu, atm_gpu.zs_gpu,
                        αs_conv, τs_batch, ds_batch, alphaC_batch, Natm, Bcur;
                        tile_offset=tile_offset)
                    calc_intensity_cfunc_dt_batched!(cfdt_batch, τs_batch,
                        atm_gpu.Ts_gpu, gpu_mem.λs, Natm, Bcur)
                end

                if iszero(star.ζ)
                    accumulate_batch!(cfunc_flux_integration, cfunc_comp,
                        cfdt_batch, all_dA_tiles_gpu, Natm1, Bcur;
                        tile_offset=tile_offset)
                else
                    # pad cfdt and batched forward FFT (full B*Natm1 buffer;
                    # multiply-accumulate limits to Bcur via its loop bound)
                    ts_pad = (32, 32)
                    bs_pad = (cld(B * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
                    @cuda threads=ts_pad blocks=bs_pad pad_signal!(mac_pad, cfdt_batch,
                                                                    Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
                    mul!(mac_ft, plan_mac_fwd, mac_pad)
                    batched_macro_multiply_accumulate!(acc_ft, mac_ft, kernel_cache_flat,
                        μ_idx_gpu, all_dA_tiles_gpu, Natm1, Bcur; tile_offset=tile_offset)
                end
            end

            # continuum absorption on stream_cont
            CUDA.stream!(stream_cont) do
                αs_conv_c = convolve_wavelength_axis_batched!(bcmem_cont, λs_G, αs_cont,
                    all_v_los_gpu, atm_gpu.v_mic, Bcur; tile_offset=tile_offset)
                if use_anchored
                    calc_tau_cfunc_dt_fused!(cfdt_batch_cont, αs_conv_c,
                        log_τ_ref, ifactor_base, all_μ_tiles_gpu,
                        atm_gpu.Ts_gpu, gpu_mem.λs, Natm, Bcur;
                        tile_offset=tile_offset)
                else
                    calc_tau_bezier_batched!(all_μ_tiles_gpu, atm_gpu.zs_gpu,
                        αs_conv_c, τs_batch_cont, ds_batch_cont, alphaC_batch_cont, Natm, Bcur;
                        tile_offset=tile_offset)
                    calc_intensity_cfunc_dt_batched!(cfdt_batch_cont, τs_batch_cont,
                        atm_gpu.Ts_gpu, gpu_mem.λs, Natm, Bcur)
                end

                if iszero(star.ζ)
                    accumulate_batch!(cfunc_flux_cont_integration, cfunc_cont_comp,
                        cfdt_batch_cont, all_dA_tiles_gpu, Natm1, Bcur;
                        tile_offset=tile_offset)
                else
                    ts_pad = (32, 32)
                    bs_pad = (cld(B * Natm1, ts_pad[1]), cld(L_mac, ts_pad[2]))
                    @cuda threads=ts_pad blocks=bs_pad pad_signal!(mac_pad_cont, cfdt_batch_cont,
                                                                    Nλ, pad_left_mac, L_mac - pad_left_mac - Nλ)
                    mul!(mac_ft_cont, plan_mac_fwd_cont, mac_pad_cont)
                    batched_macro_multiply_accumulate!(acc_ft_cont, mac_ft_cont, kernel_cache_flat,
                        μ_idx_gpu, all_dA_tiles_gpu, Natm1, Bcur; tile_offset=tile_offset)
                end
            end

            # sync both streams before next batch
            CUDA.synchronize(stream_total)
            CUDA.synchronize(stream_cont)
            for _ in 1:Bcur; next!(prog); end
        end

        if iszero(star.ζ)
            cfunc_dt_flux = cfunc_flux_integration
            cfunc_dt_flux_cont = cfunc_flux_cont_integration
        else
            # final IFFT of Fourier-space accumulators + extract valid region
            mul!(mac_ifft_buf, plan_mac_bwd, acc_ft)
            ts_ext = (32, 32)
            bs_ext = (cld(Natm1, ts_ext[1]), cld(Nλ, ts_ext[2]))
            @cuda threads=ts_ext blocks=bs_ext extract_valid!(mac_out, mac_ifft_buf, pad_left_mac, Nλ)

            mul!(mac_ifft_buf_cont, plan_mac_bwd_cont, acc_ft_cont)
            @cuda threads=ts_ext blocks=bs_ext extract_valid!(mac_out_cont, mac_ifft_buf_cont, pad_left_mac, Nλ)

            cfunc_dt_flux = mac_out
            cfunc_dt_flux_cont = mac_out_cont
        end
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
