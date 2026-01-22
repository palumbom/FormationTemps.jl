use_gpu = FT.GPU_DEFAULT

# Use the same linelist slice as scripts/generate_solar.jl
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

# wavelength grid 
Δλ = 0.01
wls = [l.wl * 1e8 for l in linelist]
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=Δλ)

# set parameters for synthesis
convolve = true
u1 = 0.43
u2 = 0.31
Nϕ = 16

# set stellar params
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 0.0
ζ_RT = 0.0
ξ = 850.0
star_stat = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# get stationary convenience flux
result_stationary_convenience = calc_formation_temp(star_stat, linelist, use_gpu=false, 
                                                    u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                                    ne_warn_thresh=Inf)
if use_gpu 
    result_stationary_convenience_gpu = calc_formation_temp(star_stat, linelist, use_gpu=true, 
                                                            u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                                            ne_warn_thresh=Inf)
end

# get model atmosphere
atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(Teff, logg, star_stat.A_X))
zs = atm_cpu.zs
Ts = atm_cpu.Ts

# get the absorption coefficients
T = Float64
Natm = length(zs)
Nλ = length(λs_korg)
αs = zeros(T, Natm, Nλ)
αs_cont = zeros(T, Natm, Nλ)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_cpu, star_stat.A_X, ne_warn_thresh=Inf)

# set microturbulent broadening
σ_v = fill(ξ, Natm)
μ_v = zeros(T, Natm)

# get the absorption coefficients
αs_broad = FT.convolve_wavelength_axis(λs_korg, αs, μ_v, σ_v)
αs_cont_broad = FT.convolve_wavelength_axis(λs_korg, αs_cont, μ_v, σ_v)

# get the taus
τs = zeros(T, Natm, Nλ)
τs_cont = zeros(T, Natm, Nλ)
FT.calc_tau_cpu!(one(T), zs, αs_broad, τs)
FT.calc_tau_cpu!(one(T), zs, αs_cont_broad, τs_cont)

# get the contribution function
cfunc_flux = zeros(T, Natm - 1, Nλ)
cfunc_flux_cont = zeros(T, Natm - 1, Nλ)
FT.calc_flux_cfunc_cpu!(cfunc_flux, Ts, λs_korg, τs)
FT.calc_flux_cfunc_cpu!(cfunc_flux_cont, Ts, λs_korg, τs_cont)

cfunc_dt_flux = cfunc_flux .* diff(τs, dims=1)
cfunc_dt_flux_cont = cfunc_flux_cont .* diff(τs_cont, dims=1)

# get the flux
flux_norm = vec(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1))

@testset "Testing dimensions" begin
    @test size(αs) == size(αs_broad) == size(αs_cont) == size(αs_cont_broad)
    @test size(result_stationary_convenience.cont_func) == (length(zs)-1, length(λs_korg))
    @test size(cfunc_dt_flux_cont) == size(result_stationary_convenience.cont_func)
end

@testset "Testing stationary flux" begin
    @test maximum(flux_norm) .<= one(T)
    @test maximum(result_stationary_convenience.flux) .<= one(T)
    @test all(isapprox.(result_stationary_convenience.flux, flux_norm))
end

# now do non-stationary flux
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0
star = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# get stationary convenience flux
result_convenience = calc_formation_temp(star, linelist, use_gpu=false, 
                                         u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                         ne_warn_thresh=Inf)
if use_gpu
    result_convenience_gpu = calc_formation_temp(star, linelist, use_gpu=true, 
                                                 u1=u1, u2=u2, Δλ=Δλ, convolve=true,
                                                 ne_warn_thresh=Inf)
end

# convolve manually
cfunc_dt_flux = FT.convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux, star.vsini, star.ζ, u1, u2)
cfunc_dt_flux_cont = FT.convolve_hirano_rotmacro(λs_korg, cfunc_dt_flux_cont, star.vsini, star.ζ, u1, u2)
flux_rotating = vec(Array(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1)))

@testset "Testing convolved convenience flux" begin
    @test maximum(flux_rotating) .<= one(T)
    @test maximum(result_convenience.flux) .<= one(T)
    @test all(isapprox.(result_convenience.flux, flux_rotating))
end

if use_gpu
    @testset "Testing GPU convolved convenience flux" begin
        @test all(isapprox.(result_stationary_convenience.flux, result_stationary_convenience_gpu.flux)) 
        @test all(isapprox.(result_convenience.flux, result_convenience_gpu.flux)) 
    end
end

# # absorption coefficients
# αs = zeros(length(atm_gpu.zs), length(λs_korg))
# αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
# FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, star_props.A_X)

# # allocate on device
# Nλ = length(λs_korg)
# Natm = size(αs, 1)
# Npad = 100

# gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)
# cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
# cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

# # microturbulence broadening
# σ_v = CUDA.zeros(Float64, length(atm_gpu.zs)) .+ star_props.ξ

# # explicit flux contribution function (flux_vs_intensity.jl style)
# cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
# cfunc_dt_flux = cfunc_flux_struct.cfunc_dt

# cfunc_flux_struct_cont = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v)
# cfunc_dt_flux_cont = cfunc_flux_struct_cont.cfunc_dt

# # optional convolution (matches convenience function)
# if convolve
#     @assert !isnan(u1)
#     @assert !isnan(u2)
#     cfunc_dt_flux = FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux, star_props.vsini, star_props.ζ, u1, u2)
#     cfunc_dt_flux_cont = FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_dt_flux_cont, star_props.vsini, star_props.ζ, u1, u2)
# end

# # cumulative contribution function and formation temperature
# cum_cfunc_flux = Array(cumsum(cfunc_dt_flux, dims=1))
# cum_cfunc_flux ./= maximum(cum_cfunc_flux, dims=1)

# flux_norm = vec(Array(sum(cfunc_dt_flux, dims=1) ./ sum(cfunc_dt_flux_cont, dims=1)))

# form_temps_explicit = zeros(length(λs_korg))
# mid_temps = FT.elav(atm_gpu.Ts)
# for i in eachindex(λs_korg)
#     xs = view(cum_cfunc_flux, :, i)
#     itp = FT.linear_interp(xs, mid_temps)
#     form_temps_explicit[i] = itp(0.5)
# end

# # convenience function output
# form_temp_result = FT.calc_formation_temp(star_props, linelist; Δλ=Δλ, convolve=convolve, u1=u1, u2=u2, Nϕ=Nϕ)

# # comparisons
# @assert length(form_temp_result.wavs) == length(λs_korg)
# wavs_diff = maximum(abs.(form_temp_result.wavs .- λs_korg))
# max_abs_temp = maximum(abs.(form_temps_explicit .- form_temp_result.form_temps))
# mean_abs_temp = mean(abs.(form_temps_explicit .- form_temp_result.form_temps))
# max_abs_flux = maximum(abs.(flux_norm .- form_temp_result.flux))
# mean_abs_flux = mean(abs.(flux_norm .- form_temp_result.flux))

# @printf("wavelength grid max abs diff: %.6e\n", wavs_diff)
# @printf("formation temperature abs diff: max=%.6e, mean=%.6e\n", max_abs_temp, mean_abs_temp)
# @printf("normalized flux abs diff: max=%.6e, mean=%.6e\n", max_abs_flux, mean_abs_flux)

# println()
# println("integration case (disk integration)")

# # explicit disk integration (scripts/hr.jl style)
# μs_gpu, dA, z_rot, _ = FT.calc_stellar_grid(star_props.ρstar, star_props.istar, star_props.vsini, Nϕ)
# idx = findall(x -> x .> zero(eltype(μs_gpu)), Array(μs_gpu))
# μs_cpu = Array(μs_gpu)[idx]
# dA_cpu = Array(dA)[idx]
# z_rot_cpu = Array(z_rot)[idx]
# if iszero(star_props.vsini)
#     z_rot_cpu .= 0.0
# end

# μ_v_rot = CUDA.zeros(Float64, Natm)
# flux_integration = CUDA.zeros(Float64, length(λs_korg))
# flux_cont_integration = CUDA.zeros(Float64, length(λs_korg))
# cfunc_flux_integration = CUDA.zeros(Float64, Natm - 1, length(λs_korg))
# cfunc_flux_cont_integration = CUDA.zeros(Float64, Natm - 1, length(λs_korg))

# @showprogress for i in eachindex(μs_cpu)
#     μ_tile = μs_cpu[i]
#     μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

#     cfunc_intensity = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
#     tbc = cfunc_intensity.cfunc_dt
#     cfunc_int_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, star_props.ζ, μ_tile)

#     cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μ_tile, μ_v_rot, σ_v)
#     tbc_cont = cfunc_intensity_cont.cfunc_dt
#     cfunc_int_cont_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc_cont, star_props.ζ, μ_tile)

#     flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
#     flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[i]
#     cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]
#     cfunc_flux_cont_integration .+= cfunc_int_cont_i_mac .* dA_cpu[i]
# end

# cum_cfunc_flux_int = Array(cumsum(cfunc_flux_integration, dims=1))
# cum_cfunc_flux_int ./= maximum(cum_cfunc_flux_int, dims=1)
# flux_norm_int = vec(Array(sum(cfunc_flux_integration, dims=1) ./ sum(cfunc_flux_cont_integration, dims=1)))

# form_temps_explicit_int = zeros(length(λs_korg))
# for i in eachindex(λs_korg)
#     xs = view(cum_cfunc_flux_int, :, i)
#     itp = FT.linear_interp(xs, mid_temps)
#     form_temps_explicit_int[i] = itp(0.5)
# end

# form_temp_result_int = FT.calc_formation_temp(star_props, linelist; Δλ=Δλ, convolve=false, Nϕ=Nϕ)

# @assert length(form_temp_result_int.wavs) == length(λs_korg)
# wavs_diff_int = maximum(abs.(form_temp_result_int.wavs .- λs_korg))
# max_abs_temp_int = maximum(abs.(form_temps_explicit_int .- form_temp_result_int.form_temps))
# mean_abs_temp_int = mean(abs.(form_temps_explicit_int .- form_temp_result_int.form_temps))
# max_abs_flux_int = maximum(abs.(flux_norm_int .- form_temp_result_int.flux))
# mean_abs_flux_int = mean(abs.(flux_norm_int .- form_temp_result_int.flux))

# @printf("wavelength grid max abs diff: %.6e\n", wavs_diff_int)
# @printf("formation temperature abs diff: max=%.6e, mean=%.6e\n", max_abs_temp_int, mean_abs_temp_int)
# @printf("normalized flux abs diff: max=%.6e, mean=%.6e\n", max_abs_flux_int, mean_abs_flux_int)
