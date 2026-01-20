using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using Printf

if !CUDA.functional()
    error("No GPU found; cannot compare CPU and GPU paths.")
end

Δλ = 0.005
Nϕ = 16
convolve_cases = [true,]# false]

# Match generate_solar.jl inputs
Teff = 5777.0
logg = 4.44
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0
u1 = 0.43
u2 = 0.31

# Use the same linelist slice as scripts/generate_solar.jl
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))[16000:16100]
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

star_props = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

for convolve in convolve_cases
    println()
    println("convolve = ", convolve)
    if convolve
        result_gpu = FT.calc_formation_temp(star_props, linelist; use_gpu=true, Δλ=Δλ, convolve=true, u1=u1, u2=u2, Nϕ=Nϕ)
        result_cpu = FT.calc_formation_temp(star_props, linelist; use_gpu=false, Δλ=Δλ, convolve=true, u1=u1, u2=u2, Nϕ=Nϕ)
    else
        result_gpu = FT.calc_formation_temp(star_props, linelist; use_gpu=true, Δλ=Δλ, convolve=false, Nϕ=Nϕ)
        result_cpu = FT.calc_formation_temp(star_props, linelist; use_gpu=false, Δλ=Δλ, convolve=false, Nϕ=Nϕ)
    end

    @assert length(result_gpu.wavs) == length(result_cpu.wavs)
    wavs_diff = maximum(abs.(result_gpu.wavs .- result_cpu.wavs))
    max_abs_temp = maximum(abs.(result_gpu.form_temps .- result_cpu.form_temps))
    mean_abs_temp = mean(abs.(result_gpu.form_temps .- result_cpu.form_temps))
    max_abs_flux = maximum(abs.(result_gpu.flux .- result_cpu.flux))
    mean_abs_flux = mean(abs.(result_gpu.flux .- result_cpu.flux))

    # plt.plot(result_cpu.wavs, result_gpu.flux)
    # plt.plot(result_cpu.wavs, result_cpu.flux)
    # plt.plot(result_cpu.wavs, result_gpu.flux .- result_cpu.flux)
    plt.plot(result_cpu.wavs, result_gpu.form_temps .- result_cpu.form_temps)
    plt.savefig("derp.pdf")
    plt.close()

    @printf("wavelength grid max abs diff: %.6e\n", wavs_diff)
    @printf("formation temperature abs diff: max=%.6e, mean=%.6e\n", max_abs_temp, mean_abs_temp)
    @printf("normalized flux abs diff: max=%.6e, mean=%.6e\n", max_abs_flux, mean_abs_flux)
end
