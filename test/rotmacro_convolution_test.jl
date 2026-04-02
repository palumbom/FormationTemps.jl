let
using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics

# load the linelist (Fe I 6301/6302)
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs    = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls      = [l.wl for l in linelist]
idx1     = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2     = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

wls     = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]
buffer  = 1.0
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.0025)

A_X     = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs      = atm_gpu.zs

αs      = zeros(length(zs), length(λs_korg))
αs_cont = zeros(length(zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

Nλ   = length(λs_korg)
Natm = size(αs, 1)
Npad = 1024
cmem  = FT.ConvolutionMemory(Nλ, Natm, Npad)
cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)
gpu_mem  = FT.GPUMemory(λs_korg, atm_gpu)

σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

cfunc_flux_stationary      = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
cfunc_flux_cont_stationary = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)

# broadening parameters
vsini = 4200.0
u1 = 0.4

# gray rotation convolution: compare CPU vs GPU
cfunc_rot_cpu = FT.convolve_gray_rotation(λs_korg, Array(cfunc_flux_stationary.cfunc_dt), vsini, u1)
cfunc_rot_gpu = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, cfunc_flux_stationary.cfunc_dt, vsini, u1))

cfunc_cont_rot_cpu = FT.convolve_gray_rotation(λs_korg, Array(cfunc_flux_cont_stationary.cfunc_dt), vsini, u1)
cfunc_cont_rot_gpu = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, cfunc_flux_cont_stationary.cfunc_dt, vsini, u1))

flux_norm_cpu = dropdims(sum(cfunc_rot_cpu, dims=1), dims=1) ./
                dropdims(sum(cfunc_cont_rot_cpu, dims=1), dims=1)
flux_norm_gpu = dropdims(sum(cfunc_rot_gpu, dims=1), dims=1) ./
                dropdims(sum(cfunc_cont_rot_gpu, dims=1), dims=1)

# isotropic RT macro convolution: compare CPU vs GPU
ζ_rt = 1200.0
cfunc_rt_cpu  = FT.convolve_iso_rt_macro(λs_korg, Array(cfunc_flux_stationary.cfunc_dt), ζ_rt)
cfunc_rt_gpu  = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, cfunc_flux_stationary.cfunc_dt, ζ_rt))

flux_iso_rt_cpu = dropdims(sum(cfunc_rt_cpu, dims=1), dims=1)
flux_iso_rt_gpu = dropdims(sum(cfunc_rt_gpu, dims=1), dims=1)

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=true)
    ax1.plot(λs_korg, flux_norm_cpu, label="{\\rm CPU}")
    ax1.plot(λs_korg, flux_norm_gpu, ls="--", label="{\\rm GPU}")
    ax1.set_ylabel("{\\rm Normalized flux (rotation)}")
    ax1.legend()
    ax2.plot(λs_korg, flux_norm_gpu .- flux_norm_cpu)
    ax2.set_ylabel("{\\rm GPU} \$-\$ {\\rm CPU}")
    ax2.set_xlabel("{\\rm Wavelength [\\AA]}")
    fig.savefig(joinpath(test_plotdir, "rotmacro_convolution.pdf"), bbox_inches="tight")
    plt.close()
end


@testset "CPU/GPU gray rotation convolution agreement" begin
    # CPU uses unpadded circular convolution; GPU uses padded linear convolution.
    # They differ at the edges (first/last ~vsini/c*λ0/Δλ pixels) due to wrap-around.
    # Compare only the interior, which should agree to floating-point precision.
    Δλ = step(λs_korg)
    λ0 = mean(collect(λs_korg))
    edge_px = ceil(Int, vsini / FT.c_ms * λ0 / Δλ) + 10
    interior = (edge_px + 1):(length(λs_korg) - edge_px)
    @test maximum(abs.(flux_norm_cpu[interior] .- flux_norm_gpu[interior])) < 1e-8
end

@testset "CPU/GPU isotropic RT macro convolution agreement" begin
    norm = maximum(abs.(flux_iso_rt_cpu))
    @test norm > 0
    # GPU uses CUDA erfc; CPU uses Julia/FFTW erfc. These differ at ~1e-4 relative to peak flux.
    @test maximum(abs.(flux_iso_rt_cpu .- flux_iso_rt_gpu)) / norm < 1e-3
end


end
