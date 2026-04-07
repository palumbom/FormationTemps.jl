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
buffer  = 2.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.001)

A_X     = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs      = atm_gpu.zs

αs      = zeros(length(zs), length(λs_korg))
αs_cont = zeros(length(zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

Nλ   = length(λs_korg)
Natm = size(αs, 1)
Npad = 5000
cmem     = FT.ConvolutionMemory(Nλ, Natm, Npad)
cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)
gpu_mem  = FT.GPUMemory(λs_korg, atm_gpu)

v_los_rot = CUDA.zeros(Float64, length(zs))
v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, v_mic)
tbc = cfunc_flux_stationary.cfunc_dt

ζ_rt  = 3400.0
v_losal = 1.0

# compare CPU vs GPU anisotropic RT macro convolution
cfunc_flux_rt_cpu = FT.convolve_rt_macro(λs_korg, Array(tbc), ζ_rt, v_losal)
cfunc_flux_rt_gpu = Array(FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt, v_losal))

flux_cpu = 2π .* dropdims(sum(cfunc_flux_rt_cpu, dims=1), dims=1)
flux_gpu = 2π .* dropdims(Array(sum(cfunc_flux_rt_gpu, dims=1)), dims=1)

# percent error, ignoring near-zero continuum
flux_err = 100.0 .* ((flux_cpu .- flux_gpu) ./ flux_cpu)

@testset "CPU/GPU anisotropic RT macro convolution agreement" begin
    @test maximum(abs.(flux_err)) < 0.01
end

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(λs_korg, flux_err)
    ax.set_xlabel("{\\rm Wavelength [\\AA]}")
    ax.set_ylabel("{\\rm CPU} \$-\$ {\\rm GPU flux error [\\%]}")
    ax.set_title("{\\rm Anisotropic RT macro (mu = $(v_losal))}")
    fig.savefig(joinpath(test_plotdir, "cusp.pdf"), bbox_inches="tight")
    plt.close()
end

end
