using FormationTemps; FT = FormationTemps
using Korg
using CUDA

# empty linelist — pure continuum
linelist = []

# wide wavelength grid
λs_korg = range(2_000.0, 8_000.0, step=0.1)
# λs_korg = range(2_000.0, 30_000.0, step=0.1)

A_X = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs = atm_gpu.zs
Ts = atm_gpu.Ts

# absorption coefficients (continuum only)
αs = zeros(length(zs), length(λs_korg))
αs_cont = zeros(length(zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

Nλ   = length(λs_korg)
Natm = size(αs, 1)
Npad = 240
cmem    = FT.ConvolutionMemory(Nλ, Natm, Npad)
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

cfunc_int  = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, 1.0, μ_v_rot, σ_v_mic)
cfunc_flux = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)

cfunc_int_cum  = Array(FT.get_cum_cfunc(cfunc_int))
cfunc_flux_cum = Array(FT.get_cum_cfunc(cfunc_flux))

mid_zs = elav(zs)
mid_Ts = elav(Ts)

form_height = zeros(length(λs_korg))
form_temp   = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cfunc_flux_cum, :, i)
    form_height[i] = FT.linear_interp(xs, mid_zs)(0.5)
    form_temp[i]   = FT.linear_interp(xs, mid_Ts)(0.5)
end

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=true)
    ax1.plot(λs_korg, form_height)
    ax1.set_ylabel("Form. Height [cm]")
    ax2.plot(λs_korg, form_temp)
    ax2.set_ylabel("Form. Temperature [K]")
    ax2.set_xlabel("Wavelength [Å]")
    fig.savefig(joinpath(test_plotdir, "height_test.pdf"), bbox_inches="tight")
    plt.close()
end

@testset "Formation height and temperature physical bounds" begin
    @test all(isfinite.(form_height))
    @test all(isfinite.(form_temp))
    @test minimum(Ts) <= minimum(form_temp)
    @test maximum(form_temp) <= maximum(Ts)
    @test minimum(zs) <= minimum(form_height)
    @test maximum(form_height) <= maximum(zs)
end
