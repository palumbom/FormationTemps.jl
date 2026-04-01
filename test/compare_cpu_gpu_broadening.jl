using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics
using ProgressMeter

# load the linelist (Fe I 6301/6302)
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs    = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls      = [l.wl for l in linelist]
idx1     = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6301, wls)
idx2     = findfirst(x -> x * FT.CM_TO_ANGSTROM >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])
wls      = [l.wl * FT.CM_TO_ANGSTROM for l in linelist]

# broadening parameters
vsini = 4200.0
u1    = 0.4
u2    = 0.26
ζ_rt  = 1200.0
μ_val = 0.9

# test at three representative wavelength spacings
steps = [0.001, 0.002, 0.005]

αs_error      = zeros(length(steps))
rot_error     = zeros(length(steps))
rt_error      = zeros(length(steps))
rt_aniso_error = zeros(length(steps))
rotmacro_error = zeros(length(steps))

@showprogress for i in eachindex(steps)
    buffer  = 0.5
    λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=steps[i])

    A_X     = Korg.asplund_2020_solar_abundances
    atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
    zs      = atm_gpu.zs

    αs      = zeros(length(zs), length(λs_korg))
    αs_cont = zeros(length(zs), length(λs_korg))
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

    Nλ   = length(λs_korg)
    Natm = size(αs, 1)
    Npad = 240
    cmem     = FT.ConvolutionMemory(Nλ, Natm, Npad)
    cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)
    gpu_mem  = FT.GPUMemory(λs_korg, atm_gpu)

    σ_v_val = 1200.0

    # microturbulence broadening (scalar overload)
    αs_cpu_new = FT.convolve_wavelength_axis(λs_korg, αs, 0.0, σ_v_val)
    αs_gpu_new = FT.convolve_wavelength_axis_gpu(cmem, CuArray(collect(λs_korg)), CuArray(αs), 0.0, σ_v_val)
    αs_error[i] = maximum(abs.((Array(αs_gpu_new) .- αs_cpu_new) ./ αs_cpu_new))

    # contribution function for convolution tests
    cfunc_flux_stationary = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_val)
    tbc = Array(cfunc_flux_stationary.cfunc_dt)

    # Edge exclusion for compact-support (rotation) kernels: CPU uses circular FFT, GPU uses
    # padded linear convolution; they differ at the first/last ~vsini/c*λ0/Δλ pixels.
    λ0_val = mean(collect(λs_korg))
    edge_px = ceil(Int, vsini / FT.c_ms * λ0_val / steps[i]) + 10
    interior = (edge_px+1):(Nλ - edge_px)

    # gray rotation (interior only — compact-support kernel, circular vs padded edge artifact)
    cfunc_rot_cpu = FT.convolve_gray_rotation(λs_korg, tbc, vsini, u1)
    cfunc_rot_gpu = Array(FT.convolve_gray_rotation_gpu(cmem_mac, λs_korg, tbc, vsini, u1))
    rot_error[i]  = maximum(abs.((cfunc_rot_cpu .- cfunc_rot_gpu) ./ cfunc_rot_cpu)[:, interior])

    # isotropic RT macro (normalize by max to avoid blow-up at near-zero continuum pixels)
    cfunc_rt_cpu = FT.convolve_iso_rt_macro(λs_korg, tbc, ζ_rt)
    cfunc_rt_gpu = Array(FT.convolve_iso_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt))
    rt_error[i]  = maximum(abs.(cfunc_rt_cpu .- cfunc_rt_gpu)) / maximum(abs.(cfunc_rt_cpu))

    # anisotropic RT macro (normalize by max to avoid blow-up at near-zero continuum pixels)
    cfunc_aniso_rt_cpu = FT.convolve_rt_macro(λs_korg, tbc, ζ_rt, μ_val)
    cfunc_aniso_rt_gpu = Array(FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt, μ_val))
    rt_aniso_error[i]  = maximum(abs.(cfunc_aniso_rt_cpu .- cfunc_aniso_rt_gpu)) / maximum(abs.(cfunc_aniso_rt_cpu))

    # Hirano combined rotation+macro (interior only — contains rotation, same edge artifact)
    cfunc_hirano_cpu  = FT.convolve_hirano_rotmacro(λs_korg, tbc, vsini, ζ_rt, u1, u2)
    cfunc_hirano_gpu  = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, tbc, vsini, ζ_rt, u1, u2))
    rotmacro_error[i] = maximum(abs.((cfunc_hirano_cpu .- cfunc_hirano_gpu) ./ cfunc_hirano_cpu)[:, interior])
end

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
    plt.ioff()
    fig, ax = plt.subplots()
    ax.scatter(steps, αs_error,       s=20, label="{\\rm alpha}")
    ax.scatter(steps, rot_error,      s=20, label="{\\rm rotation}")
    ax.scatter(steps, rt_error,       s=20, label="{\\rm iso RT macro}")
    ax.scatter(steps, rt_aniso_error, s=20, label="{\\rm aniso RT macro}")
    ax.scatter(steps, rotmacro_error, s=20, label="{\\rm Hirano}")
    ax.set_xlabel("{\\rm Wavelength spacing [\\AA]}")
    ax.set_ylabel("{\\rm Max relative error (CPU vs GPU)}")
    ax.legend()
    fig.savefig(joinpath(test_plotdir, "compare_cpu_gpu_broadening.pdf"), bbox_inches="tight")
    plt.close()
end

@testset "CPU/GPU broadening agreement across wavelength spacings" begin
    for i in eachindex(steps)
        @testset "step = $(steps[i]) Å" begin
            @test αs_error[i]       < 1e-10
            # CPU and GPU both use padded linear convolution with edge replication;
            # agreement is at floating-point precision
            @test rot_error[i]      < 1e-8
            @test rt_error[i]       < 1e-8
            @test rt_aniso_error[i] < 1e-8
            # Hirano chains two FFT convolutions (rotation + macro), ~2× single-kernel error
            @test rotmacro_error[i] < 5e-8
        end
    end
end
