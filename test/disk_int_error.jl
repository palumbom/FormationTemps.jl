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

λs_korg = range(first(wls) - 1.0, last(wls) + 1.0, step=0.01)

A_X     = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs      = atm_gpu.zs

αs      = zeros(length(zs), length(λs_korg))
αs_cont = zeros(length(zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

Nλ   = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem    = FT.ConvolutionMemory(Nλ, Natm, Npad)
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0

# reference: direct flux (no disk integration)
cfunc_flux_ref = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux_ref       = Array(FT.get_flux(cfunc_flux_ref))

# disk integration at two resolutions
Nϕ_vals = [16, 32]

mean_pct_error = zeros(length(Nϕ_vals))
max_pct_error  = zeros(length(Nϕ_vals))

ρstar = 1.0
istar = 90.0
v0    = 0.0

for j in eachindex(Nϕ_vals)
    μs, dA, z_rot, _ = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ_vals[j])

    idx       = findall(x -> x > zero(eltype(μs)), Array(μs))
    μs_cpu    = view(Array(μs), idx)
    dA_cpu    = view(Array(dA), idx)

    flux_disk = CUDA.zeros(Float64, length(λs_korg))
    @showprogress for i in eachindex(μs_cpu)
        cfunc_i = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v, σ_v)
        flux_disk .+= FT.get_intensity(cfunc_i) .* dA_cpu[i]
    end

    mean_pct_error[j] = mean(abs.(100.0 .* (flux_ref .- Array(flux_disk)) ./ flux_ref))
    max_pct_error[j]  = maximum(abs.(100.0 .* (flux_ref .- Array(flux_disk)) ./ flux_ref))
end

@testset "Disk integration error vs reference flux" begin
    # Nϕ = 16: coarse — allow up to 5% max error
    @test mean_pct_error[1] < 2.0
    @test max_pct_error[1]  < 5.0
    # Nϕ = 32: finer — tighter tolerance
    @test mean_pct_error[2] < 1.0
    @test max_pct_error[2]  < 2.0
    # error should decrease with increasing Nϕ
    @test mean_pct_error[2] < mean_pct_error[1]
end

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.pyplot.style.use(joinpath(FT.moddir, "fig.mplstyle"))
    plt.ioff()
    fig, ax = plt.subplots()
    ax.scatter(Nϕ_vals, mean_pct_error, s=20, label="{\\rm Mean abs. error}")
    ax.scatter(Nϕ_vals, max_pct_error,  s=20, label="{\\rm Max abs. error}")
    ax.set_xlabel(raw"{\rm Number of latitude tiles }$N_\phi$")
    ax.set_ylabel("{\\rm Percent error vs direct flux}")
    ax.legend()
    fig.savefig(joinpath(test_plotdir, "disk_int_error.pdf"), bbox_inches="tight")
    plt.close()
end
