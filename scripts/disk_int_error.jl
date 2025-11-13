using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using ProgressMeter
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                            \\usepackage{mathrsfs}")

# python interpolation for matplotlib stuff
interp1d = pyimport("scipy.interpolate").interp1d

# set colormaps
img_cmap = "viridis"
μ_cmap = "autumn"

# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

# make plotdir
plotdir = joinpath(pwd(), "figures")
!isdir(plotdir) && mkdir(plotdir)

# get the linelist
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]

# cut on species
linelist = linelist[specs .== "Fe I"]

# get the Fe I 6301 & 6302 lines (just cuz)
wls = [l.wl for l in linelist] 
idx1 = findfirst(x -> x * 1e8 .>= 6301, wls)
idx2 = findfirst(x -> x * 1e8 .>= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# make the wavelength grid
λs_korg = range(first(wls) - 2.0, last(wls) + 2.0, step=0.01)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
marcs_atm = FT.get_marcs_atm(5777.0, 4.44, A_X, n_layers=56)
τ_500 = Korg.get_tau_refs(marcs_atm)
zs = Korg.get_zs(marcs_atm)
Ts = Korg.get_temps(marcs_atm)
ne = Korg.get_electron_number_densities(marcs_atm)
nd = Korg.get_number_densities(marcs_atm)

# make my atmosphere 
atm_gpu = FT.AtmosphereGPU(marcs_atm)
zs = atm_gpu.zs
Ts = atm_gpu.Ts
τ5000 = atm_gpu.τs

# synthesis to get the alphas
αs = zeros(length(atm_gpu.zs), length(λs_korg))
αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
# FT.compute_alpha!(αs, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0

# get the nominal answer
cfunc_flux = 2π .* FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux = dropdims(sum(cfunc_flux, dims=1), dims=1)

@btime FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, 1.0, μ_v, σ_v)

#= # get disk stuff 
ρstar = 1.0
istar = 90.0
v0 = 0.0
Nϕ = 2 .^(range(2, 8, step=1))
Nϕ = [8, 16, 32, 48, 64, 96, 128, 156, 181, 212, 256, 512]

mean_pct_error_flux = zeros(length(Nϕ))
ntiles_real = zeros(length(Nϕ))

# allocate for output
flux_test = zeros(length(λs_korg))

for j in eachindex(Nϕ)
    @show Nϕ[j]

    # do spherical trig
    μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ[j])

    # flatten, move to cpu
    idx = findall(x -> x .> zero(eltype(μs)), Array(μs))
    μs_cpu = view(Array(μs), idx)
    dA_cpu = view(Array(dA), idx)
    z_rot_cpu = view(Array(z_rot), idx)

    @show sum(dA_cpu)
    
    ntiles_real[j] = length(μs_cpu)

    flux_test .= 0.0

    @showprogress for i in eachindex(μs_cpu)
        # μ_v .= z_rot_cpu[i] .* FT.c_ms
        cfunc_int_i = FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v, σ_v)
        flux_test .+= sum(cfunc_int_i, dims=1)' .* dA_cpu[i]
    end
    println()

    # test the flux
    flux_test .*= 1e-8 .* π ./ sum(dA_cpu) 
    mean_pct_error_flux[j] = mean(100 .* (flux .- flux_test) ./ flux)

    # # integrate the cfuncs 
    # cfunc_flux_test = π .* 1e-8 .* cfuncs_int ./ sum(dA_cpu) 
    # @show extrema(cfunc_flux_test ./ cfunc_flux)
    # println()
end

plt.scatter(Nϕ, mean_pct_error_flux, s=20)
plt.xscale("symlog")
plt.xlabel("Number of latitude tiles")
plt.ylabel("Mean % Error")
plt.savefig("figures/disk_int_error.pdf", bbox_inches="tight")
plt.show() =#