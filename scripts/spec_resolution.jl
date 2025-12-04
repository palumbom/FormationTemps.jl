using Revise
using FormationTemps; FT = FormationTemps
using Korg
using ProgressMeter
using HDF5, Printf, JLD2
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib
plt.ioff()

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

ncolors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#A6761D", "#66A61E"]

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

# make the wavelength grid
buffer = 0.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.001)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# synthesis to get the alphas
αs = zeros(length(atm_gpu.zs), length(λs_korg))
αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

μ_v_mac = CUDA.zeros(Float64, length(zs)-1)
σ_v_mac = CUDA.zeros(Float64, length(zs)-1)

cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

# get cfunc for flux
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_stationary = Array(FT.get_flux(cfunc_flux_struct)')

# get disk integrated continuum
cfunc_flux_cont_struct = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_cont_stationary = Array(FT.get_flux(cfunc_flux_cont_struct)')

# set rotational and macroturbulence 
vsinis = range(1000.0, 1.0e4, step=1000)
ζ_rt = 3400.0

# set resolution grid
R_grid1 = range(5e3, 1e5, step=2e3)
R_grid2 = range(1e5 + step(R_grid1), 1e6, step=1e4)
R_grid = vcat([collect(R_grid1), collect(R_grid2)]...)

# set limb darkening
@load joinpath(FT.datdir, "ld_coeffs.jld2") u1 u2
@show u1
@show u2
# u1 = 0.4
# u2 = 0.26

# create color map
cmap = plt.get_cmap("viridis")
# norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=maximum(μs))
norm = mpl.colors.Normalize(vmin=minimum(vsinis)/1e3, vmax=maximum(vsinis)/1e3)
colors = cmap(norm(vsinis ./ 1e3))

# loop over vsini
@showprogress for k in eachindex(vsinis)
    # get the convolved flux
    flux_convolution = Array(FT.convolve_hirano_rotmacro(λs_korg, flux_stationary, vsinis[k], ζ_rt, u1, u2))
    flux_cont_convolution = Array(FT.convolve_hirano_rotmacro(λs_korg, flux_cont_stationary, vsinis[k], ζ_rt, u1, u2))
    flux_convolution_norm = Array(flux_convolution ./ flux_cont_convolution)[1,:]

    # get disk stuff 
    ρstar = 1.0
    istar = 90.0
    v0 = vsinis[k]
    Nϕ = 128
    μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

    # flatten, move to cpu
    idx = findall(x -> x .> zero(eltype(μs)), Array(μs))
    μs_cpu = Array(μs)[idx]
    dA_cpu = Array(dA)[idx]
    z_rot_cpu = Array(z_rot)[idx]

    # allocate output
    flux_integration = CUDA.zeros(Float64, length(λs_korg))
    flux_cont_integration = CUDA.zeros(Float64, length(λs_korg))

    # do the disk integration
    for i in eachindex(μs_cpu)
        # set the rotational velocity
        μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

        # get intensity stuff
        cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)

        tbc = cfunc_intensity_struct.cfunc_dt
        cfunc_int_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt, μs_cpu[i])
        flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]

        # now do continuum intensity
        cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)

        tbc_cont = cfunc_intensity_cont.cfunc_dt
        cfunc_int_cont_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc_cont, ζ_rt, μs_cpu[i])
        flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[i]
    end

    # convolve with radial tangential
    flux_integration .*= 2π
    flux_cont_integration .*= 2π

    # normalize
    flux_integration_norm = Array(flux_integration ./ flux_cont_integration)

    # convolve and resample
    oversampling = 6.0
    rmses = zeros(length(R_grid))
    maxes = zeros(length(R_grid))
    for i in eachindex(R_grid)
        # convolve
        new_wavs_convolution, new_flux_convolution = FT.convolve_instrument_gauss(λs_korg, flux_convolution_norm, new_res=R_grid[i], oversampling=oversampling)
        new_wavs_integration, new_flux_integration = FT.convolve_instrument_gauss(λs_korg, flux_integration_norm, new_res=R_grid[i], oversampling=oversampling)

        # get rmse
        rmses[i] = sqrt(sum((100 .* (new_flux_integration .- new_flux_convolution)).^2.0) / length(new_flux_integration))
        maxes[i] = maximum(abs.(100 .* (new_flux_integration .- new_flux_convolution)))
    end
    plt.plot(R_grid, maxes, label=L"v \sin i =\ " * latexstring(vsinis[k]), c=colors[k,:])
end

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(L"v \sin i\ {\rm [km\ s}^{-1} {\rm ]}")
plt.xlabel(L"{\rm Spectral\ Resolving\ Power}")
plt.ylabel(L"{\rm Max\ Flux\ Error\ [\%\ Continuum]}")
plt.tight_layout()
plt.gca().set_xscale("log")
# plt.gca().set_yscale("symlog")
plt.savefig("figures/resolution_scaling.pdf", bbox_inches="tight")
plt.clf(); plt.close()