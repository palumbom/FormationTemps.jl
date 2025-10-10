using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")

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
buffer = 0.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.005)
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

# get the nominal answer
cfunc_flux_stationary = 2π .* FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_stationary = dropdims(sum(cfunc_flux_stationary, dims=1), dims=1)

cum_cfunc_flux_stationary = cumsum(cfunc_flux_stationary, dims=1)
cum_cfunc_flux_stationary ./= maximum(cum_cfunc_flux_stationary, dims=1)

form_temp_stationary = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cum_cfunc_flux_stationary, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_stationary[i] = itp(0.5)
end

# set equatorial velocities 
vsinis = range(0.00, 10_000, step=5_000)
vmacs = range(0.0, 10_000, step=5_000)

vsinis_kms = vsinis ./ 1000
vmacs_kms = vmacs ./ 1000

# set up a figure
# fig1, axs1 = plt.subplots(nrows=length(vsinis), ncols=length(vmacs), sharex=true, sharey=true)
fig, ax1 = plt.subplots(figsize=(8,8))
ax1.set_xlabel(L"v \sin i \ {\rm [km\ s}^{-1} {\rm ]}")
ax1.set_ylabel(L"\xi \ {\rm [km\ s}^{-1} {\rm ]}")
ax1.set_xticks(vsinis_kms)
ax1.set_yticks(vmacs_kms)
ax1.set_xlim(first(vsinis_kms) - step(vsinis_kms)/2, last(vsinis_kms) + step(vsinis_kms)/2)
ax1.set_ylim(first(vmacs_kms) - step(vmacs_kms)/2, last(vmacs_kms) + step(vmacs_kms)/2)

wstr = "100%"
hstr = "100%"

mtrans = pyimport("matplotlib.transforms")
sx = 0.1*(maximum(vsinis ./ 1000) - minimum(vsinis ./ 1000))
sy = 0.1*(maximum(vmacs ./ 1000)  - minimum(vmacs ./ 1000))

# loop over vsini
for k in eachindex(vsinis)
    @show k 

    # get disk stuff 
    ρstar = 1.0
    istar = 90.0
    A = 0.00711 * vsinis[k]
    B = 0.0
    C = 0.0
    v0 = vsinis[k]
    Nϕ = 16
    μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, A, B, C, v0, Nϕ)

    # flatten, move to cpu
    idx = findall(x -> x .> zero(eltype(μs)), Array(μs))
    μs_cpu = Array(μs)[idx]
    dA_cpu = Array(dA)[idx]
    z_rot_cpu = Array(z_rot)[idx]

    if vsinis[k] == 0.0
        z_rot_cpu .= 0.0
    end

    # loop over macro
    for j in eachindex(vmacs)
        @show j
        println()

        # allocate for output
        ints = zeros(length(λs_korg), length(μs))
        flux_rotating = zeros(length(λs_korg))
        cfunc_flux_rotating = zeros(length(zs)-1, length(λs_korg))

        for i in eachindex(μs_cpu)
            μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

            cfunc_int_i = FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)

            # convolve the cfunc with macroturbulence
            σ_v_mac .= vmacs[j]
            cfunc_int_i_mac = Array(FT.convolve_wavelength_axis_gpu(cmem_mac, CuArray(λs_korg), CuArray(cfunc_int_i), μ_v_mac, σ_v_mac))


            ints[:, i] .= sum(cfunc_int_i_mac, dims=1)'

            flux_rotating .+= ints[:,i] .* dA_cpu[i]
            cfunc_flux_rotating .+= cfunc_int_i_mac .* dA_cpu[i]
        end

        # get the flux and cfucn
        flux_rotating .*= 1e-8
        cfunc_flux_rotating .*= 1e-8

        # now get cumulative cfuncs 
        cum_cfunc_flux_rotating = cumsum(cfunc_flux_rotating, dims=1)
        cum_cfunc_flux_rotating ./= maximum(cum_cfunc_flux_rotating, dims=1)

        # loop over wavelength
        form_temp_rotating = zeros(length(λs_korg))
        for i in eachindex(λs_korg)
            xs = view(cum_cfunc_flux_rotating, :, i)
            itp = FT.linear_interp(xs, elav(Ts))
            form_temp_rotating[i] = itp(0.5)
        end

        # overplot the flux
        resid_pct = 100 .* (flux_rotating .- flux_stationary) ./ flux_rotating

        # inset axes
        bbox = mtrans[:Bbox][:from_bounds](vsinis[k] / 1000 - sx/2, vmacs[j] / 1000  - sy/2, sx, sy)
        iax = inset.inset_axes(ax1, width=wstr, height=hstr, loc="center",
                               bbox_to_anchor=bbox, 
                               bbox_transform=ax1.transData, borderpad=0)
        iax.plot(λs_korg, resid_pct, c="tab:blue")
        iax.set_xticks([])
        iax.set_yticks([])
        iax.set_frame_on(true)
        iax.set_ylim(-75, 75)

        # kplot = length(vsinis)-k+1
        # jplot = j
        # axs1[kplot,jplot].plot(λs_korg, resid_pct, c="tab:blue")
        # axs1[kplot,jplot].set_axis_off()
        # axs1[k,j].set_xlabel("Wavelength")
        # axs1[k,j].set_ylabel("Flux")
        # axs1[k,j].legend()

        # # overplot the temperature
        # fig, ax1 = plt.subplots()
        # ax1.plot(λs_korg, form_temp_stationary, c="k", ls="--", label="Stationary")
        # ax1.plot(λs_korg, form_temp_rotating, c="tab:blue", label="Solid Body Rotation")
        # ax1.set_xlabel("Wavelength")
        # ax1.set_ylabel("Formation Temperature")
        # ax1.legend()
        # fig.savefig("figures/temp_rotation.pdf", bbox_inches="tight")
        # plt.clf(); plt.close();
    end
end

fig.tight_layout()
fig.savefig("figures/big_plot_flux.pdf", bbox_inches="tight")
plt.show()
plt.clf(); plt.close();