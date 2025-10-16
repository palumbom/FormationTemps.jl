using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib
plt.ioff()

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

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

# get the formation temperature for a stationary star
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

# set rotational and macroturbulence grids 
vsinis = range(0.00, 10_000.0, step=2_000)
vmacs = range(0.0, 5_000, step=1_000)
vsinis_kms = vsinis ./ 1000
vmacs_kms = vmacs ./ 1000

# set limb darkening
u1 = 0.4
u2 = 0.26

# get a colormap 
cmap = plt.get_cmap("viridis")#colormaps.batlowk
norm = mpl.colors.Normalize(vmin=0.0, vmax=125.0)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

# set up a figure for flux
figsize=(15,15)
ticklabelsize = 24
plt.clf(); plt.close("all")
fig1, ax1 = plt.subplots(figsize=figsize)
ax1.set_xlabel(L"v \sin i \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax1.set_ylabel(L"\xi_{\rm RT} \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax1.set_xticks(vsinis_kms)
ax1.set_yticks(vmacs_kms)
ax1.xaxis.set_tick_params(labelsize=ticklabelsize)
ax1.yaxis.set_tick_params(labelsize=ticklabelsize)
ax1.set_xlim(first(vsinis_kms) - step(vsinis_kms)/1.1, last(vsinis_kms) + step(vsinis_kms)/1.8)
ax1.set_ylim(first(vmacs_kms) - step(vmacs_kms)/1.15, last(vmacs_kms) + step(vmacs_kms)/2.0)
# ax1.grid(false)

# set up a figure for temp
fig2, ax2 = plt.subplots(figsize=figsize)
ax2.set_xlabel(L"v \sin i \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax2.set_ylabel(L"\xi_{\rm RT} \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax2.set_xticks(vsinis_kms)
ax2.set_yticks(vmacs_kms)
ax2.xaxis.set_tick_params(labelsize=ticklabelsize)
ax2.yaxis.set_tick_params(labelsize=ticklabelsize)
ax2.set_xlim(first(vsinis_kms) - step(vsinis_kms)/1.1, last(vsinis_kms) + step(vsinis_kms)/1.8)
ax2.set_ylim(first(vmacs_kms) - step(vmacs_kms)/1.15, last(vmacs_kms) + step(vmacs_kms)/2.0)

wstr = "175%"
hstr = "175%"

mtrans = pyimport("matplotlib.transforms")
sx = 0.1 * (maximum(vsinis ./ 1000) - minimum(vsinis ./ 1000))
sy = 0.1 * (maximum(vmacs ./ 1000)  - minimum(vmacs ./ 1000))

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
    Nϕ = 64
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
        # allocate for output
        ints = zeros(length(λs_korg), length(μs))
        flux_integration = zeros(length(λs_korg))
        cfunc_flux_integration = zeros(length(zs)-1, length(λs_korg))

        # do the disk integration
        for i in eachindex(μs_cpu)
            # set the rotational velocity
            μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

            # get the intensity contribution function
            cfunc_int_i = FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)

            # convolve the cfunc with RT macroturbulence TODO
            σ_v_mac .= vmacs[j]
            cfunc_int_i_mac = Array(FT.convolve_wavelength_axis_gpu(cmem_mac, CuArray(λs_korg), CuArray(cfunc_int_i), μ_v_mac, σ_v_mac))

            # tabulate the intensity
            ints[:, i] .= sum(cfunc_int_i_mac, dims=1)'

            # add to the flux integral
            flux_integration .+= ints[:,i] .* dA_cpu[i]
            cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]
        end

        # convert units on disk integration
        flux_integration .*= 1e-8
        cfunc_flux_integration .*= 1e-8

        # get the convolution
        cfunc_flux_convolution = Array(FT.convolve_hirano_rotmacro(λs_korg, cfunc_flux_stationary, vsinis[k], vmacs[j], u1, u2))
        flux_convolution = dropdims(sum(cfunc_flux_convolution, dims=1), dims=1)
        
        # now get cumulative cfuncs 
        cum_cfunc_flux_integration = cumsum(cfunc_flux_integration, dims=1)
        cum_cfunc_flux_integration ./= maximum(cum_cfunc_flux_integration, dims=1)
        cum_cfunc_flux_convolution = cumsum(cfunc_flux_convolution, dims=1)
        cum_cfunc_flux_convolution ./= maximum(cum_cfunc_flux_convolution, dims=1)

        # loop over wavelength
        form_temp_integration = zeros(length(λs_korg))
        form_temp_convolution = zeros(length(λs_korg))
        for i in eachindex(λs_korg)
            xs = view(cum_cfunc_flux_integration, :, i)
            itp = FT.linear_interp(xs, elav(Ts))
            form_temp_integration[i] = itp(0.5)

            xs = view(cum_cfunc_flux_convolution, :, i)
            itp = FT.linear_interp(xs, elav(Ts))
            form_temp_convolution[i] = itp(0.5)
        end

        # overplot the flux
        resid_flux_pct = 100 .* (flux_integration .- flux_convolution) ./ flux_integration
        resid_temp_pct = 100 .* (form_temp_integration .- form_temp_convolution) ./ form_temp_integration

        # get rmse error 
        rmse_flux = round(sqrt(sum((flux_integration .- flux_convolution).^2.0) / length(flux_integration)), digits=3)
        rmse_temp = round(sqrt(sum((form_temp_integration .- form_temp_convolution).^2.0) / length(form_temp_integration)), digits=1)

        # inset axes for flux
        bbox = mtrans[:Bbox][:from_bounds](vsinis[k] / 1000 - sx/2, vmacs[j] / 1000  - sy/2, sx, sy)
        iax1 = inset.inset_axes(ax1, width=wstr, height=hstr, loc="center",
                               bbox_to_anchor=bbox, 
                               bbox_transform=ax1.transData, borderpad=0)
        iax1.plot(λs_korg, resid_flux_pct, c="tab:blue")
        iax1.set_frame_on(true)
        iax1.set_ylim(-55, 55)
        iax1.grid(false)
        # iax1.text(0.05, 0.05, L"\mathrm{RMSE} = %$rmse_flux", transform=iax1.transAxes, fontsize=12, va="bottom", ha="left")

        # inset axes for temperature 
        iax2 = inset.inset_axes(ax2, width=wstr, height=hstr, loc="center",
                               bbox_to_anchor=bbox, 
                               bbox_transform=ax2.transData, borderpad=0)
        # iax2.plot(λs_korg, resid_temp_pct, color=cmap(norm(rmse_temp)))#c="tab:blue")
        iax2.plot(λs_korg, resid_temp_pct, c="tab:blue")
        iax2.set_frame_on(true)
        iax2.set_ylim(-25, 25)
        iax2.grid(false)
        iax2.text(0.5, 0.05, L"\mathrm{RMSE} \approx %$rmse_temp \ \mathrm{K}", transform=iax2.transAxes, fontsize=12, va="bottom", ha="center")
                

        if (k == 1) & (j == 1)
            iax1.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax1.set_ylabel(L"{\rm \%\ Flux\ Error}")
            iax2.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax2.set_ylabel(L"{\rm \%\ } T_{1/2}\ {\rm Error}")
        elseif k == 1
            iax1.set_xticklabels([])
            iax2.set_xticklabels([])

            iax1.set_ylabel(L"{\rm \%\ Flux\ Error}")
            iax2.set_ylabel(L"{\rm \%\ } T_{1/2}\ {\rm Error}")
        elseif j == 1
            iax1.set_yticklabels([])
            iax2.set_yticklabels([])

            iax1.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax2.set_xlabel(L"{\rm Wavelength\ [\AA]}")
        else
            iax1.set_xticklabels([])
            iax1.set_yticklabels([])
            iax2.set_xticklabels([])
            iax2.set_yticklabels([])
        end

    end
end

fig1.tight_layout()
fig1.savefig("figures/big_plot_flux.pdf", bbox_inches="tight")


# axins = inset.inset_axes(ax2, width="5%",height="100%", loc="lower left",
#                          bbox_to_anchor=(1.05, 0., 1, 1),
#                          bbox_transform=ax2.transAxes,borderpad=0)
# fig2.colorbar(sm, cax=axins)
fig2.tight_layout()
fig2.savefig("figures/big_plot_temperature.pdf", bbox_inches="tight")
plt.clf(); plt.close("all");

