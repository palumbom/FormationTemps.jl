using Revise
using FormationTemps; FT = FormationTemps
using Korg, LsqFit
using HDF5, Printf, JLD2
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using ProgressMeter

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pyconvert
using LaTeXStrings
mpl = plt.matplotlib

# matplotlib backend
mpl.use("QtAgg")
mpl.style.use(joinpath(FT.moddir, "fig.mplstyle"))
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
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.001)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
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
Npad = 2400
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device (anchored τ integrator when tau_ref is available)
α_ref = αs_cont[:, end]
gpu_mem = isempty(atm_gpu.τs) ? FT.GPUMemory(λs_korg, atm_gpu) : FT.GPUMemory(λs_korg, atm_gpu, α_ref)

# velocities
v_los_rot = CUDA.zeros(Float64, length(zs))
v_mic = CUDA.zeros(Float64, length(zs)) .+ 1200.0

cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)

# get the formation temperature for a stationary star
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, v_mic)
flux_stationary = Array(FT.get_flux(cfunc_flux_struct)')
cfunc_flux_stationary = cfunc_flux_struct.cfunc_dt
cum_cfunc_flux_stationary = Array(FT.get_cum_cfunc(cfunc_flux_struct))

cfunc_flux_cont_struct = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, v_mic)
cfunc_flux_cont_stationary = cfunc_flux_cont_struct.cfunc_dt
flux_cont_stationary = Array(FT.get_flux(cfunc_flux_cont_struct)')

form_temp_stationary = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cum_cfunc_flux_stationary, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_stationary[i] = itp(0.5)
end

# set rotational and macroturbulence grids
# vsinis = range(0.00, 10_000.0, step=2_000.0)
vsinis = range(0.00, 16_000.0, step=2_000.0)
vmacs = range(0.0, 10_000.0, step=2_000.0)
vsinis_kms = vsinis ./ 1000
vmacs_kms = vmacs ./ 1000

# set limb darkening
@load joinpath(FT.datdir, "ld_coeffs.jld2") u1 u2

# get a colormap
cmap = plt.get_cmap("viridis")#colormaps.batlowk
norm = mpl.colors.Normalize(vmin=0.0, vmax=125.0)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

# set up figures
figsize=(24,15)
# figsize=(15,15)
ticklabelsize = 24
plt.clf(); plt.close("all")
fig1, ax1 = plt.subplots(figsize=figsize)
ax1.set_xlabel(L"v \sin i \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax1.set_ylabel(L"\zeta_{\rm RT} \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax1.set_xticks(vsinis_kms)
ax1.set_yticks(vmacs_kms)
ax1.xaxis.set_tick_params(labelsize=ticklabelsize)
ax1.yaxis.set_tick_params(labelsize=ticklabelsize)
ax1.set_xlim(first(vsinis_kms) - step(vsinis_kms)/1.1, last(vsinis_kms) + step(vsinis_kms)/1.8)
ax1.set_ylim(first(vmacs_kms) - step(vmacs_kms)/1.15, last(vmacs_kms) + step(vmacs_kms)/2.0)

fig2, ax2 = plt.subplots(figsize=figsize)
ax2.set_xlabel(L"v \sin i \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax2.set_ylabel(L"\zeta_{\rm RT} \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax2.set_xticks(vsinis_kms)
ax2.set_yticks(vmacs_kms)
ax2.xaxis.set_tick_params(labelsize=ticklabelsize)
ax2.yaxis.set_tick_params(labelsize=ticklabelsize)
ax2.set_xlim(first(vsinis_kms) - step(vsinis_kms)/1.1, last(vsinis_kms) + step(vsinis_kms)/1.8)
ax2.set_ylim(first(vmacs_kms) - step(vmacs_kms)/1.15, last(vmacs_kms) + step(vmacs_kms)/2.0)

fig3, ax3 = plt.subplots(figsize=figsize)
ax3.set_xlabel(L"v \sin i \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax3.set_ylabel(L"\zeta_{\rm RT} \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax3.set_xticks(vsinis_kms)
ax3.set_yticks(vmacs_kms)
ax3.xaxis.set_tick_params(labelsize=ticklabelsize)
ax3.yaxis.set_tick_params(labelsize=ticklabelsize)
ax3.set_xlim(first(vsinis_kms) - step(vsinis_kms)/1.1, last(vsinis_kms) + step(vsinis_kms)/1.8)
ax3.set_ylim(first(vmacs_kms) - step(vmacs_kms)/1.15, last(vmacs_kms) + step(vmacs_kms)/2.0)

fig4, ax4 = plt.subplots(figsize=figsize)
ax4.set_xlabel(L"v \sin i \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax4.set_ylabel(L"\zeta_{\rm RT} \ {\rm [km\ s}^{-1} {\rm ]}", fontsize=24)
ax4.set_xticks(vsinis_kms)
ax4.set_yticks(vmacs_kms)
ax4.xaxis.set_tick_params(labelsize=ticklabelsize)
ax4.yaxis.set_tick_params(labelsize=ticklabelsize)
ax4.set_xlim(first(vsinis_kms) - step(vsinis_kms)/1.1, last(vsinis_kms) + step(vsinis_kms)/1.8)
ax4.set_ylim(first(vmacs_kms) - step(vmacs_kms)/1.15, last(vmacs_kms) + step(vmacs_kms)/2.0)

# parameters for inset axes
wstr = "100%"
# wstr = "175%"
hstr = "175%"

mtrans = pyimport("matplotlib.transforms")
sx = 0.1 * (maximum(vsinis ./ 1000) - minimum(vsinis ./ 1000))
sy = 0.1 * (maximum(vmacs ./ 1000)  - minimum(vmacs ./ 1000))

# allocate for output
flux_integration = CUDA.zeros(Float64, length(λs_korg))
flux_cont_integration = CUDA.zeros(Float64, length(λs_korg))
cfunc_flux_integration = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))
cfunc_flux_cont_integration = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))

# loop over vsini
for k in eachindex(vsinis)
    @show k

    # get disk stuff
    ρstar = 1.0
    istar = 90.0
    v0 = vsinis[k]
    Nϕ = 64
    μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

    # flatten, move to cpu
    # idx = findall(x -> x .> zero(eltype(μs)), Array(μs))
    μs_cpu = Array(μs)
    dA_cpu = Array(dA)
    z_rot_cpu = Array(z_rot)

    if vsinis[k] == 0.0
        z_rot_cpu .= 0.0
    end

    # loop over macro
    for j in eachindex(vmacs)
        # re-zero output
        flux_integration .= 0.0
        flux_cont_integration .= 0.0
        cfunc_flux_integration .= 0.0
        cfunc_flux_cont_integration .= 0.0

        # do the disk integration
        for i in eachindex(μs_cpu)
            μs_cpu[i] <= 0.0 && continue

            # set the rotational velocity
            v_los_rot .= z_rot_cpu[i] .* FT.c_ms

            # get intensity stuff
            cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], v_los_rot, v_mic)

            tbc = cfunc_intensity_struct.cfunc_dt
            cfunc_int_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, vmacs[j], μs_cpu[i])
            flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[i]
            cfunc_flux_integration .+= cfunc_int_i_mac .* dA_cpu[i]

            # now do continuum intensity
            cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs_cpu[i], v_los_rot, v_mic)

            tbc_cont = cfunc_intensity_cont.cfunc_dt
            cfunc_int_cont_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc_cont, vmacs[j], μs_cpu[i])
            flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[i]
            cfunc_flux_cont_integration .+= cfunc_int_cont_i_mac .* dA_cpu[i]
        end

        # 2pi
        flux_integration .*= 2π
        flux_cont_integration .*= 2π

        # get the convolution
        cfunc_flux_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_stationary, vsinis[k], vmacs[j], u1, u2))
        flux_convolution = dropdims(sum(cfunc_flux_convolution, dims=1), dims=1)
        cfunc_flux_cont_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_cont_stationary, vsinis[k], vmacs[j], u1, u2))
        flux_cont_convolution = dropdims(sum(cfunc_flux_cont_convolution, dims=1), dims=1)

        # now get cumulative cfuncs
        cum_cfunc_flux_integration = Array(cumsum(cfunc_flux_integration, dims=1))
        cum_cfunc_flux_integration ./= maximum(cum_cfunc_flux_integration, dims=1)
        cum_cfunc_flux_convolution = Array(cumsum(cfunc_flux_convolution, dims=1))
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

        # get normalized flux
        flux_integration_norm = Array(flux_integration ./ flux_cont_integration)
        flux_convolution_norm = Array(flux_convolution ./ flux_cont_convolution)

        # overplot the flux
        flux_err_pct = 100 .* (Array(flux_integration) .- flux_convolution) ./ Array(flux_integration)
        flux_err_cont_pct = 100 .* (flux_integration_norm .- flux_convolution_norm) ./ flux_integration_norm
        temp_err_pct = 100 .* (form_temp_integration .- form_temp_convolution) ./ form_temp_integration
        temp_err = form_temp_integration .- form_temp_convolution

        # get rmse error
        rmse_flux = round(sqrt(sum((Array(flux_integration) .- flux_convolution).^2.0) / length(flux_integration)), digits=3)
        rmse_flux_cont = round(sqrt(sum((100 .* flux_integration_norm .- 100 .* flux_convolution_norm).^2.0) / length(flux_integration_norm)), digits=3)
        rmse_temp = round(sqrt(sum((form_temp_integration .- form_temp_convolution).^2.0) / length(form_temp_integration)), digits=1)
        max_flux_error = round(maximum(abs.(100 .* (flux_integration_norm .- flux_convolution_norm))), digits=1)

        # inset axes
        bbox = mtrans.Bbox.from_bounds(vsinis[k] / 1000 - sx/2, vmacs[j] / 1000  - sy/2, sx, sy)
        iax1 = inset.inset_axes(ax1, width=wstr, height=hstr, loc="center",
                               bbox_to_anchor=bbox,
                               bbox_transform=ax1.transData, borderpad=0)
        # iax1.plot(λs_korg, flux_err_pct, c="tab:blue")
        iax1.plot(λs_korg, flux_err_cont_pct, c="tab:blue")
        iax1.set_frame_on(true)
        iax1.set_ylim(-15, 15)
        iax1.grid(false)
        iax1.text(0.5, 0.05, L"\mathrm{Max\ Error} \approx %$max_flux_error\mathrm{\%\ Cont.}", transform=iax1.transAxes, fontsize=12, va="bottom", ha="center")

        iax2 = inset.inset_axes(ax2, width=wstr, height=hstr, loc="center",
                               bbox_to_anchor=bbox,
                               bbox_transform=ax2.transData, borderpad=0)
        # iax2.plot(λs_korg, temp_err_pct, color=cmap(norm(rmse_temp)))#c="tab:blue")
        # iax2.plot(λs_korg, temp_err_pct, c="tab:blue")
        iax2.plot(λs_korg, temp_err, c="tab:blue")
        iax2.set_frame_on(true)
        iax2.set_ylim(-150, 150)
        iax2.grid(false)
        iax2.text(0.5, 0.05, L"\mathrm{RMSE} \approx %$rmse_temp \ \mathrm{K}", transform=iax2.transAxes, fontsize=12, va="bottom", ha="center")

        iax3 = inset.inset_axes(ax3, width=wstr, height=hstr, loc="center",
                               bbox_to_anchor=bbox,
                               bbox_transform=ax3.transData, borderpad=0)
        iax3.plot(λs_korg, flux_integration_norm, c="k", label=L"{\rm Integration}")
        iax3.plot(λs_korg, flux_convolution_norm, c="tab:blue", label=L"{\rm Convolution}")
        iax3.set_frame_on(true)
        iax3.set_ylim(0.1, 1.1)
        iax3.grid(false)

        # inset axes for temperature
        iax4 = inset.inset_axes(ax4, width=wstr, height=hstr, loc="center",
                               bbox_to_anchor=bbox,
                               bbox_transform=ax4.transData, borderpad=0)
        iax4.plot(λs_korg, form_temp_integration, c="k", label=L"{\rm Integration}")
        iax4.plot(λs_korg, form_temp_convolution, c="tab:blue", label=L"{\rm Convolution}")
        iax4.set_frame_on(true)
        iax4.set_ylim(4200, 6200)
        iax4.grid(false)

        # create legend
        if (k == length(vsinis)) & (j == length(vmacs))
            iax3.legend(loc="lower center")
            iax4.legend(loc="lower center")
        end

        # axis labels
        if (k == 1) & (j == 1)
            iax1.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax1.set_ylabel(L"{\rm \%\ Flux\ Error}")
            iax2.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax2.set_ylabel(L"T_{1/2}\ {\rm Error\ [K]}")
            iax3.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax3.set_ylabel(L"{\rm Rel.\ Flux}")
            iax4.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax4.set_ylabel(L"T_{1/2}\ {\rm [K]}")
        elseif k == 1
            iax1.set_xticklabels([])
            iax2.set_xticklabels([])
            iax3.set_xticklabels([])
            iax4.set_xticklabels([])

            iax1.set_ylabel(L"{\rm \%\ Flux\ Error}")
            iax2.set_ylabel(L"T_{1/2}\ {\rm Error\ [K]}")
            iax3.set_ylabel(L"{\rm Rel.\ Flux}")
            iax4.set_ylabel(L"T_{1/2}\ {\rm [K]}")
        elseif j == 1
            iax1.set_yticklabels([])
            iax2.set_yticklabels([])
            iax3.set_yticklabels([])
            iax4.set_yticklabels([])

            iax1.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax2.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax3.set_xlabel(L"{\rm Wavelength\ [\AA]}")
            iax4.set_xlabel(L"{\rm Wavelength\ [\AA]}")
        else
            iax1.set_xticklabels([])
            iax1.set_yticklabels([])
            iax2.set_xticklabels([])
            iax2.set_yticklabels([])
            iax3.set_xticklabels([])
            iax3.set_yticklabels([])
            iax4.set_xticklabels([])
            iax4.set_yticklabels([])
        end
    end
end

#  write them out
fig1.tight_layout()
fig1.savefig("figures/big_plot_flux.pdf", bbox_inches="tight")

# axins = inset.inset_axes(ax2, width="5%",height="100%", loc="lower left",
#                          bbox_to_anchor=(1.05, 0., 1, 1),
#                          bbox_transform=ax2.transAxes,borderpad=0)
# fig2.colorbar(sm, cax=axins)
fig2.tight_layout()
fig2.savefig("figures/big_plot_temperature.pdf", bbox_inches="tight")

fig3.tight_layout()
fig3.savefig("figures/other_big_plot_flux.pdf", bbox_inches="tight")

fig4.tight_layout()
fig4.savefig("figures/other_big_plot_temperature.pdf", bbox_inches="tight")

plt.clf(); plt.close("all");
