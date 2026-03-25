using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using ProgressMeter

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pyconvert
using LaTeXStrings
mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

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
buffer = 1.5
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.005)
cont_idx = findfirst(x -> x .>= wls[2] + 0.1, λs_korg)#findfirst(x -> x .>= 6301.3, λs_korg)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
τ_500 = atm_gpu.τs
zs = atm_gpu.zs
Ts = atm_gpu.Ts
τ5000 = atm_gpu.τs

# synthesis to get the alphas
αs = zeros(length(atm_gpu.zs), length(λs_korg))
αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 240
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# loop over mus
μs = range(0.1, 1.0, step=0.1)
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0
cfuncs = zeros(length(zs)-1, length(λs_korg), length(μs))
cfuncs_cum = zeros(length(zs)-1, length(λs_korg), length(μs))
intensities = zeros(length(λs_korg), length(μs))
continuum = zeros(length(λs_korg), length(μs))

for i in eachindex(μs)
    cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    cfuncs[:,:,i] .= Array(cfunc_intensity_struct.cfunc_dt)
    cfuncs_cum[:,:,i] .= Array(FT.get_cum_cfunc(cfunc_intensity_struct))
    intensities[:,i] .= Array(FT.get_intensity(cfunc_intensity_struct))

    cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    continuum[:,i] .= Array(FT.get_intensity(cfunc_intensity_cont))
end

# get flux and flux cfunc
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux_disk_integrated = Array(FT.get_flux(cfunc_flux_struct))
cfunc_flux = Array(cfunc_flux_struct.cfunc_dt)
cfunc_flux_cum = Array(FT.get_cum_cfunc(cfunc_flux_struct))

# now get cumulative contribution functions
cum_cfuncs_norm = cfuncs_cum
cum_cfunc_flux_norm = cfunc_flux_cum

# now compute the formation temperature
form_temps_intensity = zeros(length(λs_korg), length(μs))
form_temps_flux = zeros(length(λs_korg))

for i in eachindex(λs_korg)
    local xs = view(cum_cfunc_flux_norm, :, i)
    local itp = FT.linear_interp(xs, elav(Ts))
    form_temps_flux[i] = itp(0.5)
end

for i in eachindex(λs_korg)
    for j in eachindex(μs)
        local xs = view(cum_cfuncs_norm, :, i, j)
        local itp = FT.linear_interp(xs, elav(Ts))
        form_temps_intensity[i,j] = itp(0.5)
    end
end

# get limits and such
max_val = maximum(abs.(cfuncs))
exponent = floor(Int, log10(max_val))
cb_lims = [minimum(cfuncs), round_to_power(maximum(cfuncs))] ./ 10^(exponent)
# cb_lims = [minimum(cfuncs), 2.5e13] ./ 10^(exponent)

max_val_cflux = maximum(abs.(cfunc_flux))
exponent_cflux = floor(Int, log10(max_val_cflux))
lims_cflux = [minimum(cfunc_flux), round_to_power(maximum(cfunc_flux))] ./ 10^(exponent_cflux)

max_val_int = maximum(abs.(intensities))
exponent_int = floor(Int, log10(max_val_int))
lims_int = [minimum(intensities), round_to_power(maximum(intensities))] ./ 10^(exponent_int)

max_val_flux = maximum(abs.(flux_disk_integrated))
exponent_flux = floor(Int, log10(max_val_flux))
lims_flux = [minimum(flux_disk_integrated), round_to_power(maximum(flux_disk_integrated))] ./ 10^(exponent_flux)

# now plot em
cmap = plt.get_cmap(μ_cmap)
# norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=maximum(μs))
norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=1.075)
colors = pyconvert(Array, cmap(norm(μs)))

# without flux
fig, ax1 = plt.subplots()
for i in eachindex(μs)
    ax1.plot(λs_korg, intensities[:,i] ./ 10^exponent_int, c=colors[i,:], lw=1.75)
end

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"\mu")

ax1.set_xlim(first(wls) - 0.75, last(wls) + 0.75)
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")

ax1.set_ylabel(L"I_\nu^+(\mu)\ {\rm [10^{%$exponent_int}\ erg\ s ^{-1} \ cm ^{-4} \ \AA ^{-1} \ sr ^{-1} ]}")
fig.savefig(joinpath(plotdir, "intensity_vs_limb_angle.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# with flux
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for i in eachindex(μs)
    ax1.plot(λs_korg, intensities[:,i] ./ 10^exponent_int, c=colors[i,:], lw=1.75)
end
ax2.plot(λs_korg, flux_disk_integrated , c="k", label=L"{\rm Flux}")

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"\mu")

ax1.set_xlim(first(wls) - 0.75, last(wls) + 0.75)
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")

ax1.set_ylabel(L"I_\nu^+(\mu)\ {\rm [10^{%$exponent_int}\ erg\ s ^{-1} \ cm ^{-4} \ \AA ^{-1} \ sr ^{-1} ]}")
fig.savefig(joinpath(plotdir, "intensity_vs_limb_angle_with_flux.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# plot the ratios of the intensities
fig, ax1 = plt.subplots()
for i in eachindex(μs)
    i == 1 && continue
    plt.plot(λs_korg, intensities[:,i] ./ continuum[:,i], c=colors[i,:], lw=1.75)
end

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"\mu")

ax1.set_xlim(first(wls) - 0.75, last(wls) + 0.75)
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")

# ax1.set_ylabel(L"I_\nu\," * offset * L"{\rm\, (erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1} \ sr ^{-1} )}")
ax1.set_ylabel(L"I_\nu(\mu)\ /\ I^{\rm cont}_\nu(\mu)")
fig.savefig(joinpath(plotdir, "intensity_continuum_normalized.pdf"), bbox_inches="tight")
plt.clf(); plt.close()


# now plot the contribution functions
μ_vals_to_plot = [1.0, 0.6, 0.3, 0.1]

# make three panels
fig, axs = plt.subplots(nrows=1, ncols=length(μ_vals_to_plot), sharey=true, figsize=(18.2, 4.2))#, layout="compressed")
ax1 = axs[0]

idx1 = findfirst(x -> x .>= first(wls) - 0.75, λs_korg)
idx2 = findfirst(x -> x .>= last(wls) + 0.75, λs_korg)

z_grid = elav(zs)
τ_grid = elav(τ_500)
extent = [λs_korg[idx1], λs_korg[idx2], first(τ_500), last(τ_500)]

vmin = first(cb_lims)
vmax = last(cb_lims)
# vmax = func(6)

function cmap_forward(x)
    return sqrt.(x)
end

function cmap_inverse(x)
    return x.^2.0
end

# norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
norm = mpl.colors.FuncNorm((cmap_forward, cmap_inverse), vmin=vmin, vmax=vmax)

xedges = view(λs_korg, idx1:idx2)
yedges = log10.(elav(τ_500))
yedges2 = elav(zs ./ 1e7)

imgs = []

# fig.colorbar(im, ax=axes.ravel().tolist())
for i in eachindex(μ_vals_to_plot)
    # !(i in μ_vals_to_plot) && continue
    μ_idx = findfirst(μs .== μ_vals_to_plot[i])

    # get view of cfunc
    cfunc_view = view(cfuncs,:,idx1:idx2,μ_idx)  ./ 10^(exponent)

    # img = ax3.imshow(cfunc_view, aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    img = axs[i - 1].pcolormesh(xedges, yedges2, cfunc_view,
                            shading="gouraud", cmap=img_cmap,
                            edgecolors="none", norm=norm,
                            rasterized=true)

    push!(imgs, img)
    axs[i - 1].axvline(xedges[cont_idx - idx1], c="white", ls=":", lw=2.5)

    # axs[i].set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
    # ax3.set_ylabel(L"{\rm \log _{10} (\tau_{5000})}")
    # ax3.set_ylabel(L"{\rm Physical\ Height\ [Mm]}")
    mu_val = string(μ_vals_to_plot[i])
    axs[i - 1].set_title(L"\mu = %$mu_val")

    local fwd = interp1d(yedges2, yedges, fill_value="extrapolate")
    local inv = interp1d(yedges, yedges2, fill_value="extrapolate")

    # ax3_right = ax3.secondary_yaxis("right", functions=(fwd, inv))
    # ax3_right.set_ylabel(L"{\rm Physical\ Height\ [Mm]}")
    # ax3_right.set_ylabel(L"{\rm \log _{10} (\tau_{5000})}")
    # ax3_right.yaxis.set_ticks([0, -1, -2, -3, -4])
end

fwd = interp1d(elav(zs ./ 1e7), elav(log10.(τ_500)), fill_value="extrapolate")
inv = interp1d(elav(log10.(τ_500)), elav(zs ./ 1e7), fill_value="extrapolate")

ax1_b1 = ax1.twinx()
ax1_b1.yaxis.set_ticks_position("left")
ax1_b1.yaxis.set_label_position("left")
ax1_b1.spines["left"].set_position(("axes", -0.3))
ax1_b1.set_frame_on(true)
ax1_b1.patch.set_visible(false)
for sp in ax1_b1.spines.values()
    sp.set_visible(false)
end
ax1_b1.spines["left"].set_visible(true)
new_yticks = [2, 1, 0, -1, -2, -3, -4, -5]
ax1_b1.set_yticks(inv(new_yticks))
ax1_b1.set_yticklabels(latexstring.(new_yticks))
ax1_b1.set_ylabel(L"{\rm \log _{10} (\tau_{5000})}", labelpad=8)
ax1_b1.set_ylim(ax1.get_ylim()...)
ax1_b1.grid(false)

fwd = interp1d(elav(zs ./ 1e7), elav(Ts), fill_value="extrapolate")
inv = interp1d(elav(Ts), elav(zs ./ 1e7), fill_value="extrapolate")

ax1_b2 = ax1.twinx()
ax1_b2.yaxis.set_ticks_position("left")
ax1_b2.yaxis.set_label_position("left")
ax1_b2.spines["left"].set_position(("axes", -0.6))
ax1_b2.set_frame_on(true)
ax1_b2.patch.set_visible(false)
for sp in ax1_b2.spines.values()
    sp.set_visible(false)
end
ax1_b2.spines["left"].set_visible(true)
new_yticks = [9000, 6250, 5500, 5000, 4750, 4500, 4250]
ax1_b2.set_yticks(inv(new_yticks))
ax1_b2.set_yticklabels(latexstring.(new_yticks))
ax1_b2.set_ylabel(L"{\rm Temperature\ [K]}", labelpad=8)
ax1_b2.set_ylim(ax1.get_ylim()...)
ax1_b2.grid(false)

fig.supxlabel(L"{\rm Air\ Wavelength\ [\AA]}", y=-0.02, x=0.45)
axs[0].set_ylabel(L"{\rm Physical\ Height\ [Mm]}")
fig.subplots_adjust(wspace=0.05)

cb = fig.colorbar(imgs[end], ax=axs, pad=0.01)
cb.set_label(L"C_\nu(t_\nu, \mu)\ d t_\nu\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-4} \ \AA ^{-1} \ sr ^{-1} ]}", labelpad=10.0)
cb.ax.xaxis.set_label_position("top")

fig.savefig(joinpath(plotdir, "cfunc_mus.pdf"), bbox_inches="tight", dpi=100)
plt.clf(); plt.close()


# plot slices through the contribution function at different limb angles
fig, ax1 = plt.subplots(figsize=1.1 .* (7.2, 5.6))
fig.subplots_adjust(bottom=0.3)

ax2 = ax1.twinx()
ax2.plot(elav(zs) ./ 1e7 , cfunc_flux[:,cont_idx] ./ 10^(exponent_cflux) , c="k", label=L"{\rm Flux}")
for i in eachindex(μs)
    # local xs = elav(τ_500)
    local xs = elav(zs) ./ 1e7
    local ys = view(cfuncs,:,cont_idx,i) ./ 10^(exponent)

    mu_val = μs[i]
    ax1.plot(xs, ys, c=colors[i,:], lw=1.75, label=L"\mu = %$mu_val")
end

ax1.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
ax1.set_xlim(-1.25, 7.25)

fwd = interp1d(elav(zs ./ 1e7), elav(log10.(τ_500)), fill_value="extrapolate")
inv = interp1d(elav(log10.(τ_500)), elav(zs ./ 1e7), fill_value="extrapolate")

# ax1_b1 = ax1.secondary_xaxis("top", functions=(fwd, inv))
# ax1_b1.xaxis.set_ticks([2, 1, 0, -1, -2, -3, -4, -5])
# ax1_b1.set_xlabel(L"{\rm \log _{10} (\tau_{5000})}", labelpad=10)

ax1_b1 = ax1.twiny()
ax1_b1.xaxis.set_ticks_position("top")
ax1_b1.xaxis.set_label_position("top")
ax1_b1.spines["top"].set_position(("axes", +1.0))#-0.25))
ax1_b1.set_frame_on(true)
ax1_b1.patch.set_visible(false)
for sp in ax1_b1.spines.values()
    sp.set_visible(false)
end
new_xticks = [2, 1, 0, -1, -2, -3, -4, -5]
ax1_b1.spines["top"].set_visible(true)
ax1_b1.set_xticks(inv(new_xticks))
ax1_b1.set_xticklabels(latexstring.(new_xticks))
ax1_b1.set_xlabel(L"{\rm \log _{10} (\tau_{5000})}", labelpad=10)
ax1_b1.set_xlim(ax1.get_xlim()...)
ax1_b1.grid(false)

fwd = interp1d(elav(zs ./ 1e7), elav(Ts), fill_value="extrapolate")
inv = interp1d(elav(Ts), elav(zs ./ 1e7), fill_value="extrapolate")

ax1_b2 = ax1.twiny()
ax1_b2.xaxis.set_ticks_position("top")
ax1_b2.xaxis.set_label_position("top")
ax1_b2.spines["top"].set_position(("axes", +1.25))#-0.5))
ax1_b2.set_frame_on(true)
ax1_b2.patch.set_visible(false)
for sp in ax1_b2.spines.values()
    sp.set_visible(false)
end
ax1_b2.spines["top"].set_visible(true)
new_xticks = [9000, 6250, 5500, 5000, 4750, 4500, 4250]
ax1_b2.set_xticks(inv(new_xticks))
ax1_b2.set_xticklabels(latexstring.(new_xticks))
ax1_b2.set_xlabel(L"{\rm Temperature\ [K]}", labelpad=10)
ax1_b2.set_xlim(ax1.get_xlim()...)
ax1_b2.grid(false)

wav_val = string(round(λs_korg[cont_idx], digits=1))
ax1.set_xlabel(L"{\rm Physical\ Height\ [Mm]}")
ax1.set_ylabel(L"C_{\nu}(t_\nu, \mu)\ d t_\nu\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-4} \ \AA ^{-1} \ sr ^{-1} ]}")
ax2.set_ylabel(L"\mathscr{C}_{\nu}(t_\nu)\ d t_\nu\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-4} \ \AA ^{-1}]}")
ax1.legend()

# ax1.set_ylim(cb_lims)
# ax2.set_ylim(lims_cflux)

derp1 = diff(pyconvert(Vector{Float64}, ax1.get_yticks()))
derp2 = diff(pyconvert(Vector{Float64}, ax2.get_yticks()))

# ax1.set_ylim(cb_lims[1], cb_lims[2] + derp2[end])
# ax2.set_ylim(lims_cflux[1], lims_cflux[2] + derp2[end])

ax1.set_yticks(range(cb_lims[1], cb_lims[2] + derp2[end], length=5))
ax2.set_yticks(range(lims_cflux[1], lims_cflux[2] + derp2[end], length=5))

fig.savefig(joinpath(plotdir, "cont_at_lambda.pdf"), bbox_inches="tight")
plt.clf(); plt.close()


# plot the cumulative contribution functions
cmap = plt.get_cmap(μ_cmap)
# norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=maximum(μs))
norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=1.075)
colors = pyconvert(Array, cmap(norm(μs)))

fig, ax1 = plt.subplots()
ax1.plot(elav(Ts), cum_cfunc_flux_norm[:,cont_idx], c="k", label=L"{\rm Flux}", zorder=length(μs) + 1)
for i in eachindex(μs)
    mui = μs[i]
    ax1.plot(elav(Ts), cum_cfuncs_norm[:,cont_idx, i], c=colors[i,:])#, label=L"\mu = %$mui")
end
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"\mu")

itp1 = FT.linear_interp(cum_cfunc_flux_norm[:,cont_idx], elav(Ts))
itp2 = FT.linear_interp(cum_cfuncs_norm[:,cont_idx, length(μs)], elav(Ts))

x_data1 = itp1(0.5)
x_data2 = itp2(0.5)

y0, y1 = ax1.get_ylim()
y_data = 0.5
yfrac = (y_data - y0) / (y1 - y0)

x0, x1 = ax1.get_xlim()
xfrac1 = (x_data1 - x0) / (x1 - x0)
xfrac2 = (x_data2 - x0) / (x1 - x0)

ax1.axvline(x_data1, ls="--", c="k", ymax=yfrac)
ax1.axvline(x_data2, ls="--", c=colors[end,:], ymax=yfrac)

ax1.axhline(y_data, ls="--", c="k", xmax=xfrac1)
ax1.axhline(y_data, ls="--", c=colors[end,:], xmax=xfrac2)

ax1.set_xlabel(L"{\rm Temperature\ [K]}")
ax1.set_ylabel(L"{\rm Normalized\ Cumulative\ Cont.\ Fn.}")
ax1.legend()

fig.tight_layout()
fig.savefig(joinpath(plotdir, "cum_cfunc_comparison.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# compare flux and intensity formation temperatures
fig, ax1 = plt.subplots()
for i in eachindex(μs)
    mu_val = μs[i]
    ax1.plot(λs_korg, form_temps_intensity[:,i],  c=colors[i,:])#, label=L"\mu = %$mu_val")
end
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"\mu")

ax1.plot(λs_korg, form_temps_flux, c="k", label=L"{\rm Flux}", zorder=length(μs) + 1)
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
ax1.legend()#bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

idx1 = findfirst(x -> x .>= first(wls) - 1.25, λs_korg)
idx2 = findfirst(x -> x .>= last(wls) + 1.25, λs_korg)
ax1.set_xlim(λs_korg[idx1], λs_korg[idx2])

fig.tight_layout()
fig.savefig(joinpath(plotdir, "form_temp_flux_vs_intensity.pdf"), bbox_inches="tight")
plt.clf(); plt.close()



# do the same with residuals
fig, axes = plt.subplots(nrows=2, ncols=2, width_ratios=[20,1], height_ratios=[3,1])
ax1 = axes[0, 0]
ax2 = axes[1, 0]
ax3 = axes[0, 1]
ax4 = axes[1, 1]
for i in eachindex(μs)
    mu_val = μs[i]
    ax1.plot(λs_korg, form_temps_intensity[:,i],  c=colors[i,:])#, label=L"\mu = %$mu_val")
end
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, cax=ax3)
cbar.set_label(L"\mu")
ax3.grid(false)

ax1.plot(λs_korg, form_temps_flux, c="k", label=L"{\rm Flux}", zorder=length(μs) + 1)
ax2.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
ax2.set_ylabel(L"{\rm Error\ in}\ T_{1/2}\ {\rm [K]}")
ax1.legend()#bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

ax2.plot(λs_korg, form_temps_flux .- form_temps_intensity[:,length(μs)], c="k")

idx1 = findfirst(x -> x .>= first(wls) - 1, λs_korg)
idx2 = findfirst(x -> x .>= last(wls) + 1, λs_korg)
ax1.set_xlim(λs_korg[idx1], λs_korg[idx2])
ax2.set_xlim(λs_korg[idx1], λs_korg[idx2])
ax2.set_ylim(-450, 25)
ax2.axes.yaxis.set_ticks([-400, -200, 0])
ax4.axis("off")
ax1.axes.xaxis.set_ticklabels([])

fig.tight_layout()
fig.savefig(joinpath(plotdir, "form_temp_flux_vs_intensity_resids.pdf"), bbox_inches="tight")
plt.clf(); plt.close()





# make a plot of the errors
form_temp_errors = form_temps_intensity[:,length(μs)] .- form_temps_flux
fig, ax1 = plt.subplots()
ax1.plot(λs_korg, form_temp_errors, c="k")
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"{\rm Error\ in\ } T_{1/2}\ {\rm [K]}")
ax1.set_xlim(λs_korg[idx1], λs_korg[idx2])
fig.savefig(joinpath(plotdir, "form_temp_error.pdf"))
plt.clf(); plt.close()
