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

# python interpolation for matplotlib stuff
interp1d = pyimport("scipy.interpolate").interp1d

# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

function get_marcs_atm(Teff::T, logg::T, A_X::AA{T,1}; n_layers::Int=240) where T<:AF
    # get the model atmosphere
    marcs_atm = Korg.interpolate_marcs(Teff, logg, A_X)
    τ_500 = Korg.get_tau_5000s(marcs_atm)
    zs = Korg.get_zs(marcs_atm)
    Ts = Korg.get_temps(marcs_atm)
    ne = Korg.get_electron_number_densities(marcs_atm)
    nd = Korg.get_number_densities(marcs_atm)

    # interpolate in zs 
    itp_τs = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(τ_500))
    itp_Ts = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(Ts))
    itp_ne = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(ne))
    itp_nd = Korg.CubicSplines.CubicSpline(reverse(zs), reverse(nd))

    zs_new = range(last(zs), first(zs), length=n_layers)
    τs_new = reverse(itp_τs.(zs_new))
    Ts_new = reverse(itp_Ts.(zs_new))
    ne_new = reverse(itp_ne.(zs_new))
    nd_new = reverse(itp_nd.(zs_new))
    zs_new = reverse(collect(zs_new))

    ls = Array{Korg.PlanarAtmosphereLayer{Float64, Float64, Float64, Float64, Float64}}(undef, length(zs_new))
    for i in eachindex(zs_new)
        ls[i] = Korg.PlanarAtmosphereLayer(τs_new[i], zs_new[i], Ts_new[i], ne_new[i], nd_new[i])
    end
    return Korg.PlanarAtmosphere(ls)
end

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
λs_korg = range(first(wls) - 5.0, last(wls) + 5.0, step=0.005)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# get some abundances
A_X = Korg.asplund_2020_solar_abundances

# get the atmosphere
marcs_atm = get_marcs_atm(5777.0, 4.44, A_X, n_layers=168 * 3)
τ_500 = Korg.get_tau_5000s(marcs_atm)
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
FT.compute_alpha!(αs, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

# allocate on device
gpu_mem = GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# loop over mus 
μs = range(0.1, 1.0, length=10)
μ_v = CUDA.zeros(Float64, length(zs))
σ_v = CUDA.zeros(Float64, length(zs)) .+ 1200.0
cfuncs = zeros(length(zs)-1, length(λs_korg), length(μs))
intensities = zeros(length(λs_korg), length(μs))

for i in eachindex(μs)
    cfuncs[:,:,i] .= FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs[i], μ_v, σ_v)
    intensities[:,i] .= dropdims(sum(view(cfuncs,:,:,i), dims=1), dims=1)
end
 
# get disk integrated cfunc
cfunc_flux = FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v)
flux_disk_integrated = 2π .* dropdims(sum(cfunc_flux, dims=1), dims=1)

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
# cmap = plt.get_cmap("plasma")
cmap = plt.get_cmap("autumn")
norm = mpl.colors.Normalize(vmin=minimum(μs), vmax=maximum(μs))
colors = cmap(norm(μs))

fig, ax1 = plt.subplots()
for i in eachindex(μs)
    plt.plot(λs_korg, intensities[:,i] ./ 10^exponent_int, c=colors[i,:], lw=1.75)
end 
# plt.plot(λs_korg, flux_disk_integrated, c="k")

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"\mu")

ax1.set_xlim(first(wls) - 0.75, last(wls) + 0.75)
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")

ylims = ax1.get_ylim()

# tick format 
# function sci_notation_formatter(x, pos)
#     if x == 0.0
#         return "\$0\$"
#     end
#     exponent = floor(Int, log10(abs(x)))
#     coeff = x / 10.0^exponent
#     # return "\$" * @sprintf("%.1f", coeff) * "\\times 10^{$exponent}" * "\$"
#     return "\$" * @sprintf("%.1f", coeff) * "\$"
# end
# formatter = mpl.ticker.FuncFormatter(PyObject(sci_notation_formatter))
# ax1.yaxis.set_major_formatter(sci_notation_formatter)

# exponent = log10.(ax1.get_yticks())
# fig.canvas.draw()
# offset = ax1.yaxis.get_major_formatter().get_offset()
# offset = replace(offset, "\\times" => "/")
# ax1.yaxis.get_offset_text().set_visible(false)

# ax1.set_ylabel(L"I_\nu\," * offset * L"{\rm\, (erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1} \ sr ^{-1} )}")
ax1.set_ylabel(L"I_\nu^+\ {\rm [10^{%$exponent_int}\ erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1} \ sr ^{-1} ]}")
fig.savefig(joinpath(plotdir, "intensity_vs_limb_angle.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# now plot the contribution functions 
μ_vals_to_plot = [1, 4, length(μs)]

for i in eachindex(μs)
    !(i in μ_vals_to_plot) && continue

    local fig = plt.figure(constrained_layout=true)
    local gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[1,20], figure=fig)#, width_ratios=[20,1])
    local ax1 = fig.add_subplot(gs[1]); ax1.set_xticklabels([])
    local ax2 = fig.add_subplot(gs[2])#; ax2.set_axis_off()
    # local ax3 = fig.add_subplot(gs[3])
    # local ax4 = fig.add_subplot(gs[4])
    ax3 = ax2
    ax4 = ax1

    local idx1 = findfirst(x -> x .>= first(wls) - 0.75, λs_korg)
    local idx2 = findfirst(x -> x .>= last(wls) + 0.75, λs_korg)

    #= 
    ll = ax1.plot(view(λs_korg, idx1:idx2), view(intensities, idx1:idx2,i), c="k")
    ax1.set_ylim(ylims...)

    # ax1.set_ylabel(L"I_\nu^+\ {\rm [erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1} \ sr ^{-1} ]}")
    val = μs[i]
    ax1.set_ylabel(L"I_\nu^+(\mu\, =\, %$val)") 
    =#

    z_grid = elav(zs)
    τ_grid = elav(τ_500)
    # extent = [λs_korg[idx1], λs_korg[idx2], last(z_grid)/1e7, first(z_grid)/1e7]
    extent = [λs_korg[idx1], λs_korg[idx2], first(τ_500), last(τ_500)]
    
    # vmin = maximum([1.0, first(cb_lims)])
    vmin = first(cb_lims)
    vmax = last(cb_lims)

    xedges = view(λs_korg, idx1:idx2)
    yedges = log10.(elav(τ_500))
    yedges2 = elav(zs ./ 1e7)
    cfunc_view = view(cfuncs,:,idx1:idx2,i)  ./ 10^(exponent)

    # local norm = mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=1.0, linscale=1.0)
    local norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # img = ax3.imshow(cfunc_view, aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    img = ax3.pcolormesh(xedges, yedges2, cfunc_view, 
                         shading="gouraud", cmap="viridis", 
                         edgecolors="none", norm=norm)
    ax3.axvline(xedges[cont_idx - idx1], c="k", ls=":", lw=2.5)

    ax3.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
    # ax3.set_ylabel(L"{\rm \log _{10} (\tau_{5000})}")
    ax3.set_ylabel(L"{\rm Physical\ Depth\ [Mm]}")

    local fwd = interp1d(yedges2, yedges, fill_value="extrapolate")
    local inv = interp1d(yedges, yedges2, fill_value="extrapolate")    

    ax3_right = ax3.secondary_yaxis("right", functions=(fwd, inv))
    # ax3_right.set_ylabel(L"{\rm Physical\ Depth\ [Mm]}")
    ax3_right.set_ylabel(L"{\rm \log _{10} (\tau_{5000})}")
    ax3_right.yaxis.set_ticks([0, -1, -2, -3, -4])

    mu_val = string(μs[i])
    cb = fig.colorbar(img, cax=ax4, orientation="horizontal")
    cb.set_label(L"C_\nu(t_\nu, \mu=%$mu_val)\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1} \ sr ^{-1} ]}", labelpad=10.0)
    # cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    # cb.ax.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0), useOffset=false)
    ax4.grid(false)

    val = replace(string(μs[i]), '.'=>"")
    fig.savefig(joinpath(plotdir, "cfunc_mu_$val.pdf"), bbox_inches="tight")
    plt.clf(); plt.close()
end

# plot slices through the contribution function at different limb angles 

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(elav(zs) ./ 1e7 , cfunc_flux[:,cont_idx] ./ 10^(exponent_cflux) , c="k", label=L"{\rm Flux}")
for i in eachindex(μs)
    # local xs = elav(τ_500)
    local xs = elav(zs) ./ 1e7
    local ys = view(cfuncs,:,cont_idx,i) ./ 10^(exponent)

    mu_val = μs[i]
    ax1.plot(xs, ys, c=colors[i,:], lw=1.75, label=L"\mu = %$mu_val")
end 

fwd = interp1d(elav(zs ./ 1e7), elav(log10.(τ_500)), fill_value="extrapolate")
inv = interp1d(elav(log10.(τ_500)), elav(zs ./ 1e7), fill_value="extrapolate")
ax1_top = ax1.secondary_xaxis("top", functions=(fwd, inv))
ax1_top.xaxis.set_ticks([2, 1, 0, -1, -2, -3, -4, -5])

wav_val = string(round(λs_korg[cont_idx], digits=1))
# ax1.set_xscale("symlog", linthresh=1.0)
ax1.set_xlabel(L"{\rm Physical\ Depth\ [Mm]}")
ax1_top.set_xlabel(L"{\rm \log _{10} (\tau_{5000})}", labelpad=10)
# ax1.set_ylabel(L"C_\nu(%$wav_val\ {\rm \AA})\ {\rm [erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1} \ sr ^{-1} ]}")
ax1.set_ylabel(L"C_{\nu}(t_\nu, \mu)\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1} \ sr ^{-1} ]}")
ax2.set_ylabel(L"\mathcal{C}_{\nu}(t_\nu)\ {\rm [10^{%$exponent}\ erg\ s ^{-1} \ cm ^{-2} \ Hz ^{-1}]}")
ax1.legend()

ax1.set_ylim(cb_lims)
ax2.set_ylim(lims_cflux)

derp1 = diff(ax1.get_xticks())
derp2 = diff(ax1.get_xticks())

ax1.set_ylim(cb_lims[1], cb_lims[2] + derp2[end])
ax2.set_ylim(lims_cflux[1], lims_cflux[2] + derp2[end])

ax1.set_yticks(range(cb_lims[1], cb_lims[2] + derp2[end], length=5))
ax2.set_yticks(range(lims_cflux[1], lims_cflux[2] + derp2[end], length=5))

fig.savefig(joinpath(plotdir, "cont_at_lambda.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# now get cumulative contribution functions
cum_cfuncs_norm = cumsum(cfuncs, dims=1) 
cum_cfuncs_norm ./= maximum(cum_cfuncs_norm, dims=1)
cum_cfunc_flux_norm = cumsum(cfunc_flux, dims=1) 
cum_cfunc_flux_norm ./= maximum(cum_cfunc_flux_norm, dims=1)

# plot the cumulative contribution functions 
fig, ax1 = plt.subplots()
ax1.plot(elav(Ts), cum_cfunc_flux_norm[:,cont_idx], c="k", label=L"{\rm Flux}")
ax1.plot(elav(Ts), cum_cfuncs_norm[:,cont_idx, length(μs)], c=colors[end,:], label=L"\mu = 1.0")

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

fig.savefig(joinpath(plotdir, "cum_cfunc_comparison.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

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

fig, ax1 = plt.subplots()
for i in eachindex(μs)
    mu_val = μs[i]
    ax1.plot(λs_korg, form_temps_intensity[:,i],  c=colors[i,:], label=L"\mu = %$mu_val")
end
ax1.plot(λs_korg, form_temps_flux, c="k", label=L"{\rm Flux}")
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

idx1 = findfirst(x -> x .>= first(wls) - 1.25, λs_korg)
idx2 = findfirst(x -> x .>= last(wls) + 1.25, λs_korg)
ax1.set_xlim(λs_korg[idx1], λs_korg[idx2])

fig.savefig(joinpath(plotdir, "form_temp_flux_vs_intensity.pdf"))
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