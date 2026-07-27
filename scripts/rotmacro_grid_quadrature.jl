# Same comparison as rotmacro_grid.jl -- disk integration vs. the Hirano analytic
# rotation+macroturbulence convolution, over a (vsini, ζ_RT) grid -- but the integration arm is
# the ring-by-ring μ-quadrature of src/quadrature.jl instead of the explicit tile loop.
#
# Radiative transfer here is wavelength-local, so a surface element enters only through μ:
# the zero-Doppler intensity contribution function is solved once per Gauss-Legendre μ node and
# rotation is applied afterwards as a per-ring azimuthal Doppler convolution. Those per-node
# solves depend on neither vsini nor ζ, so they are computed once for the whole (vsini, ζ) grid
# and cached; each grid point then costs only FFT convolutions.

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

# set rotational and macroturbulence grids
vsinis = range(0.00, 16_000.0, step=2_000.0)
vmacs = range(0.0, 10_000.0, step=2_000.0)
vsinis_kms = vsinis ./ 1000
vmacs_kms = vmacs ./ 1000

# microturbulence (m/s)
v_mic_val = 1200.0

# quadrature resolution: Nμ Gauss-Legendre nodes in μ, N_az floors the azimuthal arc count of
# the ring Doppler kernel. α₂ = α₄ = 0 selects the analytic solid-body kernel branch, for which
# N_az is unused.
Nμ = 32
N_az = 256
α₂ = 0.0
α₄ = 0.0

# allocate memory for convolutions. Padding is derived from the widest kernel support on the
# grid (vsini + 3ζ + 3ξ); an under-padded linear convolution wraps silently.
Nλ = length(λs_korg)
Natm = size(αs, 1)
λ0_pad = λs_korg[Nλ ÷ 2 + 1]
Npad = FT.conv_npad_for_velocity(λ0_pad, step(λs_korg),
                                 FT.conv_kernel_vmax(maximum(vsinis), maximum(vmacs), v_mic_val))
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)
cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)

# allocate on device (anchored τ integrator when tau_ref is available). Separate working memory
# for the total and continuum μ-node solves, so a cached node cannot be clobbered.
α_ref = αs_cont[:, end]
_make_gpu_mem = isempty(atm_gpu.τs) ? (() -> FT.GPUMemory(λs_korg, atm_gpu)) :
                                      (() -> FT.GPUMemory(λs_korg, atm_gpu, α_ref))
gpu_mem = _make_gpu_mem()
gpu_mem_cont = _make_gpu_mem()

# velocities
v_mic = CUDA.zeros(Float64, length(zs)) .+ v_mic_val

# get the formation temperature for a stationary star
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, v_mic)
flux_stationary = Array(FT.get_flux(cfunc_flux_struct)')
cfunc_flux_stationary = cfunc_flux_struct.cfunc_dt
cum_cfunc_flux_stationary = Array(FT.get_cum_cfunc(cfunc_flux_struct))

cfunc_flux_cont_struct = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, v_mic)
cfunc_flux_cont_stationary = cfunc_flux_cont_struct.cfunc_dt
flux_cont_stationary = Array(FT.get_flux(cfunc_flux_cont_struct)')

# formation temperature at 50% cumulative flux contribution (node-anchored CDF)
form_temp_stationary = FT.form_temps_from_cfunc(Array(cfunc_flux_stationary), Ts)

# Microturbulence depends only on v_mic and the (zero) rotation velocity, both fixed for the
# whole run, so broaden the absorption coefficients once here. convolve_wavelength_axis_gpu
# returns a view into cmem.conv_gpu, which the next per-row kernel build uses as scratch, so
# copy out.
copyto!(gpu_mem.αs, αs)
αs_b = copy(FT.convolve_wavelength_axis_gpu(cmem, gpu_mem.λs, gpu_mem.αs,
                                            gpu_mem.v_los_zeros, v_mic))
copyto!(gpu_mem.αs, αs_cont)
αs_cont_b = copy(FT.convolve_wavelength_axis_gpu(cmem, gpu_mem.λs, gpu_mem.αs,
                                                 gpu_mem.v_los_zeros, v_mic))

# Gauss-Legendre μ nodes on [0,1]. generate_mu_grid returns weights halved (summing to 1) and
# with no μ factor, so wq = w_k * μ_k is the projected-area weight for ∫ I μ dμ.
μ_grid, μ_weights = Korg.RadiativeTransfer.generate_mu_grid(Nμ)
wqs = [Float64(μ_weights[k]) * Float64(μ_grid[k]) for k in eachindex(μ_grid)]

# Zero-Doppler depth-resolved intensity contribution functions per μ node. These depend on
# neither vsini nor ζ, so they are solved once and reused across the whole grid.
G_nodes = Vector{CuArray{Float64,2}}(undef, length(μ_grid))
G_nodes_cont = Vector{CuArray{Float64,2}}(undef, length(μ_grid))
let
    for k in eachindex(μ_grid)
        μ_k = Float64(μ_grid[k])
        G_nodes[k] = copy(FT.calc_intensity_quantities_broadened!(αs_b, atm_gpu,
                                                                 gpu_mem, μ_k).cfunc_dt)
        G_nodes_cont[k] = copy(FT.calc_intensity_quantities_broadened!(αs_cont_b, atm_gpu,
                                                                      gpu_mem_cont, μ_k).cfunc_dt)
    end
end

# host wavelength grid for the ring-kernel build, and its zero-velocity index
λs_host = collect(λs_korg)
i0 = Nλ ÷ 2 + 1

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

    # disk geometry enters the quadrature only through the ring Doppler kernel, which needs the
    # inclination of the spin axis to the sky plane
    istar = 90.0
    iₛ = deg2rad(90.0 - istar)

    # loop over macro
    for j in eachindex(vmacs)
        # re-zero output
        flux_integration .= 0.0
        flux_cont_integration .= 0.0
        cfunc_flux_integration .= 0.0
        cfunc_flux_cont_integration .= 0.0

        # do the disk integration, ring by ring
        for i in eachindex(μ_grid)
            μ_i = Float64(μ_grid[i])
            wq = wqs[i]

            # μ-dependent macroturbulence; copy out of the shared cmem_mac.out_gpu buffer
            Gm = copy(FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, G_nodes[i], vmacs[j], μ_i))
            Gmc = copy(FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, G_nodes_cont[i], vmacs[j], μ_i))

            # azimuthal Doppler convolution (identity when vsini == 0)
            if iszero(vsinis[k])
                cfunc_flux_integration .+= wq .* Gm
                cfunc_flux_cont_integration .+= wq .* Gmc
            else
                K = FT._ring_doppler_kernel(μ_i, vsinis[k], iₛ, α₂, α₄, λs_host, N_az)
                kft = FT._ring_kernel_ft_gpu(cmem_mac, K, i0)
                # one kernel FT serves both signals; the cached convolution returns
                # cmem_mac.out_gpu, so accumulate immediately before it is reused
                cfunc_flux_integration .+= wq .* FT.convolve_rt_macro_gpu_cached(cmem_mac, Gm, kft)
                cfunc_flux_cont_integration .+= wq .* FT.convolve_rt_macro_gpu_cached(cmem_mac, Gmc, kft)
            end
        end

        # flux is 2π ∫ I μ dμ; the projected-area weights are already folded into the cfuncs
        flux_integration .= 2π .* vec(sum(cfunc_flux_integration, dims=1))
        flux_cont_integration .= 2π .* vec(sum(cfunc_flux_cont_integration, dims=1))

        # get the convolution
        cfunc_flux_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_stationary, vsinis[k], vmacs[j], u1, u2))
        flux_convolution = dropdims(sum(cfunc_flux_convolution, dims=1), dims=1)
        cfunc_flux_cont_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_cont_stationary, vsinis[k], vmacs[j], u1, u2))
        flux_cont_convolution = dropdims(sum(cfunc_flux_cont_convolution, dims=1), dims=1)

        # formation temperatures at 50% cumulative flux contribution (node-anchored CDF)
        form_temp_integration = FT.form_temps_from_cfunc(Array(cfunc_flux_integration), Ts)
        form_temp_convolution = FT.form_temps_from_cfunc(Array(cfunc_flux_convolution), Ts)

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
fig1.savefig("figures/big_plot_flux_quad.pdf", bbox_inches="tight")

# axins = inset.inset_axes(ax2, width="5%",height="100%", loc="lower left",
#                          bbox_to_anchor=(1.05, 0., 1, 1),
#                          bbox_transform=ax2.transAxes,borderpad=0)
# fig2.colorbar(sm, cax=axins)
fig2.tight_layout()
fig2.savefig("figures/big_plot_temperature_quad.pdf", bbox_inches="tight")

fig3.tight_layout()
fig3.savefig("figures/other_big_plot_flux_quad.pdf", bbox_inches="tight")

fig4.tight_layout()
fig4.savefig("figures/other_big_plot_temperature_quad.pdf", bbox_inches="tight")

plt.clf(); plt.close("all");
