using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics, NaNMath
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
                               
# set colormaps
img_cmap = "viridis"
μ_cmap = "autumn"
vmic_cmap = "autumn"

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
λs_korg = range(first(wls) - 5.0, last(wls) + 5.0, step=0.005)
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

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = size(αs, 1)
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# make array of vmics 
mic_min = 0.0
vmics = range(mic_min, 8000.0 + mic_min, step=800.0)

# loop over vmics
μs = 1.0
v_los = CUDA.zeros(Float64, length(zs))
v_mic = CUDA.zeros(Float64, length(zs))
cfuncs = zeros(length(zs)-1, length(λs_korg), length(vmics))
cfuncs_flux = zeros(length(zs)-1, length(λs_korg), length(vmics))
intensities = zeros(length(λs_korg), length(vmics))
fluxes = zeros(length(λs_korg), length(vmics))

for i in eachindex(vmics)
    v_mic .= vmics[i]

    cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs, v_los, v_mic)
    cfuncs[:,:,i] .= Array(FT.get_cum_cfunc(cfunc_intensity_struct))
    intensities[:,i] .= Array(FT.get_intensity(cfunc_intensity_struct))

    local cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, v_mic)
    cfuncs_flux[:,:,i] .= Array(FT.get_cum_cfunc(cfunc_flux_struct))
    fluxes[:,i] = Array(FT.get_flux(cfunc_flux_struct))
end

cum_cfuncs_norm = cfuncs
cum_cfuncs_flux_norm = cfuncs_flux

form_temps_int = zeros(length(λs_korg), length(vmics))
form_temps_flux = zeros(length(λs_korg), length(vmics))

for i in eachindex(λs_korg)
    for j in eachindex(vmics)
        local xs1 = view(cum_cfuncs_norm, :, i, j)
        local xs2 = view(cum_cfuncs_flux_norm, :, i, j)
        local itp1 = FT.linear_interp(xs1, elav(Ts))
        local itp2 = FT.linear_interp(xs2, elav(Ts))
        form_temps_int[i, j] = itp1(0.5)
        form_temps_flux[i, j] = itp2(0.5)
    end
end

# get colormaps
cmap = plt.get_cmap(vmic_cmap)
# norm = mpl.colors.Normalize(vmin=minimum(vmics), vmax=maximum(vmics) + 50.0)
norm = mpl.colors.Normalize(vmin=0.0, vmax=8000.0)
colors = pyconvert(Array, cmap(norm(vmics)))

# do some plotting 
fig, ax1 = plt.subplots()
for i in eachindex(vmics)
    ax1.plot(λs_korg, form_temps_flux[:,i], c=colors[i,:])
end

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(L"v_{\rm mic}\ {\rm[km\ s}^{-1}{\rm ]}")
tickvals = pyconvert(Vector{Float64}, cbar.get_ticks())
cbar.set_ticks(tickvals)
cbar.set_ticklabels(string.(tickvals ./ 1000.0))

ax1.set_xlim(first(wls) - 0.75, last(wls) + 0.75)
ax1.set_xlabel(L"{\rm Air\ Wavelength\ [\AA]}")
ax1.set_ylabel(L"T_{1/2}\ {\rm [K]}")
fig.savefig(joinpath(plotdir, "vmic.pdf"), bbox_inches="tight")
plt.show()
