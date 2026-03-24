using Revise
using FormationTemps; FT = FormationTemps
using Korg
using JLD2
using HDF5, Printf, LsqFit
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
buffer = 4.0
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
vmac = 3400.0

# allocate for intensities 
μs = range(1.0, 0.3, length=100)
ints = zeros(length(λs_korg), length(μs))
ints_cont = zeros(length(λs_korg), length(μs))
for i in eachindex(μs)
    cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs[i], μ_v_rot, σ_v_mic)
    ints_temp = dropdims(Array(FT.get_intensity(cfunc_intensity_struct))', dims=1)
    # ints[:,i] .= FT.convolve_rt_macro(λs_korg, ints_temp, vmac, μs[i])
    ints[:,i] .= ints_temp

    cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs[i], μ_v_rot, σ_v_mic)
    ints_cont_temp = dropdims(Array(FT.get_intensity(cfunc_intensity_cont))', dims=1)
    # ints_cont[:,i] .= FT.convolve_rt_macro(λs_korg, ints_cont_temp, vmac, μs[i])
    ints_cont[:,i] .= ints_cont_temp
end

# normalize 
ints ./= ints_cont[1,1]

# get a slice
idx1 = findfirst(x -> x .>= first(λs_korg) + 2 * step(λs_korg), λs_korg)

# make the fit 
function quad_limb_darkening(μ::T, u1::T, u2::T) where T<:AF
    μ < zero(T) && return 0.0
    return !iszero(μ) * (one(T) - u1*(one(T)-μ) - u2*(one(T)-μ)^2)
end

function quad_limb_darkening(μ::T, p::AA{T,1}) where T<:AF
    return quad_limb_darkening(μ, p[1], p[2])
end

function quad_limb_darkening(μ::AA{T,1}, p::AA{T,1}) where T<:AF
    return quad_limb_darkening.(μ, p[1], p[2])
end

# perform the fit 
xdata = μs
ydata = ints[idx1, :]
p0 = [0.46, 0.15]

fit = curve_fit(quad_limb_darkening, xdata, ydata, p0)
@show coef(fit)

# write em 
coeffs = coef(fit)
u1 = coeffs[1]
u2 = coeffs[2]
@show u1
@show u2

outfile = joinpath(FT.datdir, "ld_coeffs.jld2")
jldsave(outfile; u1, u2)

# plot it 
plt.scatter(μs, ints[idx1, :], c="k")
plt.plot(μs, quad_limb_darkening(μs, coef(fit)))
plt.xlim(reverse(pyconvert(Vector{Float64}, plt.gca().get_xlim())))
plt.show()
