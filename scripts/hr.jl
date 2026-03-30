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
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# astropy stuff
astropy = pyimport("astropy.io.ascii")

# set colormaps
img_cmap = "viridis"
μ_cmap = "autumn"

# alias type
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

# from Brewer (private communication)
const line_coeffs = (6.86575985e-01, 1.58202083e-03, -1.71374049e-07)
function isdwarf(teff, logg)
	return evalpoly(teff, line_coeffs) < logg
end

# make plotdir
plotdir = joinpath(pwd(), "figures")
!isdir(plotdir) && mkdir(plotdir)

# read in brewer data
bfile = joinpath(FT.datdir, "apjsaa6d5at8_mrt.txt")
t = astropy.read(bfile)
tmp = tempname()*".csv"
t.write(tmp, format="csv", overwrite=true)
df = CSV.read(tmp, DataFrame)

# get params
T_effs = Float64.(df.Teff)
loggs = df.logg
vsinis = df.Vsini .* 1000.0
vmacs = df.Vmac .* 1000.0
mohs = df[!, "[M/H]"]

plt.hist(vmacs, bins="auto")
plt.show()

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
buffer = 1.0
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.005)
cont_idx = findfirst(x -> x .>= 6301.3, λs_korg)

# params for LD fit
μs = range(1.0, 0.3, length=10)
ints = zeros(length(λs_korg), length(μs))
ints_cont = zeros(length(λs_korg), length(μs))

# allocate memory for convolutions
Nλ = length(λs_korg)
Natm = 56
Npad = 100
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# velocities
μ_v_rot = CUDA.zeros(Float64, Natm)
σ_v_mic = CUDA.zeros(Float64, Natm)

μ_v_mac = CUDA.zeros(Float64, Natm-1)
σ_v_mac = CUDA.zeros(Float64, Natm-1)

cmem_mac = FT.MacroConvolutionMemory(Nλ, Natm - 1, Npad)

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

flux_integration = CUDA.zeros(Float64, length(λs_korg))
flux_cont_integration = CUDA.zeros(Float64, length(λs_korg))
cfunc_flux_integration = CUDA.zeros(Float64, Natm - 1, length(λs_korg))
cfunc_flux_cont_integration = CUDA.zeros(Float64, Natm - 1, length(λs_korg))
flux_convolution_norm = zeros(length(λs_korg))
flux_integration_norm = CUDA.zeros(Float64, length(λs_korg))
# flux_integration_norm = zeros(length(λs_korg))

αs = zeros(Natm, length(λs_korg))
αs_cont = zeros(Natm, length(λs_korg))

# loop over stars
max_errors = zeros(length(T_effs))
# @showprogress for i in eachindex(T_effs)
for i in eachindex(T_effs)
    # don't do giants (very loosely defined)
    if !isdwarf(T_effs[i], loggs[i])
        max_errors[i] = NaN
        continue
    end

    # get some abundances
    A_X = Korg.format_A_X(mohs[i])

    # get the atmosphere
    atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(T_effs[i], loggs[i], A_X))
    zs = atm_gpu.zs
    Ts = atm_gpu.Ts
    τ5000 = atm_gpu.τs

    # allocate on device
    gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

    # synthesis to get the alphas
    αs .= 0.0
    αs_cont .= 0.0
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X, ne_warn_thresh=Inf)

    # get vmicro
    vmic = FT.vmic_fit(T_effs[i])
    @show vmic
    @show vsinis[i]
    @show vmacs[i]
    σ_v_mic .= vmic

    # get limb darkening
    for k in eachindex(μs)
        μ_v_rot .= 0.0
        cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs[k], μ_v_rot, σ_v_mic)
        ints[:,k] .= Array(FT.get_intensity(cfunc_intensity_struct))

        cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs[k], μ_v_rot, σ_v_mic)
        ints_cont[:,k] .= Array(FT.get_intensity(cfunc_intensity_cont))
    end
    ints ./= ints_cont[1,1]

    # get a slice
    idx_int = findfirst(x -> x .>= first(λs_korg), λs_korg)
    # plt.plot(μs, ints[idx_int, :])
    # plt.show()

    # perform the fit
    xdata = μs
    ydata = ints[idx_int, :]
    p0 = [0.5, 0.25]
    fit = curve_fit(quad_limb_darkening, xdata, ydata, p0)

    # write em
    coeffs = coef(fit)
    u1 = coeffs[1]
    u2 = coeffs[2]

    @show u1
    @show u2
    println()

    # get cfunc for flux
    cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
    flux_stationary = Array(FT.get_flux(cfunc_flux_struct)')

    # get disk integrated continuum
    cfunc_flux_cont_struct = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)
    flux_cont_stationary = Array(FT.get_flux(cfunc_flux_cont_struct)')

    # convolution
    flux_convolution = FT.convolve_hirano_rotmacro(λs_korg, flux_stationary, vsinis[i], vmacs[i], u1, u2)
    flux_cont_convolution = FT.convolve_hirano_rotmacro(λs_korg, flux_cont_stationary, vsinis[i], vmacs[i], u1, u2)
    flux_convolution_norm .= (flux_convolution ./ flux_cont_convolution)'

    # get disk stuff
    ρstar = 1.0
    istar = 90.0
    v0 = vsinis[i]
    Nϕ = 128
    μs_gpu, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

    # flatten, move to cpu
    idx = findall(x -> x .> zero(eltype(μs_gpu)), Array(μs_gpu))
    μs_cpu = Array(μs_gpu)[idx]
    dA_cpu = Array(dA)[idx]
    z_rot_cpu = Array(z_rot)[idx]

    if vsinis[i] == 0.0
        z_rot_cpu .= 0.0
    end

    # rezero
    flux_integration .= 0.0
    flux_cont_integration .= 0.0
    cfunc_flux_integration .= 0.0
    cfunc_flux_cont_integration .= 0.0

    for k in eachindex(μs_cpu)
        # set the rotational velocity
        μ_v_rot .= z_rot_cpu[k] .* FT.c_ms

        # get intensity stuff
        cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[k], μ_v_rot, σ_v_mic)

        tbc = cfunc_intensity_struct.cfunc_dt
        cfunc_int_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, vmacs[i], μs_cpu[k])
        flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[k]

        # now do continuum intensity
        cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs_cpu[k], μ_v_rot, σ_v_mic)

        tbc_cont = cfunc_intensity_cont.cfunc_dt
        cfunc_int_cont_i_mac = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc_cont, vmacs[i], μs_cpu[k])
        flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[k]
    end

    # get the flux
    flux_integration .*= 2π
    flux_cont_integration .*=2π

    # now get cumulative cfuncs
    flux_integration_norm .= flux_integration ./ flux_cont_integration

    # fill max error
    # @show maximum(abs.(flux_integration_norm .- flux_convolution_norm))
    max_errors[i] = maximum(abs.(100 .* (Array(flux_integration_norm) .- flux_convolution_norm)))
end

# save the result
outfile = joinpath(FT.datdir, "hr_error_data.jld2")
jldsave(outfile; df.Teff, df.logg, vsinis, max_errors)

# scatter plot
fig, ax1 = plt.subplots()
sc = ax1.scatter(df.Teff, df.logg, s=vsinis./1000, c=max_errors, vmin=0.0, vmax=4.0)
ax1.xaxis.set_inverted(true)
ax1.yaxis.set_inverted(true)

ax1.set_xlabel(L"T_{\rm eff}\ {\rm [K]}")
ax1.set_ylabel(L"\log g")

cb = fig.colorbar(sc)
cb.set_label(L"{\rm Maximum\ Flux\ Error\ [\%]}")
fig.tight_layout()
fig.savefig("figures/hr_error.pdf", bbox_inches="tight")
# plt.show()
plt.clf()
plt.close()
