using Revise
using FormationTemps; FT = FormationTemps
using Korg, LsqFit
using HDF5, Printf, JLD2
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


# astropy stuff
astropy = pyimport("astropy.io.ascii")

# set colormaps
img_cmap = "viridis"
μ_cmap = "autumn"

# alias type 
AA = AbstractArray
CA = CuArray
AF = AbstractFloat

# relate vmac to teff and logg
# from Doyle et al. 2014
vmac_fit(teff, logg) = 3.21 + 2.33e-3 * (teff - 5777) + 2e-6 * (teff - 5777)^2.0 - 2.0 * (logg - 4.44)

# from Bruntt et al. 2010
vmac_fit(teff) = 2.26 + 2.90e-3 * (teff - 5777) + 5.86e-7 * (teff - 5777)^2.0
vmic_fit(teff) = 1.01 + 4.56e-4 * (teff - 5777) + 2.75e-7 * (teff - 5777)^2.0

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

# params for LD fit
μs = range(1.0, 0.2, length=10)
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

cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

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

flux_integration = zeros(length(λs_korg))
flux_cont_integration = zeros(length(λs_korg))
cfunc_flux_integration = zeros(length(λs_korg))

# loop over stars
max_errors = zeros(length(T_effs))
for i in eachindex(T_effs)
    # get the atmosphere
    marcs_atm = FT.get_marcs_atm(T_effs[i], loggs[i], A_X, n_layers=Natm)
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

    # allocate on device
    gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

    # synthesis to get the alphas
    αs = zeros(length(atm_gpu.zs), length(λs_korg))
    αs_cont = zeros(length(atm_gpu.zs), length(λs_korg))
    FT.compute_alpha!(αs, αs_cont, Korg.Wavelengths(λs_korg), linelist, atm_gpu, A_X)

    # get vmicro 
    vmic = vmic_fit(T_effs[i]) * 1000.0
    @show vmic
    σ_v_mic .= vmic

    # get limb darkening
    for k in eachindex(μs)
        # get the intensity contribution function
        cfunc_int_i = FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs[k], μ_v_rot, σ_v_mic)
        cfunc_int_cont_i = FT.calc_intensity_cfunc(αs_cont, atm_gpu, gpu_mem, cmem, μs[k], μ_v_rot, σ_v_mic)

        # intensities
        sum!(view(ints,:,k), cfunc_int_i')
        sum!(view(ints_cont,:,k), cfunc_int_cont_i')
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

    # get the formation temperature for a stationary star
    cfunc_flux_stationary = 2π .* FT.calc_flux_cfunc(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
    flux_stationary = dropdims(sum(cfunc_flux_stationary, dims=1), dims=1)

    cum_cfunc_flux_stationary = cumsum(cfunc_flux_stationary, dims=1)
    cum_cfunc_flux_stationary ./= maximum(cum_cfunc_flux_stationary, dims=1)

    cfunc_flux_cont_stationary = 2π .* FT.calc_flux_cfunc(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)
    flux_cont_stationary = dropdims(sum(cfunc_flux_cont_stationary, dims=1), dims=1)

    form_temp_stationary = zeros(length(λs_korg))
    for i in eachindex(λs_korg)
        xs = view(cum_cfunc_flux_stationary, :, i)
        itp = FT.linear_interp(xs, elav(Ts))
        form_temp_stationary[i] = itp(0.5)
    end
    
    # convolution
    flux_convolution = Array(FT.convolve_hirano_rotmacro(λs_korg, flux_stationary, vsinis[i], vmacs[i], u1, u2))
    flux_cont_convolution = Array(FT.convolve_hirano_rotmacro(λs_korg, flux_cont_stationary, vsinis[i], vmacs[i], u1, u2))
    flux_convolution_norm = flux_convolution ./ flux_cont_convolution

    # get disk stuff 
    ρstar = 1.0
    istar = 90.0
    A = 0.00711 * vsinis[i]
    B = 0.0
    C = 0.0
    v0 = vsinis[i]
    Nϕ = 12
    μs_gpu, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, A, B, C, v0, Nϕ)

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

    for k in eachindex(μs_cpu)
        # set the rotational velocity
        μ_v_rot .= z_rot_cpu[k] .* FT.c_ms

        # get the intensity contribution function
        cfunc_int_i = FT.calc_intensity_cfunc(αs, atm_gpu, gpu_mem, cmem, μs_cpu[k], μ_v_rot, σ_v_mic)
        cfunc_int_cont_i = FT.calc_intensity_cfunc(αs_cont, atm_gpu, gpu_mem, cmem, μs_cpu[k], μ_v_rot, σ_v_mic)

        # convolve the cfunc with RT macroturbulence
        cfunc_int_i_mac = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, CuArray(λs_korg), CuArray(cfunc_int_i), vmacs[i]))
        cfunc_int_cont_i_mac = Array(FT.convolve_gray_rt_macro_gpu(cmem_mac, CuArray(λs_korg), CuArray(cfunc_int_cont_i), vmacs[i]))

        # add to the flux integral
        flux_integration .+= sum(cfunc_int_i_mac, dims=1)' .* dA_cpu[k]
        flux_cont_integration .+= sum(cfunc_int_cont_i_mac, dims=1)' .* dA_cpu[k]
    end
    
    # now get cumulative cfuncs 
    flux_integration_norm = flux_integration ./ flux_cont_integration
    # plt.plot(λs_korg, flux_integration_norm .- flux_convolution_norm)

    # fill max error
    # @show maximum(abs.(flux_integration_norm .- flux_convolution_norm))
    max_errors[i] = maximum(abs.(100 .* (flux_integration_norm .- flux_convolution_norm)))
end

# scatter plot 
sc = plt.scatter(df.Teff, df.logg, s=vsinis./1000, c=max_errors, vmin=1.0, vmax=5.0)
plt.colorbar()
plt.gca().xaxis.set_inverted(true)
plt.gca().yaxis.set_inverted(true)
plt.show()
plt.show()