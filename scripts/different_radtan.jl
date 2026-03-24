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
Npad = 2400
cmem = FT.ConvolutionMemory(Nλ, Natm, Npad)

# allocate on device
gpu_mem = FT.GPUMemory(λs_korg, atm_gpu)

# velocities
μ_v_rot = CUDA.zeros(Float64, length(zs))
σ_v_mic = CUDA.zeros(Float64, length(zs)) .+ 800.0

μ_v_mac = CUDA.zeros(Float64, length(zs)-1)
σ_v_mac = CUDA.zeros(Float64, length(zs)-1)

cmem_mac = FT.ConvolutionMemory(Nλ, Natm - 1, Npad)

# get the formation temperature for a stationary star
cfunc_flux_struct = FT.calc_flux_quantities(αs, atm_gpu, gpu_mem, cmem, σ_v_mic)
flux_stationary = Array(FT.get_flux(cfunc_flux_struct)')
cfunc_flux_stationary = cfunc_flux_struct.cfunc_dt
cum_cfunc_flux_stationary = Array(FT.get_cum_cfunc(cfunc_flux_struct))

cfunc_flux_cont_struct = FT.calc_flux_quantities(αs_cont, atm_gpu, gpu_mem, cmem, σ_v_mic)
cfunc_flux_cont_stationary = cfunc_flux_cont_struct.cfunc_dt
flux_cont_stationary = Array(FT.get_flux(cfunc_flux_cont_struct)')

form_temp_stationary = zeros(length(λs_korg))
for i in eachindex(λs_korg)
    xs = view(cum_cfunc_flux_stationary, :, i)
    itp = FT.linear_interp(xs, elav(Ts))
    form_temp_stationary[i] = itp(0.5)
end

# set parameters
ζ_rt = 3400.0
ζ_rt_quad_sum = sqrt(2.0 * ζ_rt^2.0)

ζ_r = range(200.0, 4800.0, step=50.0)
ζ_t = @. sqrt(ζ_rt_quad_sum^2.0 - ζ_r^2.0)
ζ_t_string = round.(ζ_t, digits=0)
vsini = 2100.0
u1 = 0.4
u2 = 0.26

# convolution model 
cfunc_flux_convolution = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_stationary, vsini, ζ_rt, u1, u2))
flux_convolution = 2π .* dropdims(sum(cfunc_flux_convolution, dims=1), dims=1)

cfunc_flux_convolution_cont = Array(FT.convolve_hirano_rotmacro_gpu(cmem_mac, λs_korg, cfunc_flux_cont_stationary, vsini, ζ_rt, u1, u2))
flux_convolution_cont = 2π .* dropdims(sum(cfunc_flux_convolution_cont, dims=1), dims=1)

flux_convolution_norm = Array(flux_convolution ./ flux_convolution_cont)

# allocate for "error"
err_same = zeros(length(λs_korg), length(ζ_r))
err_diff = zeros(length(λs_korg), length(ζ_r))

# disk integration stuff
ρstar = 1.0
istar = 90.0
v0 = vsini
Nϕ = 16
μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

# flatten, move to cpu
idx = findall(x -> x .> zero(eltype(μs)), Array(μs))
μs_cpu = Array(μs)[idx]
dA_cpu = Array(dA)[idx]
z_rot_cpu = Array(z_rot)[idx]

# allocate memory
flux_integration_same = CUDA.zeros(Float64, length(λs_korg))
flux_integration_diff = CUDA.zeros(Float64, length(λs_korg))
flux_cont_integration_same = CUDA.zeros(Float64, length(λs_korg))
flux_cont_integration_diff = CUDA.zeros(Float64, length(λs_korg))

cfunc_flux_integration_same = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))
cfunc_flux_integration_diff = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))
cfunc_flux_cont_integration_same = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))
cfunc_flux_cont_integration_diff = CUDA.zeros(Float64, length(zs)-1, length(λs_korg))

if !isdir("figures/RT_frames/")
    mkdir("figures/RT_frames/")
end

@showprogress for j in eachindex(ζ_r)
    # re-zero 
    flux_integration_diff .= 0.0
    flux_cont_integration_diff .= 0.0
    cfunc_flux_integration_diff .= 0.0
    cfunc_flux_cont_integration_diff .= 0.0

    # loop over disk
    for i in eachindex(μs_cpu)
        # set the rotational velocity
        μ_v_rot .= z_rot_cpu[i] .* FT.c_ms

        # get intensity stuff
        cfunc_intensity_struct = FT.calc_intensity_quantities(αs, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)
        cfunc_intensity_cont = FT.calc_intensity_quantities(αs_cont, atm_gpu, gpu_mem, cmem, μs_cpu[i], μ_v_rot, σ_v_mic)

        # convolution with same RT
        if j == 1
            tbc = cfunc_intensity_struct.cfunc_dt
            cfunc_int_i_mac_same = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_rt, μs_cpu[i])
            flux_integration_same .+= sum(cfunc_int_i_mac_same, dims=1)' .* dA_cpu[i]
            cfunc_flux_integration_same .+= cfunc_int_i_mac_same .* dA_cpu[i]


            # continuum convolution with same RT
            tbc_cont = cfunc_intensity_cont.cfunc_dt
            cfunc_int_cont_i_mac_same = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc_cont, ζ_rt, μs_cpu[i])
            flux_cont_integration_same .+= sum(cfunc_int_cont_i_mac_same, dims=1)' .* dA_cpu[i]
            cfunc_flux_cont_integration_same .+= cfunc_int_cont_i_mac_same .* dA_cpu[i]
        end

        # convolution with different R and T
        tbc = cfunc_intensity_struct.cfunc_dt
        cfunc_int_i_mac_diff = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc, ζ_r[j], ζ_t[j], μs_cpu[i])
        flux_integration_diff .+= sum(cfunc_int_i_mac_diff, dims=1)' .* dA_cpu[i]
        cfunc_flux_integration_diff .+= cfunc_int_i_mac_diff .* dA_cpu[i]

        # continuum convolution with different R and T
        tbc_cont = cfunc_intensity_cont.cfunc_dt
        cfunc_int_cont_i_mac_diff = FT.convolve_rt_macro_gpu(cmem_mac, λs_korg, tbc_cont, ζ_r[j], ζ_t[j], μs_cpu[i])
        flux_cont_integration_diff .+= sum(cfunc_int_cont_i_mac_diff, dims=1)' .* dA_cpu[i]
        cfunc_flux_cont_integration_diff .+= cfunc_int_cont_i_mac_diff .* dA_cpu[i]
    end

    # 2pi
    flux_integration_same .*= 2π
    flux_integration_diff .*= 2π
    flux_cont_integration_same .*= 2π
    flux_cont_integration_diff .*= 2π

    # normalize
    flux_integration_same_norm = Array(flux_integration_same ./ flux_cont_integration_same)
    flux_integration_diff_norm = Array(flux_integration_diff ./ flux_cont_integration_diff)

    # get errors
    err_same[:,j] .= flux_integration_same_norm .- flux_convolution_norm
    err_diff[:,j] .= flux_integration_diff_norm .- flux_convolution_norm

    rl = ζ_r[j]
    tl = ζ_t_string[j]

    # plot it 
    plt.plot(λs_korg, flux_convolution_norm, label="Convolution, R=T=$ζ_rt m/s")
    plt.plot(λs_korg, flux_integration_same_norm, label="Integration, R=T=$ζ_rt m/s")
    plt.plot(λs_korg, flux_integration_diff_norm, label="Integration, R=$rl m/s, T=$tl m/s")
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Normalized Flux")
    plt.xlim(6301.25, 6301.75)
    plt.ylim(0.35, 0.55)
    plt.tight_layout()
    plt.savefig("figures/RT_frames/frame_$j.png", bbox_inches="tight")
    plt.clf(); plt.close()
end

plt.axhline(maximum(abs.(err_same)), c="k", ls=":", label="Integration (same RT) - Convolution (same RT)")
plt.plot(ζ_r, maximum(abs.(err_diff), dims=1)', label="Integration (diff RT) - Convolution (same RT)")
plt.xlabel("zeta_R")
plt.ylabel("Flux 'Error'")
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0)
plt.tight_layout()
plt.show()
