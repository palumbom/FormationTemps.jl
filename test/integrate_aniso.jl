using FormationTemps; FT = FormationTemps
using Korg
using CUDA
using Statistics

# load the linelist (Fe I 6301/6302)
linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]
linelist = linelist[specs .== "Fe I"]
wls = [l.wl for l in linelist]
idx1 = findfirst(x -> x * 1e8 >= 6301, wls)
idx2 = findfirst(x -> x * 1e8 >= 6302, wls)
linelist = vcat([linelist[idx1], linelist[idx2]])

wls = [l.wl * 1e8 for l in linelist]
buffer = 0.1
λs_korg = range(first(wls) - buffer, last(wls) + buffer, step=0.00005)

A_X = Korg.asplund_2020_solar_abundances
atm_gpu = FT.AtmosphereGPU(Korg.interpolate_marcs(5777.0, 4.44, A_X))
zs = atm_gpu.zs

# velocity grid in m/s centered on wavelength grid midpoint
λ0 = mean(λs_korg)
vs = FT.c_ms .* (collect(λs_korg) .- λ0) ./ λ0

# macroturbulence parameters
ζ_rt = 1400.0

# isotropic RT macro kernel
iso_rt_macro_kernel = FT.gray_iso_rt_macro_kernel(vs, ζ_rt)

# disk-integrated anisotropic RT macro kernel
ρstar = 1.0
istar = 90.0
v0 = 0.0
Nϕ = 256
μs, dA, z_rot, z_cbs = FT.calc_stellar_grid(ρstar, istar, v0, Nϕ)

idx = findall(x -> x > 0.0, Array(μs))
μs_cpu = Array(μs)[idx]
dA_cpu = Array(dA)[idx]

int_kernel = zeros(length(vs))
for i in eachindex(μs_cpu)
    int_kernel .+= FT.rt_macro_kernel(vs, ζ_rt, μs_cpu[i]) .* dA_cpu[i]
end
int_kernel ./= π

# test: disk-integrated aniso kernel converges to the isotropic kernel
@testset "Anisotropic disk integration converges to isotropic kernel" begin
    peak = maximum(abs.(iso_rt_macro_kernel))
    @test peak > 0
    rel_err = maximum(abs.(int_kernel .- iso_rt_macro_kernel)) / peak
    @test rel_err < 0.05
end

if make_plots
    import PythonPlot; plt = PythonPlot
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(vs, iso_rt_macro_kernel, label="Isotropic")
    ax.plot(vs, int_kernel, ls="--", label="Disk-integrated aniso")
    ax.set_xlim(-10_000, 10_000)
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Kernel")
    ax.legend()
    fig.savefig(joinpath(test_plotdir, "integrate_aniso.pdf"), bbox_inches="tight")
    plt.close()
end
