Pkg.activate("/mnt/home/mpalumbo/work/FormationTemps")
using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using PyPlot, PyCall; mpl = plt.matplotlib

# matplotlib backend
mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
# mpl.style.use("tableau-colorblind10")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# set directory
cephdir = abspath("/mnt/home/mpalumbo/ceph/")
outdir = joinpath(cephdir, "formation_temps")
tmpdir = joinpath(outdir, "tmp")
if !isdir(tmpdir); mkdir(tmpdir); end
outfile = joinpath(outdir, "temp_spectrum_chunks_ryan.h5")

# get the linelist
linelist = Korg.read_linelist("/mnt/home/mpalumbo/ceph/formation_temps/Sun_VALD_BIG.lin")
# linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
specs = [string(l.species) for l in linelist]

# re-get values
wls = [l.wl * 1e8 for l in linelist]
log_gf =  [l.log_gf for l in linelist]
species =  [l.species for l in linelist]
E_lower =  [l.E_lower for l in linelist]
gamma_rad =  [l.gamma_rad for l in linelist]
gamma_stark =  [l.gamma_stark for l in linelist]

# set parameters
Teff = 5777.0
logg = 4.44
A_X = Korg.asplund_2020_solar_abundances
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0

# consolidate 
star_props = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H, 
                          vsini=vsini, v_macro=ζ_RT, v_micro=ξ)

# write out the temperature, etc. 
atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(star_props.Teff, star_props.logg, star_props.A_X))
zs = atm_cpu.zs
Ts = atm_cpu.Ts
τs_ref = atm_cpu.τs

# write out
atm_file = joinpath(outdir, "solar_model_atmosphere.h5")
h5open(atm_file, "w") do h5
    h5["zs"] = zs
    h5["Ts"] = Ts
    h5["τs_ref"] = τs_ref
end

# set linelist chunk size
chunksize = 200

h5open(outfile, "w") do h5
    HDF5.attributes(h5)["chunksize"] = chunksize
    HDF5.attributes(h5)["n_lines"] = length(linelist)

    # loop over chunks
    for i in 1:chunksize:length(linelist)
        # get view of linelist
        ll = view(linelist, i:min(i + chunksize - 1, length(linelist)))

        # high-level formation temperature calculation
        Δλ=0.001
        # Δλ=0.0001
        form_temp_result = FT.calc_formation_temp(star_props, ll; Δλ=Δλ, 
                                                  convolve=false, Nϕ=16, 
                                                  ne_warn_thresh=Inf)

        # parse out results
        wavs = collect(form_temp_result.wavs)
        flux = form_temp_result.flux
        temp = form_temp_result.form_temps
        cfunc = form_temp_result.cont_func

        # write to a file
        chunk_idx = (i - 1) ÷ chunksize + 1
        g = create_group(h5, @sprintf("chunk_%04d", chunk_idx))
        g["wavs"] = wavs
        g["flux"] = flux
        g["temp"] = temp
        g["cfunc"] = cfunc
        HDF5.attributes(g)["start_index"] = i
        HDF5.attributes(g)["end_index"] = i + length(ll) - 1
    end
end