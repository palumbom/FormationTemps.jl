Pkg.activate("/mnt/home/mpalumbo/work/FormationTemps")
using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, JLD2, Printf
using CUDA, BenchmarkTools
using CSV, DataFrames, Statistics
using ProgressMeter

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pyconvert
using LaTeXStrings
mpl = plt.matplotlib

# matplotlib backend
# mpl.use("Qt5Agg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# set directory
cephdir = abspath("/mnt/home/mpalumbo/ceph/")
outdir = joinpath(cephdir, "formation_temps")
tmpdir = joinpath(outdir, "tmp")
if !isdir(tmpdir); mkdir(tmpdir); end
outfile = joinpath(outdir, "temp_spectrum_chunks_hires.h5")
outfile_1d = joinpath(outdir, "temp_spectrum_hires_1D.h5")

# get the linelist
linelist = Korg.read_linelist("/mnt/home/mpalumbo/ceph/formation_temps/Sun_VALD_BIG.lin")
# wls = [l.wl * 1e8 for l in linelist]
# idx1 = findfirst(wls .>= 5000.0)
# idx2 = findfirst(wls .>= 5050.0)
# linelist = linelist[idx1:idx2]

# convert to air wavelengths (if necessary)
# linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]

# parse values values
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

# set linelist chunk size
chunksize = 400
overlap_lines = 100
@assert 0 <= overlap_lines < chunksize "overlap_lines must satisfy 0 <= overlap_lines < chunksize."
chunk_step = chunksize - overlap_lines

println(">>> Synthesizing chunks...")
h5open(outfile, "w") do h5
    # metadata
    HDF5.attributes(h5)["chunksize"] = chunksize
    HDF5.attributes(h5)["overlap_lines"] = overlap_lines
    HDF5.attributes(h5)["chunk_step"] = chunk_step
    HDF5.attributes(h5)["n_lines"] = length(linelist)

    # include atmosphere data in the same output file
    g_atm = create_group(h5, "model_atmosphere")
    g_atm["zs"] = zs
    g_atm["Ts"] = Ts
    g_atm["τs_ref"] = τs_ref

    # loop over chunks
    for (chunk_idx, i) in enumerate(1:chunk_step:length(linelist))
        # get view of linelist
        chunk_end = min(i + chunksize - 1, length(linelist))
        ll = view(linelist, i:chunk_end)
        line_centers = [l.wl * 1e8 for l in ll]

        # high-level formation temperature calculation
        Δλ = 0.0001
        form_temp_result = FT.calc_formation_temp(star_props, ll; Δλ=Δλ,
                                                  convolve=false, Nϕ=128,
                                                  buffer=3.0,
                                                  ne_warn_thresh=Inf)

        # parse out results
        wavs = collect(form_temp_result.wavs)
        flux = form_temp_result.flux
        temp = form_temp_result.form_temps
        cfunc = form_temp_result.cont_func

        # write to a file
        g = create_group(h5, @sprintf("chunk_%04d", chunk_idx))
        g["line_centers"] = line_centers
        g["wavs"] = wavs
        g["flux"] = flux
        g["temp"] = temp
        g["cfunc"] = cfunc
        HDF5.attributes(g)["start_index"] = i
        HDF5.attributes(g)["end_index"] = chunk_end
    end
end

println(">>> Splicing chunks...")
h5open(outfile, "r") do h5in
    chunk_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5in))))
    nchunks = length(chunk_names)
    if nchunks == 0
        error("No chunk groups found in $(outfile).")
    end

    # determine chunk centers to split overlap regions consistently
    centers = zeros(Float64, nchunks)
    for (i, group_name) in enumerate(chunk_names)
        g = h5in[group_name]
        if haskey(g, "line_centers")
            line_centers = vec(read(g["line_centers"]))
            centers[i] = median(line_centers)
        else
            wavs = vec(read(g["wavs"]))
            centers[i] = 0.5 * (first(wavs) + last(wavs))
        end
    end

    # enforce increasing wavelength order in the output chunks
    sort_idx = sortperm(centers)
    centers = centers[sort_idx]
    chunk_names = chunk_names[sort_idx]

    h5open(outfile_1d, "w") do h5out
        HDF5.attributes(h5out)["chunksize"] = chunksize
        HDF5.attributes(h5out)["n_lines"] = length(linelist)
        HDF5.attributes(h5out)["n_chunks"] = nchunks
        HDF5.attributes(h5out)["spliced"] = 1

        # retain model atmosphere info in the new file
        g_atm = create_group(h5out, "model_atmosphere")
        g_atm["zs"] = zs
        g_atm["Ts"] = Ts
        g_atm["τs_ref"] = τs_ref

        for i in eachindex(chunk_names)
            g_in = h5in[chunk_names[i]]
            wavs = vec(read(g_in["wavs"]))
            flux = vec(read(g_in["flux"]))
            temp = vec(read(g_in["temp"]))
            cfunc = read(g_in["cfunc"])
            line_centers = haskey(g_in, "line_centers") ? vec(read(g_in["line_centers"])) : [0.5 * (first(wavs) + last(wavs))]

            left_bound = i == 1 ? -Inf : 0.5 * (centers[i - 1] + centers[i])
            right_bound = i == nchunks ? Inf : 0.5 * (centers[i] + centers[i + 1])
            keep = (wavs .>= left_bound) .& (wavs .< right_bound)

            g_out = create_group(h5out, @sprintf("chunk_%04d", i))
            g_out["line_centers"] = line_centers
            g_out["wavs"] = wavs[keep]
            g_out["flux"] = flux[keep]
            g_out["temp"] = temp[keep]
            g_out["cfunc"] = cfunc[:, keep]
        end
    end
end
