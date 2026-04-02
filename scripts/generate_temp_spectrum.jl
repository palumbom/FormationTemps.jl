Pkg.activate("/mnt/home/mpalumbo/work/FormationTemps")
using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, HDF5_jll, JLD2, Printf
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

# vacuum or air wavelengths
vacuum_wavs = false
wav_label = vacuum_wavs ? "vacuum" : "air"

# set directory
cephdir = abspath("/mnt/home/mpalumbo/ceph/")
outdir = joinpath(cephdir, "formation_temps")
tmpdir = joinpath(outdir, "tmp")
if !isdir(tmpdir); mkdir(tmpdir); end
outfile = joinpath(outdir, "temp_spectrum_$(wav_label)_chunks_debug.h5")
outfile_1d = joinpath(outdir, "temp_spectrum_$(wav_label)_1D_debug.h5")

# get the linelist
linelist = Korg.read_linelist("/mnt/home/mpalumbo/ceph/formation_temps/Sun_VALD_BIG.lin")
wls = [l.wl * 1e8 for l in linelist]
idx1 = findfirst(wls .>= 5000.0)
idx2 = findfirst(wls .>= 5300.0)
linelist = linelist[idx1:idx2]

# convert to air wavelengths
if !vacuum_wavs
    linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
end

# parse values values
wls = [l.wl * 1e8 for l in linelist]
log_gf = [l.log_gf for l in linelist]
species = [l.species for l in linelist]
E_lower = [l.E_lower for l in linelist]
gamma_rad = [l.gamma_rad for l in linelist]
gamma_stark = [l.gamma_stark for l in linelist]

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
nd = atm_cpu.nd
Ts = atm_cpu.Ts
τs_ref = atm_cpu.τs

# chunked computation parameters
chunk_width = 50.0    # Å per wavelength chunk
wing_padding = 30.0   # Å beyond chunk edges for linelist selection
overlap = 5.0         # Å overlap between chunks for stitching
Δλ = 0.001
Nϕ = 32
buffer = 3.0

println(">>> Synthesizing chunks...")
let chunk_idx = Ref(0)
    h5open(outfile, "w") do h5
        # metadata
        HDF5.attributes(h5)["chunk_width"] = chunk_width
        HDF5.attributes(h5)["wing_padding"] = wing_padding
        HDF5.attributes(h5)["overlap"] = overlap
        HDF5.attributes(h5)["n_lines"] = length(linelist)
        HDF5.attributes(h5)["Teff"] = star_props.Teff
        HDF5.attributes(h5)["logg"] = star_props.logg
        HDF5.attributes(h5)["Fe_H"] = star_props.Fe_H
        HDF5.attributes(h5)["vsini"] = star_props.vsini
        HDF5.attributes(h5)["zeta_RT"] = star_props.ζ
        HDF5.attributes(h5)["xi"] = star_props.ξ
        HDF5.attributes(h5)["rho_star"] = star_props.ρstar
        HDF5.attributes(h5)["i_star"] = star_props.istar
        HDF5.attributes(h5)["wavelength_frame"] = wav_label

        # include atmosphere data in the same output file
        g_atm = create_group(h5, "model_atmosphere")
        g_atm["zs"] = zs
        g_atm["nd"] = nd
        g_atm["Ts"] = Ts
        g_atm["τs_ref"] = τs_ref

        # callback writes each chunk to HDF5 as it completes
        function write_chunk(ci, result, ll_chunk)
            chunk_idx[] = ci
            wavs = collect(result.wavs)
            line_centers = [l.wl * 1e8 for l in ll_chunk]
            g = create_group(h5, @sprintf("chunk_%04d", ci))
            g["line_centers"] = line_centers
            g["wavs"] = wavs
            g["flux"] = result.flux
            g["temp"] = result.form_temps
            g["cfunc"] = result.cont_func
        end

        FT.calc_formation_temp_chunked(star_props, linelist;
                                        chunk_width=chunk_width,
                                        wing_padding=wing_padding,
                                        overlap=overlap,
                                        Δλ=Δλ, buffer=buffer,
                                        convolve=false, Nϕ=Nϕ,
                                        ne_warn_thresh=Inf,
                                        callback=write_chunk)
    end
end

# repack output file
if isfile(outfile)
    let tmp = outfile * ".tmp"
        HDF5_jll.h5repack() do exe
            run(`$exe $outfile $tmp`)
        end
        mv(tmp, outfile; force=true)
    end
end

println(">>> Splicing chunks...")
h5open(outfile, "r") do h5in
    chunk_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5in))))
    nchunks = length(chunk_names)
    if nchunks == 0
        error("No chunk groups found in $(outfile).")
    end

    # chunks are already in wavelength order; compute centers for midpoint stitching
    centers = zeros(Float64, nchunks)
    for (i, group_name) in enumerate(chunk_names)
        g = h5in[group_name]
        wavs = vec(read(g["wavs"]))
        centers[i] = 0.5 * (first(wavs) + last(wavs))
    end

    h5open(outfile_1d, "w") do h5out
        HDF5.attributes(h5out)["chunk_width"] = chunk_width
        HDF5.attributes(h5out)["n_lines"] = length(linelist)
        HDF5.attributes(h5out)["n_chunks"] = nchunks
        HDF5.attributes(h5out)["spliced"] = 1
        HDF5.attributes(h5out)["Teff"] = star_props.Teff
        HDF5.attributes(h5out)["logg"] = star_props.logg
        HDF5.attributes(h5out)["Fe_H"] = star_props.Fe_H
        HDF5.attributes(h5out)["vsini"] = star_props.vsini
        HDF5.attributes(h5out)["zeta_RT"] = star_props.ζ
        HDF5.attributes(h5out)["xi"] = star_props.ξ
        HDF5.attributes(h5out)["rho_star"] = star_props.ρstar
        HDF5.attributes(h5out)["i_star"] = star_props.istar
        HDF5.attributes(h5out)["wavelength_frame"] = wav_label

        # include atmosphere data in the same output file
        g_atm = create_group(h5out, "model_atmosphere")
        g_atm["zs"] = zs
        g_atm["nd"] = nd
        g_atm["Ts"] = Ts
        g_atm["τs_ref"] = τs_ref

        for i in eachindex(chunk_names)
            g_in = h5in[chunk_names[i]]
            wavs = vec(read(g_in["wavs"]))
            flux = vec(read(g_in["flux"]))
            temp = vec(read(g_in["temp"]))
            cfunc = read(g_in["cfunc"])
            line_centers = vec(read(g_in["line_centers"]))

            left_bound = i == 1 ? -Inf : 0.5 * (centers[i - 1] + centers[i])
            right_bound = i == nchunks ? Inf : 0.5 * (centers[i] + centers[i + 1])
            keep = (wavs .>= left_bound) .& (wavs .< right_bound)

            # filter line centers to the trimmed wavelength range
            wavs_kept = wavs[keep]
            lc_keep = (line_centers .>= first(wavs_kept)) .& (line_centers .<= last(wavs_kept))

            g_out = create_group(h5out, @sprintf("chunk_%04d", i))
            g_out["line_centers"] = line_centers[lc_keep]
            g_out["wavs"] = wavs_kept
            g_out["flux"] = flux[keep]
            g_out["temp"] = temp[keep]
            g_out["cfunc"] = cfunc[:, keep]
        end
    end
end

# repack output file
if isfile(outfile_1d)
    let tmp = outfile_1d * ".tmp"
        HDF5_jll.h5repack() do exe
            run(`$exe $outfile_1d $tmp`)
        end
        mv(tmp, outfile_1d; force=true)
    end
end
