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
# mpl.use("QtAgg")
mpl.style.use(FT.moddir * "fig.mplstyle")
inset = pyimport("mpl_toolkits.axes_grid1.inset_locator")
colormaps = pyimport("colormaps")

# get fancy fonts
plt.rc("text", usetex=true)
plt.rc("text.latex", preamble="\\usepackage{amsmath}
                               \\usepackage{mathrsfs}")

# vacuum or air wavelengths
vacuum_wavs = true
wav_label = vacuum_wavs ? "vacuum" : "air"

# set directory
cephdir = abspath("/mnt/home/mpalumbo/ceph/")
outdir = joinpath(cephdir, "formation_temps")
tmpdir = joinpath(outdir, "tmp")
if !isdir(tmpdir); mkdir(tmpdir); end
outfile = joinpath(outdir, "temp_spectrum_$(wav_label)_chunks_new.h5")
outfile_1d = joinpath(outdir, "temp_spectrum_$(wav_label)_1D_new.h5")

# [doc:linelist-start]
# get the linelist
linelist = Korg.read_linelist("/mnt/home/mpalumbo/ceph/formation_temps/Sun_VALD_BIG.lin")
# wls = [l.wl * 1e8 for l in linelist]
# idx1 = findfirst(wls .>= 3000.0)
# idx2 = findfirst(wls .>= 4000.0)
# linelist = linelist[idx1:idx2]

# convert to air wavelengths
if !vacuum_wavs
    linelist = [Korg.Line(l, wl=Korg.vacuum_to_air(l.wl)) for l in linelist]
end
# [doc:linelist-end]

# parse values values
wls = [l.wl * 1e8 for l in linelist]
log_gf = [l.log_gf for l in linelist]
species = [l.species for l in linelist]
E_lower = [l.E_lower for l in linelist]
gamma_rad = [l.gamma_rad for l in linelist]
gamma_stark = [l.gamma_stark for l in linelist]

# [doc:params-start]
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

# [doc:params-end]

# write out the temperature, etc.
atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(star_props.Teff, star_props.logg, star_props.A_X))
zs = atm_cpu.zs
nd = atm_cpu.nd
Ts = atm_cpu.Ts
τs_ref = atm_cpu.τs

# boundary-contamination mask: flag pixels whose flux forms too high in the atmosphere
# to trust (saturated line cores at/above the top of the LTE photospheric model).
# top_frac = fraction of the contribution function (cfunc = C·Δτ) forming above τ_boundary.
# Pure reduction over cfunc — no recomputation. See scripts/apply_formation_mask.jl.
τ_boundary  = 1e-4    # τ_ref above which the model is untrustworthy
frac_thresh = 0.5     # flag pixel if > this fraction of its flux forms above τ_boundary
τmid = 0.5 .* (τs_ref[1:end-1] .+ τs_ref[2:end])   # mid-layer τ_ref (length Natm-1)
hi_layers = τmid .< τ_boundary
_top_frac(cfunc) = vec(sum(view(cfunc, hi_layers, :), dims=1)) ./ max.(vec(sum(cfunc, dims=1)), eps())

# [doc:chunked-start]
# chunked computation parameters
chunk_width = 50.0    # Å per wavelength chunk
wing_padding = 30.0   # Å beyond chunk edges for linelist selection
overlap = 5.0         # Å overlap between chunks for stitching
Δλ = 0.001
Nϕ = 128
buffer = 3.0

# [doc:chunked-end]

# [doc:callback-start]
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
        HDF5.attributes(h5)["mask_tau_boundary"] = τ_boundary
        HDF5.attributes(h5)["mask_frac_thresh"] = frac_thresh

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
            tf = _top_frac(result.cont_func)
            g["top_frac"] = tf
            g["mask"] = UInt8.(tf .> frac_thresh)
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

# [doc:callback-end]

# repack output file
if isfile(outfile)
    let tmp = outfile * ".tmp"
        HDF5_jll.h5repack() do exe
            run(`$exe $outfile $tmp`)
        end
        mv(tmp, outfile; force=true)
    end
end

# [doc:blend-start]
println(">>> Splicing chunks (blend)...")
h5open(outfile, "r") do h5in
    chunk_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5in))))
    nchunks = length(chunk_names)
    if nchunks == 0
        error("No chunk groups found in $(outfile).")
    end

    # read all chunks into memory for blending
    raw_wavs  = Vector{Vector{Float64}}(undef, nchunks)
    raw_flux  = Vector{Vector{Float64}}(undef, nchunks)
    raw_temp  = Vector{Vector{Float64}}(undef, nchunks)
    raw_cfunc = Vector{Matrix{Float64}}(undef, nchunks)
    raw_lc    = Vector{Vector{Float64}}(undef, nchunks)
    for (i, gn) in enumerate(chunk_names)
        g = h5in[gn]
        raw_wavs[i]  = vec(read(g["wavs"]))
        raw_flux[i]  = vec(read(g["flux"]))
        raw_temp[i]  = vec(read(g["temp"]))
        raw_cfunc[i] = read(g["cfunc"])
        raw_lc[i]    = vec(read(g["line_centers"]))
    end

    # blend: accumulate starting from chunk 1, linearly crossfade in overlap
    N_ov = max(0, round(Int, overlap / Δλ) + 1)

    all_wavs  = copy(raw_wavs[1])
    all_flux  = copy(raw_flux[1])
    all_temps = copy(raw_temp[1])
    all_cfunc = copy(raw_cfunc[1])
    all_lc    = copy(raw_lc[1])

    for i in 2:nchunks
        Nλ_new = length(raw_wavs[i])
        N_ov_actual = min(N_ov, length(all_wavs), Nλ_new)

        # blend the overlap region
        for k in 1:N_ov_actual
            w_new = Float64(k) / Float64(N_ov_actual + 1)
            w_old = 1.0 - w_new
            acc_idx = length(all_wavs) - N_ov_actual + k
            all_flux[acc_idx]  = w_old * all_flux[acc_idx]  + w_new * raw_flux[i][k]
            all_temps[acc_idx] = w_old * all_temps[acc_idx] + w_new * raw_temp[i][k]
            all_cfunc[:, acc_idx] .= w_old .* all_cfunc[:, acc_idx] .+ w_new .* raw_cfunc[i][:, k]
        end

        # append the non-overlapping tail
        if N_ov_actual < Nλ_new
            tail = (N_ov_actual + 1):Nλ_new
            append!(all_wavs, raw_wavs[i][tail])
            append!(all_flux, raw_flux[i][tail])
            append!(all_temps, raw_temp[i][tail])
            all_cfunc = hcat(all_cfunc, raw_cfunc[i][:, tail])
            append!(all_lc, raw_lc[i][filter(j -> raw_lc[i][j] > all_wavs[end - length(tail)], eachindex(raw_lc[i]))])
        end
    end
    all_lc = sort(unique(all_lc))

    # trim to actual linelist extent + buffer
    wls_Å = [l.wl * 1e8 for l in linelist]
    trim_lo = first(wls_Å) - buffer
    trim_hi = last(wls_Å) + buffer
    keep = (all_wavs .>= trim_lo) .& (all_wavs .<= trim_hi)
    all_wavs  = all_wavs[keep]
    all_flux  = all_flux[keep]
    all_temps = all_temps[keep]
    all_cfunc = all_cfunc[:, keep]
    all_lc = all_lc[(all_lc .>= trim_lo) .& (all_lc .<= trim_hi)]

    # write blended result as a single group
    h5open(outfile_1d, "w") do h5out
        HDF5.attributes(h5out)["chunk_width"] = chunk_width
        HDF5.attributes(h5out)["n_lines"] = length(linelist)
        HDF5.attributes(h5out)["n_chunks"] = nchunks
        HDF5.attributes(h5out)["spliced"] = 1
        HDF5.attributes(h5out)["stitch_mode"] = "blend"
        HDF5.attributes(h5out)["Teff"] = star_props.Teff
        HDF5.attributes(h5out)["logg"] = star_props.logg
        HDF5.attributes(h5out)["Fe_H"] = star_props.Fe_H
        HDF5.attributes(h5out)["vsini"] = star_props.vsini
        HDF5.attributes(h5out)["zeta_RT"] = star_props.ζ
        HDF5.attributes(h5out)["xi"] = star_props.ξ
        HDF5.attributes(h5out)["rho_star"] = star_props.ρstar
        HDF5.attributes(h5out)["i_star"] = star_props.istar
        HDF5.attributes(h5out)["wavelength_frame"] = wav_label
        HDF5.attributes(h5out)["mask_tau_boundary"] = τ_boundary
        HDF5.attributes(h5out)["mask_frac_thresh"] = frac_thresh

        g_atm = create_group(h5out, "model_atmosphere")
        g_atm["zs"] = zs
        g_atm["nd"] = nd
        g_atm["Ts"] = Ts
        g_atm["τs_ref"] = τs_ref

        g_out = create_group(h5out, "chunk_0001")
        g_out["line_centers"] = all_lc
        g_out["wavs"] = all_wavs
        g_out["flux"] = all_flux
        g_out["temp"] = all_temps
        g_out["cfunc"] = all_cfunc
        tf = _top_frac(all_cfunc)
        g_out["top_frac"] = tf
        g_out["mask"] = UInt8.(tf .> frac_thresh)
    end
end
# [doc:blend-end]

# repack output file
if isfile(outfile_1d)
    let tmp = outfile_1d * ".tmp"
        HDF5_jll.h5repack() do exe
            run(`$exe $outfile_1d $tmp`)
        end
        mv(tmp, outfile_1d; force=true)
    end
end
