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
mkpath(outdir)
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
Fe_H = 0.0
vsini = 2100.0
ζ_RT = 3400.0
ξ = 850.0

# solar surface differential rotation (Snodgrass & Ulrich 1990):
#   Ω(ϕ) = A + B·sin²ϕ + C·sin⁴ϕ,  A=14.713, B=-2.396, C=-1.787 deg/day
#   normalized rate law  Ω(ϕ)/Ω_eq = 1 - α₂·sin²ϕ - α₄·sin⁴ϕ
istar = 90.0            # inclination (deg); matters once α ≠ 0
α₂ = 0.16285            # = -B/A
α₄ = 0.12145            # = -C/A

# consolidate
star_props = StellarProps(Teff=Teff, logg=logg, Fe_H=Fe_H,
                          vsini=vsini, v_macro=ζ_RT, v_micro=ξ,
                          istar=istar, α₂=α₂, α₄=α₄)

# [doc:params-end]

# write out the temperature, etc.
atm_cpu = FT.AtmosphereCPU(Korg.interpolate_marcs(star_props.Teff, star_props.logg, star_props.A_X))
zs = atm_cpu.zs
nd = atm_cpu.nd
Ts = atm_cpu.Ts
τs_ref = atm_cpu.τs

# boundary-contamination mask: flag pixels whose flux contribution has not decayed by the top
# of the LTE model, so form_temp is biased toward the truncated ceiling. r_thresh is passed
# into the calculation, which warns on it and records it on each result, so FT.boundary_mask
# below reproduces exactly the pixels that were warned about.
r_thresh = FT.BOUNDARY_R_THRESH

# [doc:chunked-start]
# chunked computation parameters
chunk_width = 50.0    # Å per wavelength chunk
wing_padding = 30.0   # Å beyond chunk edges for linelist selection
overlap = 5.0         # Å overlap between chunks for stitching
Δλ = 0.001
Nϕ = 128
buffer = 3.0

# disk-integration method: true → :quadrature (fast ring-by-ring μ-quadrature),
# false → :disk (explicit tiling reference). Nμ/N_az are the quadrature node counts.
use_quadrature = true
Nμ = 32
N_az = 256
integration_method = use_quadrature ? :quadrature : :disk
quad_kwargs = use_quadrature ? (Nμ=Nμ, N_az=N_az) : NamedTuple()

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
        HDF5.attributes(h5)["alpha2"] = star_props.α₂
        HDF5.attributes(h5)["alpha4"] = star_props.α₄
        HDF5.attributes(h5)["integration_method"] = String(integration_method)
        # resolution parameters: Nϕ applies to :disk, Nμ/N_az to :quadrature. All are
        # written unconditionally so integration_method alone says which were in force.
        HDF5.attributes(h5)["N_phi"] = Nϕ
        HDF5.attributes(h5)["N_mu"] = Nμ
        HDF5.attributes(h5)["N_az"] = N_az
        HDF5.attributes(h5)["delta_lambda"] = Δλ
        HDF5.attributes(h5)["buffer"] = buffer
        HDF5.attributes(h5)["wavelength_frame"] = wav_label
        HDF5.attributes(h5)["mask_r_thresh"] = r_thresh

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
            r = FT.ceiling_ratio(result)
            g["ceiling_ratio"] = r
            g["mask"] = UInt8.(FT.boundary_mask(result))
        end

        FT.calc_formation_temp_chunked(star_props, linelist;
                                        chunk_width=chunk_width,
                                        wing_padding=wing_padding,
                                        overlap=overlap,
                                        Δλ=Δλ, buffer=buffer,
                                        method=integration_method, Nϕ=Nϕ, quad_kwargs...,
                                        r_thresh=r_thresh, ne_warn_thresh=Inf,
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

# one chunk's datasets; chunks are streamed rather than all held at once, so peak memory is
# the accumulator plus a single chunk
function read_chunk(h5in, gn)
    g = h5in[gn]
    return (wavs = vec(read(g["wavs"])),
            flux = vec(read(g["flux"])),
            temp = vec(read(g["temp"])),
            cfunc = read(g["cfunc"]),
            lc = vec(read(g["line_centers"])))
end

# Stitch per-chunk results into one spectrum, linearly crossfading each overlap.
#
# The accumulator is preallocated at its final width and written in place. Growing it with
# hcat once per chunk instead costs O(nchunks^2) bytes copied — 223 GiB at the production
# config, against 2.9 GiB of writes here.
function blend_chunks(h5in, chunk_names; overlap, Δλ)
    nchunks = length(chunk_names)

    # the width arithmetic requires every chunk to share one Nλ; calc_formation_temp_chunked
    # rounds its upper bound up so that holds, but check before allocating rather than assume
    Nλ_chunk = length(h5in[first(chunk_names)]["wavs"])
    for gn in chunk_names
        @assert length(h5in[gn]["wavs"]) == Nλ_chunk "chunk $gn has non-uniform Nλ"
    end

    N_ov = max(0, round(Int, overlap / Δλ) + 1)
    n_tail = Nλ_chunk - N_ov
    total = Nλ_chunk + (nchunks - 1) * n_tail
    @assert n_tail > 0 "overlap must be smaller than chunk_width"

    c1 = read_chunk(h5in, first(chunk_names))
    Nrows = size(c1.cfunc, 1)

    all_wavs = Vector{Float64}(undef, total)
    all_flux = Vector{Float64}(undef, total)
    all_temps = Vector{Float64}(undef, total)
    all_cfunc = Matrix{Float64}(undef, Nrows, total)
    all_lc = Float64[]

    all_wavs[1:Nλ_chunk] .= c1.wavs
    all_flux[1:Nλ_chunk] .= c1.flux
    all_temps[1:Nλ_chunk] .= c1.temp
    all_cfunc[:, 1:Nλ_chunk] .= c1.cfunc
    append!(all_lc, c1.lc)
    off = Nλ_chunk

    for i in 2:nchunks
        c = read_chunk(h5in, chunk_names[i])

        # crossfade the overlap: accumulator cols (off-N_ov+1):off against chunk cols 1:N_ov
        @inbounds for k in 1:N_ov
            w_new = Float64(k) / Float64(N_ov + 1)
            w_old = 1.0 - w_new
            d = off - N_ov + k
            all_flux[d] = w_old * all_flux[d] + w_new * c.flux[k]
            all_temps[d] = w_old * all_temps[d] + w_new * c.temp[k]
            @views all_cfunc[:, d] .= w_old .* all_cfunc[:, d] .+ w_new .* c.cfunc[:, k]
        end

        # write the non-overlapping tail in place
        dst = (off + 1):(off + n_tail)
        src = (N_ov + 1):Nλ_chunk
        @views all_wavs[dst] .= c.wavs[src]
        @views all_flux[dst] .= c.flux[src]
        @views all_temps[dst] .= c.temp[src]
        @views all_cfunc[:, dst] .= c.cfunc[:, src]
        # no need to filter to the new tail: line centres are bit-identical across
        # chunks (same Line objects), so sort(unique(·)) and the trim below dedupe
        append!(all_lc, c.lc)
        off += n_tail
    end
    @assert off == total
    return (wavs = all_wavs, flux = all_flux, temps = all_temps, cfunc = all_cfunc,
            lc = sort(unique(all_lc)))
end

h5open(outfile, "r") do h5in
    chunk_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5in))))
    if isempty(chunk_names)
        error("No chunk groups found in $(outfile).")
    end
    nchunks = length(chunk_names)
    b = blend_chunks(h5in, chunk_names; overlap=overlap, Δλ=Δλ)

    # Trim to the actual linelist extent + buffer. The grid is uniform and trim_lo coincides
    # with the first chunk's start, so the kept region is one contiguous range — the trim only
    # drops the overshoot from rounding the last chunk's upper bound. Taking a view of the
    # cfunc columns rather than a slice avoids a second full-size copy.
    wls_Å = [l.wl * 1e8 for l in linelist]
    trim_lo = first(wls_Å) - buffer
    trim_hi = last(wls_Å) + buffer
    lo = findfirst(>=(trim_lo), b.wavs)
    hi = findlast(<=(trim_hi), b.wavs)
    @assert lo !== nothing && hi !== nothing && hi >= lo "trim removed the whole spectrum"
    keep = lo:hi

    all_wavs = b.wavs[keep]
    all_flux = b.flux[keep]
    all_temps = b.temps[keep]
    all_cfunc = view(b.cfunc, :, keep)
    all_lc = b.lc[(b.lc .>= trim_lo) .& (b.lc .<= trim_hi)]

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
        HDF5.attributes(h5out)["alpha2"] = star_props.α₂
        HDF5.attributes(h5out)["alpha4"] = star_props.α₄
        HDF5.attributes(h5out)["integration_method"] = String(integration_method)
        HDF5.attributes(h5out)["N_phi"] = Nϕ
        HDF5.attributes(h5out)["N_mu"] = Nμ
        HDF5.attributes(h5out)["N_az"] = N_az
        HDF5.attributes(h5out)["delta_lambda"] = Δλ
        HDF5.attributes(h5out)["buffer"] = buffer
        HDF5.attributes(h5out)["wavelength_frame"] = wav_label
        HDF5.attributes(h5out)["mask_r_thresh"] = r_thresh

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
        r = FT.ceiling_ratio(all_cfunc)
        g_out["ceiling_ratio"] = r
        g_out["mask"] = UInt8.(FT.boundary_mask(all_cfunc; r_thresh=r_thresh))
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
