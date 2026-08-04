Pkg.activate("/mnt/home/mpalumbo/work/FormationTemps")
using Revise
using FormationTemps; FT = FormationTemps
using Korg
using HDF5, Printf
using Statistics

# plotting
import PythonPlot; plt = PythonPlot
using PythonCall: pyimport, pyconvert
using LaTeXStrings
mpl = plt.matplotlib

# matplotlib backend
# mpl.use("QtAgg")
mpl.style.use(FT.moddir * "fig.mplstyle")
axes_grid1 = pyimport("mpl_toolkits.axes_grid1")

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
h5path_chunks = joinpath(outdir, "temp_spectrum_$(wav_label)_chunks_debug.h5")
h5path_splice = joinpath(outdir, "temp_spectrum_$(wav_label)_1D_debug.h5")

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

show_spliced = true

h5open(h5path_chunks, "r") do h5
    group_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5))))
    chunks = length(group_names)
    if chunks == 0
        error("No chunk groups found in HDF5 file.")
    end

    # overplot each chunk as-is
    for (idx, group_name) in enumerate(group_names)
        # idx > 10 && break
        g = h5[group_name]
        wavs = vec(read(g["wavs"]))
        flux = vec(read(g["flux"]))
        temp = vec(read(g["temp"]))

        ax1.plot(wavs, flux, lw=1.0, alpha=0.8)
        ax2.plot(wavs, temp, lw=1.0, alpha=0.8)
    end
end

if show_spliced
    wavs_spliced = Float64[]
    flux_spliced = Float64[]
    temp_spliced = Float64[]
    cfunc_cols   = Matrix{Float64}[]
    _Ts_atm      = Ref{Vector{Float64}}()
    _τs_ref      = Ref{Vector{Float64}}()

    h5open(h5path_splice, "r") do h5
        group_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5))))
        if isempty(group_names)
            error("No chunk groups found in spliced file: $(h5path_splice)")
        end

        _Ts_atm[] = vec(read(h5["model_atmosphere"]["Ts"]))
        _τs_ref[] = vec(read(h5["model_atmosphere"]["τs_ref"]))

        # spliced file is already reconciled: reconstruct by simple concatenation
        for (idx, group_name) in enumerate(group_names)
            # idx > 10 && break
            g = h5[group_name]
            append!(wavs_spliced, vec(read(g["wavs"])))
            append!(flux_spliced, vec(read(g["flux"])))
            append!(temp_spliced, vec(read(g["temp"])))
            push!(cfunc_cols, read(g["cfunc"]))
        end
    end

    ax1.plot(wavs_spliced, flux_spliced, lw=1.0, c="k", ls=":", label="Spliced")
    ax2.plot(wavs_spliced, temp_spliced, lw=1.0, c="k", ls=":", label="Spliced")

    # --- flux + contribution function heatmap ---
    # The stored cfunc holds per-interval integrals, so on the native MARCS grid its
    # magnitude jumps with the layer spacing (2x at log τ_ref = -3 and +1). Plot the density
    # per dex instead, which is what varies smoothly with depth.
    cfunc_spliced = FT.cfunc_per_dex(hcat(cfunc_cols...), _τs_ref[])
    Ts_atm = _Ts_atm[]
    nrows  = size(cfunc_spliced, 1)
    T_mids = 0.5 .* (Ts_atm[1:nrows] .+ Ts_atm[2:nrows+1])

    # set to (λ_lo, λ_hi) to zoom in, or nothing for full range
    zoom_wav = nothing
    # zoom_wav = (5024.0, 5025.0)

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, sharex=true, figsize=(10, 6),
                                       gridspec_kw=Dict("height_ratios" => [1, 2]))
    fig3.subplots_adjust(hspace=0.05)

    ax3a.plot(wavs_spliced, flux_spliced, lw=0.8, c="k")
    ax3a.set_ylabel("Normalized Flux")
    ax3a.tick_params(labelbottom=false)

    # log-scaled (no column normalization)
    colors = pyimport("matplotlib.colors")
    vmin = maximum(cfunc_spliced) * 1e-4  # clip 4 decades below peak
    im = ax3b.pcolormesh(wavs_spliced, T_mids, cfunc_spliced,
                          cmap="inferno", shading="auto", rasterized=true,
                          norm=colors.LogNorm(vmin=vmin, vmax=maximum(cfunc_spliced)))
    ax3b.invert_yaxis()
    ax3b.set_xlabel("Wavelength [\\AA]")
    ax3b.set_ylabel("Temperature [K]")

    # append a dummy axis to ax3a so both rows shrink by the same amount
    divider_a = axes_grid1.make_axes_locatable(ax3a)
    cax_dummy = divider_a.append_axes("right", size="2%", pad=0.05)
    cax_dummy.set_visible(false)

    divider_b = axes_grid1.make_axes_locatable(ax3b)
    cax = divider_b.append_axes("right", size="2%", pad=0.05)
    cbar = fig3.colorbar(im, cax=cax,
                         label="\$dF/d\\log_{10}\\tau_{\\rm ref}\$ (log scale)")
    cbar.ax.tick_params(length=0)
    cbar.ax.grid(false)

    if !isnothing(zoom_wav)
        ax3b.set_xlim(zoom_wav...)
    end

    fig3.savefig("cfunc_heatmap.pdf", bbox_inches="tight")
end

ax1.set_xlabel("Wavelength [\\AA]")
ax1.set_ylabel("Normalized Flux")
ax1.legend()
fig1.savefig("flux_spectrum.pdf", bbox_inches="tight")

ax2.set_xlabel("Wavelength [\\AA]")
ax2.set_ylabel("Formation Temperature [K]")
ax2.legend()
fig2.savefig("temp_spectrum.pdf", bbox_inches="tight")
plt.show()
plt.clf(); plt.close()
plt.close("all")
plt.close()
