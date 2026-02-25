using HDF5
using Korg
using FormationTemps; FT = FormationTemps
using PyPlot
using Printf
using ProgressMeter

cephdir = abspath("/mnt/home/mpalumbo/ceph/formation_temps")
h5path_chunks = joinpath(cephdir, "temp_spectrum_chunks_ryan.h5")
h5path_splice = joinpath(cephdir, "temp_spectrum_1D.h5")

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

show_spliced = true

h5open(h5path_chunks, "r") do h5
    group_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5))))
    chunks = length(group_names)
    if chunks == 0
        error("No chunk groups found in HDF5 file.")
    end

    # overplot each chunk as-is.
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

    h5open(h5path_splice, "r") do h5
        group_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5))))
        if isempty(group_names)
            error("No chunk groups found in spliced file: $(h5path_splice)")
        end

        # Spliced file is already reconciled: reconstruct by simple concatenation.
        for (idx, group_name) in enumerate(group_names)
            # idx > 10 && break
            g = h5[group_name]
            append!(wavs_spliced, vec(read(g["wavs"])))
            append!(flux_spliced, vec(read(g["flux"])))
            append!(temp_spliced, vec(read(g["temp"])))
        end
    end

    ax1.plot(wavs_spliced, flux_spliced, lw=1.0, c="k", ls=":", label="Spliced")
    ax2.plot(wavs_spliced, temp_spliced, lw=1.0, c="k", ls=":", label="Spliced")
end
ax1.set_xlabel("Wavelength [Å]")
ax1.set_ylabel("Normalized Flux")
ax1.legend()
fig1.savefig("flux_spectrum.pdf", bbox_inches="tight")

ax2.set_xlabel("Wavelength [Å]")
ax2.set_ylabel("Formation Temperature [K]")
ax2.legend()
fig2.savefig("temp_spectrum.pdf", bbox_inches="tight")

plt.show()
plt.clf(); plt.close()
