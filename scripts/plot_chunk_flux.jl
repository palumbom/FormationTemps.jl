using HDF5
using Korg
using FormationTemps; FT = FormationTemps
using PyPlot
using Printf
using ProgressMeter
using Statistics

cephdir = abspath("/mnt/home/mpalumbo/ceph/formation_temps")
h5path = joinpath(cephdir, "temp_spectrum_chunks_ryan.h5")

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

show_spliced = true

h5open(h5path, "r") do h5
    group_names = sort(filter(name -> startswith(name, "chunk_"), collect(keys(h5))))
    chunks = length(group_names)
    if chunks == 0
        error("No chunk groups found in HDF5 file.")
    end

    # overplot each chunk as-is.
    for group_name in group_names
        g = h5[group_name]
        wavs = vec(read(g["wavs"]))
        flux = vec(read(g["flux"]))
        temp = vec(read(g["temp"]))

        ax1.plot(wavs, flux, lw=1.0, alpha=0.8)
        ax2.plot(wavs, temp, lw=1.0, alpha=0.8)
    end

    # splice chunk spectra into one 1D spectrum.
    centers = zeros(Float64, chunks)
    for (i, group_name) in enumerate(group_names)
        g = h5[group_name]
        if haskey(g, "line_centers")
            line_centers = vec(read(g["line_centers"]))
            centers[i] = median(line_centers)
        else
            wavs = vec(read(g["wavs"]))
            centers[i] = 0.5 * (first(wavs) + last(wavs))
        end
    end

    sort_idx = sortperm(centers)
    centers_sorted = centers[sort_idx]
    names_sorted = group_names[sort_idx]

    wavs_spliced = Float64[]
    flux_spliced = Float64[]
    temp_spliced = Float64[]
    for i in eachindex(names_sorted)
        g = h5[names_sorted[i]]
        wavs = vec(read(g["wavs"]))
        flux = vec(read(g["flux"]))
        temp = vec(read(g["temp"]))

        left_bound = i == 1 ? -Inf : 0.5 * (centers_sorted[i - 1] + centers_sorted[i])
        right_bound = i == chunks ? Inf : 0.5 * (centers_sorted[i] + centers_sorted[i + 1])
        keep = (wavs .>= left_bound) .& (wavs .< right_bound)

        append!(wavs_spliced, wavs[keep])
        append!(flux_spliced, flux[keep])
        append!(temp_spliced, temp[keep])
    end

    if show_spliced
        ax1.plot(wavs_spliced, flux_spliced, lw=1.0, c="k", ls=":", label="Spliced")
        ax2.plot(wavs_spliced, temp_spliced, lw=1.0, c="k", ls=":", label="Spliced")
    end
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
