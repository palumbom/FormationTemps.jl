using HDF5
using Korg
using FormationTemps; FT = FormationTemps
using PyPlot
using Printf
using ProgressMeter


h5path = joinpath("/mnt/ceph/users/mpalumbo/formation_temps", "temp_spectrum_chunks.h5")

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

h5open(h5path, "r") do h5
    group_names = sort(filter(name -> startswith(name, "chunk_"), keys(h5)))
    chunks = length(group_names)
    if chunks == 0
        error("No chunk groups found in HDF5 file.")
    end

    for group_name in group_names
        g = h5[group_name]
        wavs = read(g["wavs"])
        flux = read(g["flux"])
        temp = read(g["temp"])
        cfunc = read(g["cfunc"])

        ax1.plot(wavs, flux, lw=1.0)
        ax2.plot(wavs, temp, lw=1.0)
    end
end
ax1.set_xlabel("Wavelength [Å]")
ax1.set_ylabel("Normalized Flux")
fig1.savefig("flux_spectrum.pdf", bbox_inches="tight")
ax2.set_xlabel("Wavelength [Å]")
ax2.set_ylabel("Formation Temperature [K]")
fig2.savefig("temp_spectrum.pdf", bbox_inches="tight")
plt.clf(); plt.close()
