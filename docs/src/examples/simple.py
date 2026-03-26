import matplotlib
matplotlib.use("Agg")

from juliacall import Main as jl
import numpy as np
import matplotlib.pyplot as plt

# load FormationTemps
jl.seval("using Korg")
jl.seval("using FormationTemps")
FT = jl.FormationTemps
Korg = jl.Korg

# read the linelist (ships with the package)
# slicing done on Julia side to preserve 1-based indexing
linelist = jl.seval(
    'Korg.read_linelist(joinpath(FormationTemps.datdir, "Sun_VALD.lin"))[16000:16100]'
)

# set stellar parameters (velocities in m/s)
star = FT.StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0,
                       vsini=2100.0, v_macro=3400.0, v_micro=850.0)

# compute formation temperatures (use_gpu=False for portability)
result = FT.calc_formation_temp(star, linelist, use_gpu=False, convolve=True, u1=0.43, u2=0.31)

# extract results into numpy arrays
wavs = np.asarray(result.wavs)
flux = np.asarray(result.flux)
temps = np.asarray(result.form_temps)

# plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
ax1.plot(wavs, flux, c="k")
ax1.set_ylabel("Normalized Flux")
ax2.plot(wavs, temps, c="k")
ax2.set_xlabel("Vacuum Wavelength [Ang]")
ax2.set_ylabel("Formation Temperature [K]")
fig.tight_layout()
fig.savefig("formation_temps.png", dpi=150, bbox_inches="tight")
print("Saved formation_temps.png")
