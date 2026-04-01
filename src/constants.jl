# constants in cgs units
const c = 2.99792458e10      # cm s^-1
const σ = 5.6704e-5          # erg cm^-2 s^-1 K^-4
const kB = 1.380649e-16      # erg K^-1
const mH = 1.6737236e-24     # gm
const h = 6.62607015e-27

# constants in other units
const c_cms = 2.99792458e10  # cm s^-1
const c_ms = 2.99792458e8    # m s^-1
const c_kms = 2.99792458e5   # km s^-1
const c_Rsun_day = 37230.42169210867

# for unit conversions
const rsun_au = 0.00465
const sec_year = 3.154e7

# wavelength unit conversions (Korg uses cm internally, public API uses Angstrom)
const CM_TO_ANGSTROM = 1e8
const ANGSTROM_TO_CM = 1e-8

# optical depth floor (sentinel to avoid log(0) in τ integration)
const TAU_FLOOR = 1e-5
