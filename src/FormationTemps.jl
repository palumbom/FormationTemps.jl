module FormationTemps

# general imports
using CUDA
using Korg
using FFTW
using Statistics
using AbstractFFTs
using ProgressMeter
using LinearAlgebra
using ImageFiltering
using SpecialFunctions

# abbreviations for commonly used types
import Base: AbstractArray as AA
import Base: AbstractFloat as AF
import CUDA: CuArray as CA, CuDeviceMatrix as CDM, CuDeviceVector as CDV

# determine if there is a GPU
if CUDA.functional()
    const GPU_DEFAULT = true
else
    const GPU_DEFAULT = false
end

# configure directories
include("config.jl")

# ancillary functions + constants
include("utils.jl")
include("constants.jl")
include("functions.jl")
include("expint.jl")

# structures
include("structures/ConvolutionMemory.jl")
include("structures/Atmosphere.jl")
include("structures/GPUMemory.jl")
include("structures/ContFunc.jl")
include("structures/StellarProps.jl")
include("structures/FormTempResult.jl")
include("structures/CPUTileWorkspace.jl")

# numerical stuff
include("interpolations.jl")

# microturbulence
include("microturbulence.jl")

# macroturbulence
include("macroturbulence/gray_rot.jl")
include("macroturbulence/rad_tan.jl")
include("macroturbulence/rad_tan_two.jl")
include("macroturbulence/iso_rad_tan.jl")
include("macroturbulence/hirano.jl")
# include("macroturbulence/iso_gaussian.jl")

# instrumental profile
include("instrumental.jl")

# chemical equilibrium and absorption coefficients
include("absorption.jl")
include("structures/AlphaCache.jl")

# radaiative transport
include("contribution.jl")
include("tau.jl")

# disk integration
include("geometry.jl")
include("disk_calculations.jl")

# convenient high-level functions
include("convenience.jl")
include("turb_fits.jl")

export round_to_power, elav, searchsortednearest,
       calc_formation_temp, StellarProps,
       FormTempResult, vmac_fit, vmic_fit, convolve_gray_rotation,
       convolve_hirano_rotmacro, convolve_iso_rt_macro,
       convolve_rt_macro, gray_rot_kernel, hirano_rotmacro_ft_kernel,
       gray_iso_rt_macro_kernel, rt_macro_kernel, convolve_instrument_gauss,
       rebin_spectrum, Atmosphere, get_τs, get_zs, get_Ts, AlphaCache

end
