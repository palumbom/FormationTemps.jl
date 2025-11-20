module FormationTemps

# general imports
using CUDA
using Korg
using FFTW
using Statistics
using AbstractFFTs
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
include("structures/AtmosphereGPU.jl")
include("structures/GPUMemory.jl")
include("structures/ContFunc.jl")

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
include("macroturbulence/iso_gaussian.jl")

# instrumental profile
include("instrumental.jl")

# linelist + stellar model stuff
include("atmosphere.jl")

# radiative transport
include("absorption.jl")
include("contribution.jl")
include("tau.jl")

# disk integration
include("geometry.jl")
include("disk_calculations.jl")

# convenient high-level functions
include("convenience.jl")

export round_to_power, elav, calc_formation_temp

end
