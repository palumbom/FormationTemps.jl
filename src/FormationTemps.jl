module FormationTemps

# general imports
using CUDA
using Korg
using AbstractFFTs
using ImageFiltering
using SpecialFunctions

# abbreviations for commonly used types
import Base: AbstractArray as AA
import Base: AbstractFloat as AF
import CUDA: CuArray as CA, CuDeviceMatrix as CDM, CuDeviceVector as CDV

# configure directories
include("config.jl")

# ancillary functions + constants
include("utils.jl")
include("interpolations.jl")

# structures
include("structures/ConvolutionMemory.jl")
include("structures/AtmosphereGPU.jl")
include("structures/GPUMemory.jl")

# linelist + stellar model stuff
include("linelist.jl")

# radiative transport
include("absorption.jl")
include("contribution.jl")
include("tau.jl")

export round_to_power

end
