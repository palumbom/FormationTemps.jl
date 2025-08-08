module FormationTemps

# general imports
using Anemoi

# abbreviations for commonly used types
import Base: AbstractArray as AA
import Base: AbstractFloat as AF
import CUDA: CuArray as CA, CuDeviceMatrix as CDM, CuDeviceVector as CDV

# configure directories
include("config.jl")

# ancillary functions + constants
include("utils.jl")

end
