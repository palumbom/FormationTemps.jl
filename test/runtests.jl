using FormationTemps; FT = FormationTemps
using Korg
using FFTW
using Printf
using Statistics
using ProgressMeter
using Test

# conditional CUDA useage
global use_gpu = FT.GPU_DEFAULT
if use_gpu
    using CUDA
end

# run tests
include("test_definitions.jl")
include("test_atmosphere.jl")
include("test_kernels.jl")
include("test_convenience.jl")
