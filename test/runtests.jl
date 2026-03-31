using FormationTemps; FT = FormationTemps
using Korg
using FFTW
using Printf
using Statistics
using ProgressMeter
using Test

# conditional CUDA usage
global use_gpu = FT.GPU_DEFAULT
if use_gpu
    using CUDA
end

# diagnostic plots: true when running locally, false on any CI runner (CI env var set by GH Actions etc.)
const make_plots = !haskey(ENV, "CI")
const test_plotdir = joinpath(@__DIR__, "plots")
if make_plots
    mkpath(test_plotdir)
end

# run tests
# include("Aqua.jl") # [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
include("test_definitions.jl")
include("test_atmosphere.jl")
include("test_kernels.jl")
include("test_convenience.jl")
include("test_threading.jl")
include("test_inplace_convolutions.jl")

# GPU-required tests
if use_gpu
    include("compare_korg.jl")
    include("integrate_aniso.jl")
    include("cusp.jl")
    include("rotmacro_convolution_test.jl")
    include("height_test.jl")
    include("compare_cpu_gpu_convenience.jl")
    include("compare_cpu_gpu_broadening.jl")
    include("disk_int_error.jl")
    include("test_buffer_safety.jl")
    include("compare_cpu_gpu_disk_integration.jl")
    include("test_convolution_alignment.jl")
    include("test_hirano_gpu_kernel.jl")
    include("test_fused_kernels.jl")
    include("test_dual_stream.jl")
    include("test_convmem_types.jl")
    include("test_batched_kernels.jl")
    include("test_varying_sigma.jl")
    include("test_gpu_precision.jl")
end
