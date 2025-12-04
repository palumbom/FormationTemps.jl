abstract type ContFunc{T<:AF} end 

struct IntensityContFunc{T<:AF} <: ContFunc{T}
    cfunc::CA{T,2}
    cfunc_dt::CA{T,2}
end

struct FluxContFunc{T<:AF} <: ContFunc{T}
    cfunc::CA{T,2}
    cfunc_dt::CA{T,2}
end

function get_cum_cfunc(cfunc::ContFunc)
    ccum = cumsum(cfunc.cfunc_dt, dims=1)
    ccum ./= maximum(ccum, dims=1)
    return ccum
end

function sum_cfunc_dt(cfunc::ContFunc)
    return sum(cfunc.cfunc_dt, dims=1)'
end

function get_intensity(cfunc::IntensityContFunc)
    return sum_cfunc_dt(cfunc)
end

function get_flux(cfunc::FluxContFunc)
    return 2π .* sum_cfunc_dt(cfunc)
end