abstract type ContFunc{T<:AF} end 

struct IntensityContFunc{T<:AF} <: ContFunc{T}
    cfunc::CA{T,2}
    cfunc_dt::CA{T,2}
end

struct FluxContFunc{T<:AF} <: ContFunc{T}
    cfunc::CA{T,2}
    cfunc_dt::CA{T,2}
end

"""
    get_cum_cfunc(cfunc::ContFunc) -> Matrix

Cumulative contribution curve for plotting: an `(Natm-1, Nλ)` cell-edge cumulative,
max-normalized per column.

Not a CDF. It has no top-node anchor and is normalized by the column maximum rather than
the total, so inverting it at 0.5 against interval-center temperatures gives a formation
temperature biased half an interval cool. Use [`form_temps_from_cfunc`](@ref) instead.
"""
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