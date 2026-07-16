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

Cumulative contribution curve for **plotting only**. Returns an `(Natm-1, Nλ)`
cell-edge cumulative, max-normalized per column.

Do NOT invert this for a 50% "formation" median: it is a cell-edge quantity with
no top-node anchor and is max- (not total-) normalized. Pairing it with
interval-center temperatures (`elav(Ts)`) reintroduces a half-interval cool bias.
Use [`form_temps_from_cfunc`](@ref) for any 50%-cumulative median.
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