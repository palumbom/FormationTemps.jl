"""
Pre-allocated per-thread working arrays and FFT infrastructure for the CPU
disk integration tile loop. Eliminates per-tile heap allocations so the
threaded loop scales without GC contention.
"""
struct CPUTileWorkspace{T<:AF, P1, P2}
    # radiative transfer working arrays
    τs_int::Matrix{T}
    τs_int_cont::Matrix{T}
    cfunc_int::Matrix{T}
    cfunc_int_cont::Matrix{T}
    cfunc_dt_int::Matrix{T}
    cfunc_dt_int_cont::Matrix{T}
    cfunc_flux_acc::Matrix{T}
    cfunc_flux_cont_acc::Matrix{T}

    # convolution output buffers
    αs_broad::Matrix{T}
    αs_cont_broad::Matrix{T}
    macro_out::Matrix{T}

    # per-tile velocity buffer (length Natm)
    μ_v_buf::Vector{T}

    # FFT working buffers (length Nλ)
    kernel_real::Vector{T}
    row_buf::Vector{Complex{T}}
    kernel_ft::Vector{Complex{T}}

    # pre-computed in-place FFTW plans
    fft_plan::P1
    ifft_plan::P2
end

function CPUTileWorkspace(::Type{T}, Natm::Int, Nλ::Int) where T<:AF
    # create in-place FFT plans (one per thread for thread safety)
    plan_buf = zeros(Complex{T}, Nλ)
    fwd = plan_fft!(plan_buf)
    bwd = plan_ifft!(plan_buf)

    return CPUTileWorkspace(
        zeros(T, Natm, Nλ),
        zeros(T, Natm, Nλ),
        zeros(T, Natm - 1, Nλ),
        zeros(T, Natm - 1, Nλ),
        zeros(T, Natm - 1, Nλ),
        zeros(T, Natm - 1, Nλ),
        zeros(T, Natm - 1, Nλ),
        zeros(T, Natm - 1, Nλ),
        zeros(T, Natm, Nλ),
        zeros(T, Natm, Nλ),
        zeros(T, Natm - 1, Nλ),
        zeros(T, Natm),
        zeros(T, Nλ),
        zeros(Complex{T}, Nλ),
        zeros(Complex{T}, Nλ),
        fwd,
        bwd,
    )
end

"""
    _apply_fft_kernel!(out, ys, kernel_ft, ws, Nrows)

Convolve the first `Nrows` rows of `ys` with a pre-computed kernel FFT, writing
results into `out`. Uses the FFT buffers and in-place plans from `ws` (zero
heap allocation per row).
"""
function _apply_fft_kernel!(out::AA{T,2}, ys::AA{T,2}, kernel_ft::Vector{Complex{T}},
                            ws::CPUTileWorkspace, Nrows::Int) where T<:AF
    for t in 1:Nrows
        @inbounds for j in eachindex(ws.row_buf)
            ws.row_buf[j] = complex(ys[t, j])
        end
        ws.fft_plan * ws.row_buf
        ws.row_buf .*= kernel_ft
        ws.ifft_plan * ws.row_buf
        @inbounds for j in axes(out, 2)
            out[t, j] = real(ws.row_buf[j])
        end
    end
    return nothing
end
