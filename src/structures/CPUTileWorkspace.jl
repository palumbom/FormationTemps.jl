"""
Pre-allocated per-thread working arrays and FFT infrastructure for the CPU
disk integration tile loop. Eliminates per-tile heap allocations so the
threaded loop scales without GC contention. Uses padded linear convolution
with edge replication and R2C FFT plans, matching the GPU path.
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
    v_los_buf::Vector{T}

    # FFT working buffers (padded length L)
    kernel_real::Vector{T}
    signal_padded::Vector{T}
    result_buf::Vector{T}
    kernel_ft::Vector{Complex{T}}
    signal_ft::Vector{Complex{T}}

    # padding geometry
    L::Int
    pad_left::Int
    Nλ::Int

    # pre-computed FFTW R2C plans
    fft_plan::P1
    ifft_plan::P2
end

function CPUTileWorkspace(::Type{T}, Natm::Int, Nλ::Int; Npad::Int=512) where T<:AF
    L, _, pad_left, _ = _conv_mem_geometry(Nλ, Npad)
    Nft = L ÷ 2 + 1

    # R2C plans operate on length-L real buffers
    rfft_buf = zeros(T, L)
    fwd = plan_rfft(rfft_buf)
    bwd = plan_irfft(zeros(Complex{T}, Nft), L)

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
        zeros(T, L),
        zeros(T, L),
        zeros(T, L),
        zeros(Complex{T}, Nft),
        zeros(Complex{T}, Nft),
        L,
        pad_left,
        Nλ,
        fwd,
        bwd,
    )
end

"""
    _apply_fft_kernel!(out, ys, kernel_ft, ws, Nrows)

Convolve the first `Nrows` rows of `ys` with a pre-computed kernel FFT, writing
results into `out`. Uses padded linear convolution with edge replication and
R2C FFT plans from `ws` (zero heap allocation per row).
"""
function _apply_fft_kernel!(out::AA{T,2}, ys::AA{T,2}, kernel_ft::Vector{Complex{T}},
                            ws::CPUTileWorkspace, Nrows::Int) where T<:AF
    for t in 1:Nrows
        # pad row with edge replication
        _pad_edges!(ws.signal_padded, view(ys, t, :), ws.pad_left, ws.Nλ)
        # R2C forward FFT
        mul!(ws.signal_ft, ws.fft_plan, ws.signal_padded)
        ws.signal_ft .*= kernel_ft
        # R2C inverse FFT
        mul!(ws.result_buf, ws.ifft_plan, ws.signal_ft)
        # extract valid region
        @inbounds for j in 1:ws.Nλ
            out[t, j] = ws.result_buf[ws.pad_left + j]
        end
    end
    return nothing
end
