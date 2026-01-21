# follows from implementation at https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
function get_bin_edges(arr::AA{T,1}) where T<:AF
    edges = zeros(length(arr)+1)
    edges[1] = arr[1] - 0.5 * (arr[2] - arr[1])
    edges[end] = arr[end] + 0.5 * (arr[end] - arr[end-1])
    edges[2:end-1] = 0.5 .* (arr[2:end] .+ arr[1:end-1])
    widths = diff(edges)
    return edges, widths
end

"""
    rebin_spectrum(xs_old, ys_old, xs_new)
    rebin_spectrum(xs_old, ys_old, σs_old, xs_new)

Rebin a spectrum (and optional uncertainties) from `xs_old` to `xs_new`. # follows from implementation at https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py

TODO: finish docs.
"""

# TODO algorithm might cause shift based on input wavelength grid?
# might cause issue if wavelengths shift across
function rebin_spectrum(xs_old::AA{T,1}, ys_old::AA{T,1}, xs_new::AA{T,1}) where T<:AF
    @assert issorted(xs_old)
    @assert issorted(xs_new)

    # get edges of bins
    old_edges, old_widths = get_bin_edges(xs_old)
    new_edges, new_widths = get_bin_edges(xs_new)

    # allocate memory for output array
    ys_new = zeros(length(xs_new))

    # loop over new bins
    start = 0
    stop = 0
    for i in eachindex(xs_new)
        # boundary conditions
        if new_edges[i] < first(old_edges)
            ys_new[i] = first(ys_old)
            continue
        elseif new_edges[i+1] > last(old_edges)
            ys_new[i] = last(ys_old)
            continue
        end

        while old_edges[start+1] <= new_edges[i]
            start += 1
        end

        while old_edges[stop+1] < new_edges[i+1]
            stop += 1
        end

        if start == stop
            ys_new[i] = ys_old[start]
        else
            start_factor = ((old_edges[start+1] - new_edges[i]) / (old_edges[start+1] - old_edges[start]))
            stop_factor = ((new_edges[i+1] - old_edges[stop]) / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= stop_factor

            f_widths = old_widths[start:stop] .* ys_old[start:stop]
            ys_new[i] = sum(f_widths)
            ys_new[i] /= sum(old_widths[start:stop])

            old_widths[start] /= start_factor
            old_widths[stop] /= stop_factor
        end
    end
    return ys_new
end

# follows from implementation at https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
function rebin_spectrum(xs_old::AA{T,1}, ys_old::AA{T,1}, σs_old::AA{T,1}, xs_new::AA{T,1}, ) where T<:AF
    @assert length(σs_old) == length(ys_old)
    @assert issorted(xs_old)
    @assert issorted(xs_new)

    # get edges of bins
    old_edges, old_widths = get_bin_edges(xs_old)
    new_edges, new_widths = get_bin_edges(xs_new)

    # allocate memory for output arrays
    ys_new = zeros(length(xs_new))
    σs_new = zeros(length(xs_new))

    # loop over new bins
    start = 0
    stop = 0
    for i in eachindex(xs_new)
        # boundary conditions
        if new_edges[i] < first(old_edges)
            ys_new[i] = first(ys_old)
            σs_new[i] = first(σs_old)
            continue
        elseif new_edges[i+1] > last(old_edges)
            ys_new[i] = last(ys_old)
            σs_new[i] = last(σs_old)
            continue
        end

        while old_edges[start+1] <= new_edges[i]
            start += 1
        end

        while old_edges[stop+1] < new_edges[i+1]
            stop += 1
        end

        if start == stop
            ys_new[i] = ys_old[start]
            σs_new[i] = σs_old[start]
        else
            start_factor = ((old_edges[start+1] - new_edges[i]) / (old_edges[start+1] - old_edges[start]))
            stop_factor = ((new_edges[i+1] - old_edges[stop]) / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= stop_factor

            f_widths = old_widths[start:stop] .* ys_old[start:stop]
            ys_new[i] = sum(f_widths)
            ys_new[i] /= sum(old_widths[start:stop])

            e_wid = old_widths[start:stop] .* σs_old[start:stop]
            σs_new[i] = sqrt(sum(e_wid.^2.0))
            σs_new[i] /= sum(old_widths[start:stop])

            old_widths[start] /= start_factor
            old_widths[stop] /= stop_factor
        end
    end
    return ys_new, σs_new
end

"""
    convolve_instrument_gauss(xs, ys; new_res=1.17e5, oversampling=2.0)

Convolve a spectrum with a Gaussian LSF at resolving power `new_res`.

TODO: finish docs.
"""
function convolve_instrument_gauss(xs::AA{T,1}, ys::AA{T,1}; new_res::T=1.17e5,
                                   oversampling::T=2.0) where T<:AF
    # get kernel
    σ(x) = x / new_res / (2.0 * sqrt(2 * log(2)))
    g(x, n) = (one(T)/(σ(x) * sqrt(2.0 * π))) * exp(-0.5 * ((x - n)/σ(x))^2)

    # offset the kernel by the velocity
    i0 = length(xs) ÷ 2 + 1
    λ0 = xs[i0]
    λc = λ0

    # sample and normalize the kernel
    kernel = g.(xs, λc)
    kernel ./= sum(kernel)

    # convolve it
    convolved = imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())

    # get wavelength grid at lower resolution
    Δlnλ = 1.0 / new_res
    lnλs = range(log(first(xs)), log(last(xs)), step=Δlnλ/oversampling)
    xs_out = exp.(lnλs)

    # return rebinned spectrum
    return xs_out, rebin_spectrum(xs, convolved, xs_out)
end
