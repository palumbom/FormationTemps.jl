"""
Equation 18.14 from The Observation and Analysis of Stellar Photospheres
(Gray 2008)
"""
function gray_rot_kernel(vs::AA{T,1}, vsini::T, u1::T) where T<:AF
    # get LD terms
    ld1 = 2.0 * (one(T) - u1)
    ld2 = 0.5 * π * u1 
    ld3 = π * (one(T) - u1 / 3.0)

    # evaluate the kernel 
    xs = vs ./ vsini
    omx2 = abs.(one(T) .- xs .^ 2.0)
    kernel = (ld1 .* sqrt.(omx2) .+ ld2 .* omx2) ./ ld3
    kernel[abs.(xs) .> one(T)] .= zero(T)
    return kernel ./ sum(kernel)
end


"""
Equation B12 from Hirano et al. (2011). NOTE: This is returns the Fourier
Transform of the convolution kernel!! 
"""
function hirano_rotmacro_ft_kernel(vs, σ, ξmac, vsini, u1, u2)

    return nothing 
end

function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,1}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rot_kernel(vs, vsini, u1)

    # return convolution
    return imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
end

function convolve_gray_rotation(xs::AA{T,1}, ys::AA{T,2}, vsini::T, u1::T) where T<:AF
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rot_kernel(vs, vsini, u1)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= imfilter(ys[t, :], reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
    end
    return ys_out
end