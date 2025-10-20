"""
Equation 17.8 from Gray (2008), assuming A_R = A_T and ξ_R = ξ_T
"""
function gray_rt_macro_kernel(vs::AA{T,1}, ζ_rt::T) where T<:AF
    t1 = 2.0 .* exp.(-1.0 .* (vs ./ ζ_rt).^2.0) ./ (sqrt(π) .* ζ_rt)
    t2 = -2.0 .* abs.(vs) .* erfc.(abs.(vs) ./ ζ_rt) ./ ζ_rt.^2.0
    kernel = t1 .+ t2
    return kernel ./ sum(kernel)
end


function convolve_gray_rt_macro(xs::AA{T,1}, ys::AA{T,1}, ζ_rt::T) where T<:AF
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rt_macro_kernel(vs, ζ_rt)

    # return convolution
    return imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
end 

function convolve_gray_rt_macro(xs::AA{T,1}, ys::AA{T,2}, ζ_rt::T) where T<:AF
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rt_macro_kernel(vs, ζ_rt)

    # allocate array for output spectrum
    ys_out = zeros(size(ys))
    for t in axes(ys, 1)
        ys_out[t, :] .= imfilter(ys[t, :], reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
    end
    return ys_out
end 