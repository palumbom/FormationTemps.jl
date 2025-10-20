"""
Equation 17.8 from Gray (2008), assuming A_R = A_T and ξ_R = ξ_T
"""
function gray_rt_macro_kernel(vs::AA{T,1}, ζ_rt::T) where T<:AF
    t1 = 2.0 .* exp.(-1.0 .* (vs ./ ζ_rt).^2.0) ./ (sqrt(π) .* ζ_rt)
    t2 = -2.0 .* abs.(vs) .* erfc.(abs.(vs) ./ ζ_rt) ./ ζ_rt.^2.0
    kernel = t1 .+ t2
    return kernel ./ sum(kernel)
end


function convolve_gray_rt_macro()
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rt_macro_kernel(vs, vsini)

    # return convolution
    return imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
end 

function convolve_gray_rt_macro()
    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # offset the kernel by the velocity
    λ0 = mean(xs)
    vs = c_ms .* (xs .- λ0) ./ λ0

    # get the normalized kernel
    kernel = gray_rt_macro_kernel(vs, vsini)

    # return convolution
    return imfilter(ys, reflect(centered(kernel)), Pad(:replicate), ImageFiltering.FFT())
end 