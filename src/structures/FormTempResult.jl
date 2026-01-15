struct FormTempResult{T<:AF}
    wavs::AA{T,1}
    flux::AA{T,1}
    form_temps::AA{T,1}
    cont_func::AA{T,2}
end
