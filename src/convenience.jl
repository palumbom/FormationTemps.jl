"""
    calc_formation_temp()

Compute TODO

# Examples
```julia-repl
TODO
```
"""
function calc_formation_temp(linelist, Teff, logg, A_X; use_gpu::Bool=GPU_DEFAULT)
    if use_gpu
        _calc_formation_temp_gpu()
    else
        _calc_formation_temp_cpu()
    end
    return nothing
end

function _calc_formation_temp_cpu()

    return nothing
end

function _calc_formation_temp_gpu()

    return nothing
end

