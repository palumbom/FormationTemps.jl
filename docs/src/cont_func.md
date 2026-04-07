# Manipulating Contribution Functions

Formation temperatures are computed from the model contribution functions, and can be thought of as a simple summary statistic thereof. For some users, it may be advantageous to view and manipulate these contribution functions. 

Following a procedure like that in the [Basic Tutorial](@ref "Basic Tutorial"), we can see that the ```FormTempResult``` composite type also contains a ```cont_func``` field. We can plot this like so:

```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "cont_func.jl"), String)
break_marker = "# BREAK1"
stop_idx = findfirst(break_marker, code)
code = stop_idx === nothing ? code : code[1:prevind(code, stop_idx.start)]
Markdown.parse("```julia\n" * code * "\n```")
```
![formation_temps](static/cont_func_simple_example.png)

We can also add axis values. The x-axis is wavelength, and y-axis is a coordinate in the model atmosphere. Let's first parse out and view the model atmosphere structure from the ```FormTempResult``` composite type instance. 

```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "cont_func.jl"), String)
start_marker = "# BREAK1"
end_marker = "# BREAK2"
start_idx = findfirst(start_marker, code)
end_idx = findfirst(end_marker, code)
if start_idx !== nothing && end_idx !== nothing && start_idx.start < end_idx.start
    start_nl = findnext('\n', code, start_idx.start)
    slice_start = start_nl === nothing ? lastindex(code) + 1 : nextind(code, start_nl)
    slice_end = prevind(code, end_idx.start)
    code = slice_start <= slice_end ? code[slice_start:slice_end] : ""
end
Markdown.parse("```julia\n" * code * "\n```")
```
![formation_temps](static/atmosphere.png)

Now we can zoom-in on some lines (to better see the structure of the contribution function) and add on the axis values:
```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "cont_func.jl"), String)
start_marker = "# BREAK2"
end_marker = "# BREAK3"
start_idx = findfirst(start_marker, code)
end_idx = findfirst(end_marker, code)
if start_idx !== nothing && end_idx !== nothing && start_idx.start < end_idx.start
    start_nl = findnext('\n', code, start_idx.start)
    slice_start = start_nl === nothing ? lastindex(code) + 1 : nextind(code, start_nl)
    slice_end = prevind(code, end_idx.start)
    code = slice_start <= slice_end ? code[slice_start:slice_end] : ""
end
Markdown.parse("```julia\n" * code * "\n```")
```
![formation_temps](static/cont_func_example.png)

## Formation temperatures can lie to you!

We can also slice through the contribution functions of individual pixels in lines. Doing so, we can demonstrate that wavelength elements which share a formation temperature need not have the same (or even similar) contribution functions! In each frame of the below animations, the contribution functions for the black pixel in each absorption line are shown. These pixels share the same formation temperature (within a tolerance of a couple Kelvin), yet their contribution functions look very different in the cores of the lines! Of course, the contribution functions tend to each other toward the continuum, but the bulk of the radial velocity information is found where the derivative of the spectrum is highest. 

![line_animation](static/line_lineup.gif)
![cont_func_animation](static/cont_comparison.gif)