# Large Linelists & Chunked Synthesis

A linelist covering hundreds or thousands of Angstroms may not fit in GPU memory in a single call, and may simply be too slow to compute all at once. ```calc_formation_temp_chunked``` splits the wavelength range into fixed-width chunks and computes each one independently. It hands back the raw per-chunk results and leaves the stitching to you.

## In-memory workflow

By default we get back a `Vector{FormTempResult}`, one element per chunk:

```julia
using Korg
using FormationTemps; FT = FormationTemps

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0,
                    vsini=2100.0, v_macro=3400.0, v_micro=850.0)

chunks = calc_formation_temp_chunked(star, linelist;
    chunk_width=50.0, wing_padding=30.0, overlap=5.0,
    Δλ=0.01, method=:quadrature)
```

Each element is a full ```FormTempResult```, so we have `wavs`, `flux`, `form_temps`, `cont_func`, `ceiling_ratio`, `r_thresh`, and `atmosphere` for every chunk. Adjacent chunks overlap by `overlap` Angstroms, which gives us room to blend across the seam or to cut at its midpoint.

One thing to watch out for: `ceiling_ratio` is a per-chunk reduction, so we should not just concatenate the per-chunk vectors when stitching. Recompute it from the stitched contribution function instead. ```ceiling_ratio``` will accept a bare matrix, along with the reference optical depth grid it needs in order to compare layer intervals of unequal width:

```julia
r = ceiling_ratio(stitched_cont_func, get_τs(chunks[1].atmosphere))
```

## Choosing a method

Chunking multiplies the cost of a single ```calc_formation_temp``` call by the number of chunks, so the [integration method](@ref "Integration Methods") matters a great deal here. `method=:quadrature` reproduces the `:disk` physics, inclination and differential rotation included, at a fraction of the cost. Reach for it before coarsening `Nϕ`.

## Streaming to disk

For a long spectrum the accumulated results will not fit in memory. Passing a `callback` writes each chunk out as it completes, and ```calc_formation_temp_chunked``` then returns `nothing` rather than accumulating anything. Here is the callback from `scripts/generate_temp_spectrum.jl`, which writes each chunk into its own HDF5 group:

```@eval
using Markdown
let
    script = joinpath(@__DIR__, "..", "..", "scripts", "generate_temp_spectrum.jl")
    lines = readlines(script)
    tag = "writechunk"
    start_marker = "# [doc:$(tag)-start]"
    end_marker   = "# [doc:$(tag)-end]"
    i_start = findfirst(l -> strip(l) == start_marker, lines)
    i_end   = findfirst(l -> strip(l) == end_marker, lines)
    section = if isnothing(i_start) || isnothing(i_end)
        "# section '$tag' not found"
    else
        s = lines[(i_start + 1):(i_end - 1)]
        while !isempty(s) && isempty(strip(s[end]))
            pop!(s)
        end
        join(s, "\n")
    end
    Markdown.parse("```julia\n" * section * "\n```")
end
```

The callback signature is `(chunk_idx::Int, result::FormTempResult, ll_chunk) -> nothing`, where `ll_chunk` is the padded linelist view used for that chunk.

## Chunking parameters

```@eval
using Markdown
let
    script = joinpath(@__DIR__, "..", "..", "scripts", "generate_temp_spectrum.jl")
    lines = readlines(script)
    tag = "chunked"
    start_marker = "# [doc:$(tag)-start]"
    end_marker   = "# [doc:$(tag)-end]"
    i_start = findfirst(l -> strip(l) == start_marker, lines)
    i_end   = findfirst(l -> strip(l) == end_marker, lines)
    section = if isnothing(i_start) || isnothing(i_end)
        "# section '$tag' not found"
    else
        s = lines[(i_start + 1):(i_end - 1)]
        while !isempty(s) && isempty(strip(s[end]))
            pop!(s)
        end
        join(s, "\n")
    end
    Markdown.parse("```julia\n" * section * "\n```")
end
```

| Parameter | Default | Description |
|:--|:--|:--|
| `chunk_width` | 50.0 | Width of each wavelength chunk in Angstroms. |
| `wing_padding` | 30.0 | Extra range (Angstroms) beyond each chunk edge for linelist selection, so that broad wings (H-alpha, Ca II) still contribute at chunk boundaries. |
| `overlap` | 5.0 | Overlap width (Angstroms) between adjacent chunks, for blending or cutting when stitching. |
| `Δλ` | 0.001 | Wavelength step size in Angstroms. |
| `buffer` | 2.0 | Extra wavelength range (Angstroms) beyond the linelist edges. |

`method`, `Nϕ`, `Nμ`, `N_az`, `use_gpu`, `gpu_precision`, `r_thresh` and `ne_warn_thresh` are forwarded unchanged to ```calc_formation_temp```.

!!! tip "Choosing `wing_padding`"
    If this is too small, lines near a chunk edge lose their wings and we are left with residuals at the stitch boundaries. 30 Angstroms is conservative; 15 is enough for all but the broadest lines.

!!! tip "Choosing `overlap`"
    The broadening kernels pad by edge replication, so within one kernel half-support of a chunk edge the result draws on replicated samples rather than on the true neighboring spectrum. Cutting at the overlap midpoint only discards that band if `overlap` is larger than twice the half-support, roughly ``(v \sin i + 3\zeta + 3\xi)/c \cdot \lambda``. For a solar-like star this sits comfortably inside the 5-Angstrom default, but it scales with ``v \sin i``, and by 100 km/s the default is marginal. Scale `overlap` up for rapid rotators.

## Stitching chunks

Here we read the chunks back and crossfade the flux, formation temperatures, and contribution functions across each overlap. The accumulator is preallocated at its final width, since growing it one chunk at a time costs ``O(N^2)`` bytes copied:

```@eval
using Markdown
let
    script = joinpath(@__DIR__, "..", "..", "scripts", "generate_temp_spectrum.jl")
    lines = readlines(script)
    tag = "crossfade"
    start_marker = "# [doc:$(tag)-start]"
    end_marker   = "# [doc:$(tag)-end]"
    i_start = findfirst(l -> strip(l) == start_marker, lines)
    i_end   = findfirst(l -> strip(l) == end_marker, lines)
    section = if isnothing(i_start) || isnothing(i_end)
        "# section '$tag' not found"
    else
        s = lines[(i_start + 1):(i_end - 1)]
        while !isempty(s) && isempty(strip(s[end]))
            pop!(s)
        end
        join(s, "\n")
    end
    Markdown.parse("```julia\n" * section * "\n```")
end
```

## Complete example

`scripts/generate_temp_spectrum.jl` puts all of the above together. It streams chunked results to HDF5, reads the raw chunks back and stitches them, trims to the linelist extent, and writes the spliced spectrum alongside the model atmosphere and a per-pixel quality mask.

!!! note "Air and vacuum wavelengths"
    Korg computes in vacuum, so the synthesis runs on the vacuum grid that `read_linelist` returns and an air axis is applied only as a relabeling of the stored output. Rewriting the linelist to air instead would move the metal lines while leaving Korg's internally tabulated hydrogen lines where they were, putting the two roughly 1.8 Angstroms apart at H-alpha.
