# Large Linelists & Chunked Synthesis

When synthesizing spectra from large linelists (thousands of lines spanning hundreds of Angstroms), the full wavelength range may not fit in GPU memory at once, or the single-call computation may be impractically slow. [`calc_formation_temp_chunked`](@ref) solves this by dividing the wavelength range into fixed-width chunks, computing each independently, and returning the raw per-chunk results for the caller to stitch together.

## In-memory workflow

For linelists that fit in RAM after chunking, the simplest approach returns a `Vector{FormTempResult}`:

```julia
using Korg
using FormationTemps; FT = FormationTemps

linelist = Korg.read_linelist(joinpath(FT.datdir, "Sun_VALD.lin"))
star = StellarProps(Teff=5777.0, logg=4.44, Fe_H=0.0,
                    vsini=2100.0, v_macro=3400.0, v_micro=850.0)

# returns Vector{FormTempResult}, one per chunk
chunks = calc_formation_temp_chunked(star, linelist;
    chunk_width=50.0, wing_padding=30.0, overlap=5.0,
    Δλ=0.01, convolve=false, Nϕ=32)
```

Each element of `chunks` is a full [`FormTempResult`](@ref) with `wavs`, `flux`, `form_temps`, `cont_func`, and `atmosphere` fields. Adjacent chunks overlap by `overlap` Angstroms, so the caller can blend or cut at the midpoint.

## Streaming to disk

For very large linelists where even the accumulated `Vector{FormTempResult}` would exceed available memory, pass a `callback` function. This streams each chunk to disk as it completes, and `calc_formation_temp_chunked` returns `nothing`.

The following example is from `scripts/generate_temp_spectrum.jl` and writes each chunk to an HDF5 group:

```@eval
using Markdown
let
    script = joinpath(@__DIR__, "..", "..", "scripts", "generate_temp_spectrum.jl")
    lines = readlines(script)
    tag = "callback"
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

The `callback` signature is `(chunk_idx::Int, result::FormTempResult, ll_chunk) -> nothing`, where `ll_chunk` is the padded linelist view used for that chunk.

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

The key parameters are:

| Parameter | Default | Description |
|:--|:--|:--|
| `chunk_width` | 50.0 | Width of each wavelength chunk in Angstroms. |
| `wing_padding` | 30.0 | Extra range (Angstroms) beyond each chunk edge for linelist selection. Ensures that broad line wings (e.g., H-alpha, Ca II) contribute correctly at chunk boundaries. |
| `overlap` | 5.0 | Overlap width (Angstroms) between adjacent chunks. Used during post-hoc stitching to blend or cut. |
| `Δλ` | 0.001 | Wavelength step size in Angstroms. |
| `buffer` | 2.0 | Extra wavelength range (Angstroms) beyond the linelist edges. |

!!! tip "Choosing `wing_padding`"
    Too small a `wing_padding` causes lines near chunk edges to be computed without their full wings, introducing residuals at stitch boundaries. 30 Angstroms is conservative; 15 Angstroms suffices for most lines but not for very broad resonant lines (e.g., H-alpha).

## Stitching chunks

`calc_formation_temp_chunked` returns raw chunks that need to be stitched together.

The below example from `scripts/generate_temp_spectrum.jl` linearly crossfades flux, formation temperatures, and contribution functions across the overlap region.

```@eval
using Markdown
let
    script = joinpath(@__DIR__, "..", "..", "scripts", "generate_temp_spectrum.jl")
    lines = readlines(script)
    tag = "blend"
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

The full production script combining all of the above is at `scripts/generate_temp_spectrum.jl`. It:

1. Loads a large VALD linelist and converts to air wavelengths.
2. Streams chunked results to HDF5.
3. Reads the raw chunks back and stitches them.
4. Trims the result to the linelist extent and writes the final spliced output.
