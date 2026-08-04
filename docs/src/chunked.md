# Large Linelists & Chunked Synthesis

Large linelists (thousands of lines over hundreds of Angstroms) may not fit in GPU memory in a single call, or may simply be too slow. [`calc_formation_temp_chunked`](@ref) splits the wavelength range into fixed-width chunks, computes each independently, and returns the raw per-chunk results for the caller to stitch.

## In-memory workflow

Returns a `Vector{FormTempResult}`, one per chunk:

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

Each element is a full [`FormTempResult`](@ref) (`wavs`, `flux`, `form_temps`, `cont_func`, `ceiling_ratio`, `r_thresh`, `atmosphere`). Adjacent chunks overlap by `overlap` Angstroms, so you can blend or cut at the midpoint.

`ceiling_ratio` is a per-chunk reduction over `cont_func`. When stitching, recompute it from the stitched contribution function instead of concatenating the per-chunk vectors: [`ceiling_ratio`](@ref) accepts a bare matrix alongside the atmosphere's reference optical depth grid, which it needs because the statistic compares two layer intervals of unequal width.

```julia
r = ceiling_ratio(stitched_cont_func, get_τs(results[1].atmosphere))
```

## Choosing a method

Chunking multiplies the cost of one `calc_formation_temp` call by the number of chunks, so the [integration method](@ref "Integration Methods") matters here. `method=:quadrature` reproduces the `:disk` physics, inclination and differential rotation included, at a fraction of the cost — prefer it over coarsening `Nϕ`.

## Streaming to disk

Pass a `callback` to write each chunk out as it completes; `calc_formation_temp_chunked` then returns `nothing` instead of accumulating results. From `scripts/generate_temp_spectrum.jl`, writing each chunk to an HDF5 group:

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

`callback` is `(chunk_idx::Int, result::FormTempResult, ll_chunk) -> nothing`, where `ll_chunk` is the padded linelist view for that chunk.

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
| `wing_padding` | 30.0 | Extra range (Angstroms) beyond each chunk edge for linelist selection, so broad wings (H-alpha, Ca II) contribute at chunk boundaries. |
| `overlap` | 5.0 | Overlap width (Angstroms) between adjacent chunks, for blending or cutting when stitching. |
| `Δλ` | 0.001 | Wavelength step size in Angstroms. |
| `buffer` | 2.0 | Extra wavelength range (Angstroms) beyond the linelist edges. |

`method`, `Nϕ`, `Nμ`, `N_az`, `use_gpu`, `gpu_precision` and `r_thresh` are forwarded unchanged to [`calc_formation_temp`](@ref).

!!! tip "Choosing `wing_padding`"
    Too small, and lines near chunk edges lose their wings, leaving residuals at stitch boundaries. 30 Angstroms is conservative; 15 suffices for all but the broadest lines.

!!! tip "Choosing `overlap`"
    The broadening kernels pad by edge replication, so within one kernel half-support of a chunk edge the result draws on replicated samples rather than the true neighbouring spectrum. Cutting at the overlap midpoint discards that band only if `overlap` exceeds twice the half-support, `(vsini + 3ζ + 3ξ)/c · λ`.

    That half-support is ~0.3 Angstroms for a solar-like star at 6000 Å, well inside the 5-Angstrom default, but ~2.3 Angstroms at `vsini = 100` km/s, where the default is marginal. Scale `overlap` with `vsini`.

## Stitching chunks

Crossfading flux, formation temperatures, and contribution functions across the overlap, from `scripts/generate_temp_spectrum.jl`:

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

`scripts/generate_temp_spectrum.jl` combines all of the above:

1. Loads a large VALD linelist and converts to air wavelengths.
2. Streams chunked results to HDF5.
3. Reads the raw chunks back and stitches them.
4. Trims to the linelist extent and writes the spliced output.
