# Curated mask for lines whose formation temperature is not trustworthy under LTE in a 1D
# static atmosphere. Complements FT.boundary_mask, which detects only that the contribution
# function had not decayed by the top of the truncated model — a statement about the
# integration domain, not about the physics.
#
# Included by scripts/generate_temp_spectrum.jl and by test/test_bad_lines_mask.jl. Holds
# no pipeline state, so it is safe to include from either.

using CSV, DataFrames, Printf
using FormationTemps: c_ms
import Korg

# Mask bits. 0x01 is the existing boundary mask, so the two compose in one UInt8 array.
const MASK_BOUNDARY = 0x01
const MASK_CHROMO   = 0x02
const MASK_NLTE     = 0x04
const MASK_LINEDATA = 0x08

const MASK_BIT_MEANINGS = ["0x01 boundary", "0x02 chromo", "0x04 nlte", "0x08 linedata"]

# The data file carries flag names, so a misspelling is an error rather than a wrong bit.
const FLAG_CODES = Dict("chromo"   => MASK_CHROMO,
                        "nlte"     => MASK_NLTE,
                        "linedata" => MASK_LINEDATA)

# Growth knobs. These tune the algorithm and stay in code; the data file records which
# transitions are suspect, which is scientific judgement.
const N_SIGMA_HALO       = 5.0    # broadening widths covered by the minimum halo
const DEPTH_THRESH       = 0.4    # absolute depth floor for growth
const CORE_FRAC          = 0.5    # growth stops below this fraction of the line's core depth
const MIN_CORE_DEPTH     = 0.05   # below this the seed is not a detected line
const MAX_EXTENT_DEFAULT = 5.0    # Angstrom half-width cap, overridable per row

# Korg synthesizes hydrogen lines from its own data rather than from the linelist, so an H I
# entry is present in the spectrum regardless of what the linelist holds.
const SPECIES_NOT_IN_LINELIST = Set([Korg.Species("H I")])

"""
    read_bad_lines(path; vacuum, max_extent_default) -> Vector{NamedTuple}

Parse the curated bad-lines CSV into `(λ, species, label, flag, max_extent)` entries,
ascending in λ.

`λ` is in the frame the caller asks for: the file tabulates air wavelengths (hence the
required column name `lambda_air`), converted with `Korg.air_to_vacuum` when `vacuum`.
A blank `max_extent` cell inherits `max_extent_default`.

Validation is deliberately loud: a hand-edited data file gets none of the checking a
compiler would give a literal table.
"""
function read_bad_lines(path::AbstractString; vacuum::Bool, max_extent_default::Real)
    df = CSV.read(path, DataFrame; comment="#", stripwhitespace=true)
    cols = names(df)
    for c in ("lambda_air", "species", "label", "flag")
        @assert c in cols "bad-lines file $path is missing required column '$c'"
    end

    @assert eltype(df.lambda_air) <: Real "lambda_air in $path did not parse as numbers; " *
        "check for a stray character or a misplaced delimiter"
    λ_air = Float64.(df.lambda_air)
    labels = string.(df.label)      # `string`, not `String`: a numeric-looking label parses as Int
    @assert issorted(λ_air) "lambda_air must be ascending in $path"
    @assert allunique(λ_air) "lambda_air must be unique in $path"
    @assert allunique(labels) "labels must be unique in $path"

    has_cap = "max_extent" in cols
    entries = map(1:nrow(df)) do i
        flag_name = string(df.flag[i])
        @assert haskey(FLAG_CODES, flag_name) "unknown flag '$flag_name' for " *
            "'$(labels[i])' in $path; expected one of $(sort(collect(keys(FLAG_CODES))))"
        cap = (has_cap && !ismissing(df.max_extent[i])) ? Float64(df.max_extent[i]) :
                                                          Float64(max_extent_default)
        @assert cap > 0 "max_extent must be positive for '$(labels[i])' in $path"
        λ = vacuum ? Korg.air_to_vacuum(λ_air[i]) : λ_air[i]
        # an unparseable code raises from Korg naming the offending string
        sp = Korg.Species(string(df.species[i]))
        return (λ=λ, species=sp, label=labels[i], flag=FLAG_CODES[flag_name],
                max_extent=cap)
    end
    return entries
end

"""
    verify_species_present(entries, line_wavs, line_species; n_sigma_halo, v_broad) -> Vector

Keep only entries whose species actually appears in the linelist within the seed search
halo of the tabulated wavelength.

This is the check that makes a curated label trustworthy. `grow_line_region` seeds on the
flux minimum in the halo, and in a spectrum with ~10⁵ lines that minimum is nearly always
*some* deep line — so an entry naming a transition the linelist lacks does not degrade to
"not masked", it masks an unrelated neighbour under the curated label. `min_core_depth`
cannot catch it: the neighbour is often deeper than the intended line.

`line_wavs` must be ascending and in the same frame as `entries` — both come from the
linelist the synthesis actually uses. Species in `SPECIES_NOT_IN_LINELIST` are kept without
a lookup.
"""
function verify_species_present(entries, line_wavs::AbstractVector{<:Real}, line_species;
                                n_sigma_halo::Real, v_broad::Real)
    @assert length(line_wavs) == length(line_species) "line_wavs and line_species must match"
    @assert issorted(line_wavs) "line_wavs must be ascending"

    kept = eltype(entries)[]
    for e in entries
        if e.species in SPECIES_NOT_IN_LINELIST
            push!(kept, e)
            continue
        end
        halo = e.λ * n_sigma_halo * v_broad / c_ms
        i1 = searchsortedfirst(line_wavs, e.λ - halo)
        i2 = searchsortedlast(line_wavs, e.λ + halo)
        if !any(i -> line_species[i] == e.species, i1:i2)
            @warn "bad-lines mask: '$(e.label)' names $(e.species) at " *
                  "$(round(e.λ, digits=3)) Å, but the linelist has no $(e.species) line " *
                  "within $(round(halo, digits=3)) Å of it; entry dropped. Masking it would " *
                  "flag whichever unrelated line is deepest there under this label."
            continue
        end
        push!(kept, e)
    end
    return kept
end

"""
    grow_line_region(wavs, flux, λ0; halo, depth_thresh, min_core_depth, max_extent,
                     core_frac=0.0)

Pixel range around `λ0` over which the model is not trusted, or `nothing` if no line is
detected there.

`flux` must be continuum-normalized, so line depth is `1 - flux`. Locates the core as the
flux minimum within `±halo` of `λ0`, walks outward while depth exceeds
`max(depth_thresh, core_frac * core_depth)`, unions the result with the `±halo` floor, and
clamps to `±max_extent`.

The two thresholds do different jobs. `depth_thresh` is an absolute floor. `core_frac` scales
with the line's own depth, so it selects the core of a strong and a weak line alike:
`core_frac = 0.5` stops at half depth, giving the region a full-width-half-maximum sense.
An absolute floor alone cannot do that — set low it walks out through the damping wings
(2% depth puts Hα's edge tens of Angstrom out), set high it silently declines to grow at all
for any line shallower than the floor. `core_frac = 0` restores pure absolute behaviour.

Returns `nothing` — the caller's cue to warn and skip — when the seed falls outside `wavs`
or the core is shallower than `min_core_depth`. That rejection is the guard against a
mistyped wavelength, a wrong air/vacuum conversion, or a transition the linelist lacks:
without it, `argmin` latches onto an unrelated neighbour.

`capped` in the returned tuple flags that `max_extent` and not the depth walk set an edge.
Below ~4000 Å line blanketing means no pixel clears `depth_thresh`, so the cap is what the
extent actually is there, not a safety margin.
"""
function grow_line_region(wavs::AbstractVector{<:Real}, flux::AbstractVector{<:Real},
                          λ0::Real; halo::Real, depth_thresh::Real,
                          min_core_depth::Real, max_extent::Real, core_frac::Real=0.0)
    @assert 0 <= core_frac < 1 "core_frac must be in [0, 1); at 1 growth cannot leave the core"
    @assert length(wavs) == length(flux) "wavs and flux must have equal length"
    @assert max_extent >= halo "max_extent ($max_extent Å) is below the broadening halo " *
        "($halo Å) at λ0 = $λ0; the cap would clip the minimum region"
    N = length(wavs)

    # search window; wavs is ascending, so bracket by binary search
    s_lo = searchsortedfirst(wavs, λ0 - halo)
    s_hi = searchsortedlast(wavs, λ0 + halo)
    s_lo > s_hi && return nothing            # seed's halo does not intersect the grid

    i0 = s_lo - 1 + argmin(view(flux, s_lo:s_hi))
    core_depth = 1 - Float64(flux[i0])
    core_depth < min_core_depth && return nothing

    # whichever threshold stops growth sooner
    thresh = max(depth_thresh, core_frac * core_depth)

    i_lo = i0
    while i_lo > 1 && (1 - flux[i_lo - 1]) > thresh
        i_lo -= 1
    end
    i_hi = i0
    while i_hi < N && (1 - flux[i_hi + 1]) > thresh
        i_hi += 1
    end

    # halo floor, then the cap
    i_lo = min(i_lo, s_lo)
    i_hi = max(i_hi, s_hi)
    cap_lo = clamp(searchsortedfirst(wavs, λ0 - max_extent), 1, N)
    cap_hi = clamp(searchsortedlast(wavs, λ0 + max_extent), 1, N)
    capped = i_lo < cap_lo || i_hi > cap_hi
    i_lo = max(i_lo, cap_lo)
    i_hi = min(i_hi, cap_hi)

    return (i_lo=i_lo, i_hi=i_hi, i0=i0, core_depth=core_depth, capped=capped)
end

"""
    build_line_mask(wavs, flux, entries; n_sigma_halo, depth_thresh, min_core_depth, v_broad,
                    core_frac=0.0)

Expand curated entries into a `UInt8` bitflag mask over `wavs`, plus one provenance record
per applied entry.

`v_broad` is the quadrature sum of the broadening velocities in m/s — a width proxy, not a
true Gaussian σ, since neither rotation nor radial-tangential macroturbulence is Gaussian;
`n_sigma_halo` absorbs the shape difference.

`regions` may be shorter than `entries`. An entry falling outside `wavs` is skipped
silently, since the chunked caller passes one narrow window at a time; an entry inside the
window whose line cannot be confirmed is warned about, because that one indicates a bad row.

Overlapping regions need no merging: bits are OR-ed per pixel and each entry keeps its own
record.
"""
function build_line_mask(wavs::AbstractVector{<:Real}, flux::AbstractVector{<:Real},
                         entries; n_sigma_halo::Real, depth_thresh::Real,
                         min_core_depth::Real, v_broad::Real, core_frac::Real=0.0)
    mask = zeros(UInt8, length(wavs))
    regions = Vector{NamedTuple{(:label, :λ_lo, :λ_hi, :flag, :n_pix, :core_depth, :capped),
                                Tuple{String, Float64, Float64, UInt8, Int, Float64, Bool}}}()

    for e in entries
        halo = e.λ * n_sigma_halo * v_broad / c_ms

        # A seed outside this window is not a problem worth reporting: the chunked path calls
        # this once per 50 Å chunk, so most curated lines fall in none of them. Warning here
        # would emit thousands of spurious messages across a production run.
        if e.λ + halo < first(wavs) || e.λ - halo > last(wavs)
            continue
        end

        g = grow_line_region(wavs, flux, e.λ; halo=halo, depth_thresh=depth_thresh,
                             min_core_depth=min_core_depth, max_extent=e.max_extent,
                             core_frac=core_frac)
        if g === nothing
            @warn "bad-lines mask: no line confirmed near '$(e.label)' at " *
                  "$(round(e.λ, digits=3)) Å (core depth below $min_core_depth); " *
                  "entry skipped. Check the tabulated wavelength and its air/vacuum frame."
            continue
        end
        @views mask[g.i_lo:g.i_hi] .|= e.flag
        push!(regions, (label=e.label, λ_lo=Float64(wavs[g.i_lo]),
                        λ_hi=Float64(wavs[g.i_hi]), flag=e.flag,
                        n_pix=g.i_hi - g.i_lo + 1, core_depth=g.core_depth,
                        capped=g.capped))
    end
    return (mask=mask, regions=regions)
end

"""
    report_line_mask(regions; n_read, io=stdout) -> nothing

Print the applied mask regions, one row per entry, and flag the ones whose extent was set
by `max_extent` rather than by the depth walk.

`io` is a parameter rather than a bare `stdout` write so the output is capturable:
`redirect_stdout` cannot target an `IOBuffer`.
"""
function report_line_mask(regions; n_read::Int, io::IO=stdout)
    @printf(io, ">>> Bad-lines mask: %d of %d curated entries applied\n",
            length(regions), n_read)
    @printf(io, "%-14s %10s %10s %8s %8s %11s %7s\n",
            "label", "lam_lo", "lam_hi", "width", "n_pix", "core depth", "capped")
    for r in regions
        @printf(io, "%-14s %10.3f %10.3f %8.3f %8d %11.3f %7s\n",
                r.label, r.λ_lo, r.λ_hi, r.λ_hi - r.λ_lo, r.n_pix, r.core_depth,
                r.capped ? "yes" : "no")
    end
    n_capped = count(r -> r.capped, regions)
    if n_capped > 0
        @warn "bad-lines mask: $n_capped of $(length(regions)) regions were clamped at " *
              "max_extent, so the cap and not the line profile set their width."
    end
    return nothing
end
