# Unit tests for the curated bad-lines mask helpers in scripts/bad_lines_mask.jl. The
# helpers are script-local rather than package API, but they are pure functions that a
# production run cannot cheaply re-verify, so they are checked here. Skips cleanly when
# the script is absent.
const bad_lines_script = joinpath(@__DIR__, "..", "scripts", "bad_lines_mask.jl")

if !isfile(bad_lines_script)
    @info "skipping bad-lines mask tests: $bad_lines_script not found"
else
    include(bad_lines_script)

    # write `body` to a temp file and return its path
    function tmp_bad_lines(body)
        path, io = mktemp()
        write(io, body)
        close(io)
        return path
    end

    @testset "bad-lines mask" begin
        @testset "read_bad_lines parses and maps flags" begin
            body = "# a comment line, skipped\n" *
                   "lambda_air,species,label,flag,max_extent,citation\n" *
                   "5000.0,Fe I,line A,nlte,,\n" *
                   "6000.0,Ca II,line B,chromo,10.0,Someone et al. 2020\n"
            e = read_bad_lines(tmp_bad_lines(body); vacuum=false, max_extent_default=5.0)

            @test length(e) == 2
            @test e[1].label == "line A"
            @test e[1].flag == MASK_NLTE
            @test e[2].flag == MASK_CHROMO

            # species parses to a Korg.Species, so it compares against linelist entries
            @test e[1].species == Korg.Species("Fe I")
            @test e[2].species == Korg.Species("Ca II")

            # a blank max_extent inherits the default; a present one wins
            @test e[1].max_extent == 5.0
            @test e[2].max_extent == 10.0

            # vacuum=false leaves the tabulated air wavelengths untouched
            @test e[1].λ == 5000.0
        end

        @testset "read_bad_lines converts to vacuum on request" begin
            body = "lambda_air,species,label,flag,max_extent\n5000.0,Fe I,line A,nlte,\n"
            path = tmp_bad_lines(body)
            ev = read_bad_lines(path; vacuum=true, max_extent_default=5.0)

            @test ev[1].λ ≈ Korg.air_to_vacuum(5000.0)
            @test ev[1].λ > 5000.0            # air → vacuum shifts red, by ~1.4 Å here
        end

        @testset "read_bad_lines rejects malformed files" begin
            hdr = "lambda_air,species,label,flag,max_extent\n"
            bad(body) = read_bad_lines(tmp_bad_lines(body); vacuum=false, max_extent_default=5.0)

            # unrecognized flag name
            @test_throws AssertionError bad(hdr * "5000.0,Fe I,a,bogus,\n")
            # missing a required column
            @test_throws AssertionError bad("lambda_air,flag\n5000.0,nlte\n")
            # missing the species column
            @test_throws AssertionError bad("lambda_air,label,flag\n5000.0,a,nlte\n")
            # not strictly ascending — catches transposed rows
            @test_throws AssertionError bad(hdr * "6000.0,Fe I,a,nlte,\n5000.0,Fe I,b,nlte,\n")
            # duplicate wavelength
            @test_throws AssertionError bad(hdr * "5000.0,Fe I,a,nlte,\n5000.0,Fe I,b,nlte,\n")
            # duplicate label — labels key the provenance table
            @test_throws AssertionError bad(hdr * "5000.0,Fe I,a,nlte,\n6000.0,Fe I,a,nlte,\n")
            # non-positive cap
            @test_throws AssertionError bad(hdr * "5000.0,Fe I,a,nlte,-1.0\n")
            # unparseable species code — raised by Korg, not by an assertion
            @test_throws ArgumentError bad(hdr * "5000.0,Zz IV V,a,nlte,\n")
        end

        @testset "verify_species_present keeps only species the linelist has" begin
            # a two-species linelist: Fe I at 5000, Ti II at 6000
            line_wavs    = [5000.02, 6000.01]
            line_species = [Korg.Species("Fe I"), Korg.Species("Ti II")]

            entries = [(λ=5000.0, species=Korg.Species("Fe I"),  label="keep",  flag=MASK_NLTE, max_extent=5.0),
                       (λ=6000.0, species=Korg.Species("Mg I"),  label="drop",  flag=MASK_NLTE, max_extent=5.0)]

            kept = @test_logs (:warn,) match_mode=:any verify_species_present(
                entries, line_wavs, line_species; n_sigma_halo=5.0, v_broad=4085.0)

            # Mg I is dropped even though Ti II sits 0.01 Å away and would have been the
            # seed's flux minimum — that substitution is the bug this check exists to stop
            @test [e.label for e in kept] == ["keep"]
        end

        @testset "verify_species_present exempts hydrogen" begin
            # Korg synthesizes Balmer lines from its own data, so H I is never in the linelist
            line_wavs    = [5000.02]
            line_species = [Korg.Species("Fe I")]
            entries = [(λ=6562.8, species=Korg.Species("H I"), label="Ha",
                        flag=MASK_NLTE, max_extent=25.0)]

            kept = @test_logs verify_species_present(entries, line_wavs, line_species;
                                                     n_sigma_halo=5.0, v_broad=4085.0)
            @test [e.label for e in kept] == ["Ha"]
        end

        @testset "verify_species_present respects the halo width" begin
            entries = [(λ=5000.0, species=Korg.Species("Fe I"), label="a",
                        flag=MASK_NLTE, max_extent=5.0)]
            # halo at 5000 Å for v_broad = 4085 m/s and 5σ is ~0.34 Å
            inside  = verify_species_present(entries, [5000.30], [Korg.Species("Fe I")];
                                             n_sigma_halo=5.0, v_broad=4085.0)
            @test length(inside) == 1

            outside = @test_logs (:warn,) match_mode=:any verify_species_present(
                entries, [5000.50], [Korg.Species("Fe I")];
                n_sigma_halo=5.0, v_broad=4085.0)
            @test isempty(outside)
        end

        @testset "verify_species_present rejects an unsorted linelist" begin
            entries = [(λ=5000.0, species=Korg.Species("Fe I"), label="a",
                        flag=MASK_NLTE, max_extent=5.0)]
            @test_throws AssertionError verify_species_present(entries,
                [6000.0, 5000.0], [Korg.Species("Fe I"), Korg.Species("Fe I")];
                n_sigma_halo=5.0, v_broad=4085.0)
        end

        @testset "the committed list loads and is self-consistent" begin
            committed = joinpath(FT.datdir, "bad_lines.csv")
            if !isfile(committed)
                @info "skipping: $committed not found"
            else
                e = read_bad_lines(committed; vacuum=true,
                                   max_extent_default=MAX_EXTENT_DEFAULT)
                @test !isempty(e)
                @test issorted([x.λ for x in e])
                @test all(x -> x.flag in values(FLAG_CODES), e)
                @test all(x -> x.max_extent > 0, e)
                # every species code parses; read_bad_lines would have thrown otherwise
                @test all(x -> x.species isa Korg.Species, e)
            end
        end

        # A triangular absorption line: depth 0 outside ±hw, rising linearly to `depth`
        # at the center. Chosen over a Gaussian so the threshold crossing is exact
        # arithmetic and the expected growth extent can be written down.
        function triangle_spectrum(wavs, centers; depth=0.8, hw=0.5)
            flux = ones(length(wavs))
            for λc in centers, (i, λ) in enumerate(wavs)
                flux[i] -= depth * max(0.0, 1 - abs(λ - λc) / hw)
            end
            return flux
        end

        @testset "grow_line_region walks out to the depth threshold" begin
            wavs = collect(5000.0:0.01:5010.0)
            λc = 5005.0
            flux = triangle_spectrum(wavs, (λc,))

            g = grow_line_region(wavs, flux, λc; halo=0.05, depth_thresh=0.02,
                                 min_core_depth=0.05, max_extent=5.0)
            @test g !== nothing
            @test wavs[g.i0] ≈ λc
            @test g.core_depth ≈ 0.8
            @test !g.capped

            # depth(Δ) = 0.8(1 - |Δ|/0.5) exceeds 0.02 for |Δ| < 0.4875, so on a 0.01 Å
            # grid the last included pixel is at ±0.48
            @test wavs[g.i_lo] ≈ λc - 0.48
            @test wavs[g.i_hi] ≈ λc + 0.48
        end

        @testset "grow_line_region clamps at max_extent and reports it" begin
            wavs = collect(5000.0:0.01:5010.0)
            λc = 5005.0
            flux = triangle_spectrum(wavs, (λc,))

            g = grow_line_region(wavs, flux, λc; halo=0.05, depth_thresh=0.02,
                                 min_core_depth=0.05, max_extent=0.2)
            @test g.capped
            @test wavs[g.i_lo] ≈ λc - 0.2
            @test wavs[g.i_hi] ≈ λc + 0.2
        end

        @testset "grow_line_region floors at the halo" begin
            wavs = collect(5000.0:0.01:5010.0)
            λc = 5005.0
            # one deep pixel, no wings: growth stops immediately, the halo must still hold
            flux = ones(length(wavs))
            flux[findfirst(≈(λc), wavs)] = 0.5

            g = grow_line_region(wavs, flux, λc; halo=0.1, depth_thresh=0.02,
                                 min_core_depth=0.05, max_extent=5.0)
            @test wavs[g.i_lo] ≈ λc - 0.1
            @test wavs[g.i_hi] ≈ λc + 0.1
        end

        @testset "grow_line_region rejects seeds it cannot confirm" begin
            wavs = collect(5000.0:0.01:5010.0)
            λc = 5005.0
            flux = triangle_spectrum(wavs, (λc,))

            # too shallow to be the intended line
            @test grow_line_region(wavs, flux, λc; halo=0.05, depth_thresh=0.02,
                                   min_core_depth=0.9, max_extent=5.0) === nothing
            # seed off the grid entirely
            @test grow_line_region(wavs, flux, 4000.0; halo=0.05, depth_thresh=0.02,
                                   min_core_depth=0.05, max_extent=5.0) === nothing
            # seed on the grid but in a flat region
            @test grow_line_region(wavs, flux, 5001.0; halo=0.05, depth_thresh=0.02,
                                   min_core_depth=0.05, max_extent=5.0) === nothing
        end

        @testset "build_line_mask ORs distinct bits per reason" begin
            wavs = collect(5000.0:0.01:5010.0)
            flux = triangle_spectrum(wavs, (5002.0, 5008.0))
            entries = [(λ=5002.0, label="A", flag=MASK_NLTE,   max_extent=5.0),
                       (λ=5008.0, label="B", flag=MASK_CHROMO, max_extent=5.0)]

            out = build_line_mask(wavs, flux, entries; n_sigma_halo=5.0,
                                  depth_thresh=0.02, min_core_depth=0.05,
                                  v_broad=4085.0)

            @test eltype(out.mask) == UInt8
            @test length(out.mask) == length(wavs)
            @test length(out.regions) == 2

            @test out.mask[findfirst(≈(5002.0), wavs)] == MASK_NLTE
            @test out.mask[findfirst(≈(5008.0), wavs)] == MASK_CHROMO
            # the two regions do not reach each other, so no pixel carries both bits
            @test !any(out.mask .== (MASK_NLTE | MASK_CHROMO))
            # nothing masked well away from either line
            @test all(out.mask[wavs .< 5001.0] .== 0x00)

            # regions carry provenance in the order the entries were given
            @test [r.label for r in out.regions] == ["A", "B"]
            @test out.regions[1].λ_lo < 5002.0 < out.regions[1].λ_hi
            @test out.regions[1].n_pix == count(!=(0x00), out.mask) ÷ 2
        end

        @testset "build_line_mask warns and skips an unconfirmed entry" begin
            wavs = collect(5000.0:0.01:5010.0)
            flux = triangle_spectrum(wavs, (5002.0,))
            # 5006.0 is flat here, so the seed cannot be confirmed
            entries = [(λ=5006.0, label="C", flag=MASK_NLTE, max_extent=5.0)]

            out = @test_logs (:warn,) match_mode=:any build_line_mask(
                wavs, flux, entries; n_sigma_halo=5.0, depth_thresh=0.02,
                min_core_depth=0.05, v_broad=4085.0)

            @test isempty(out.regions)
            @test all(out.mask .== 0x00)
        end

        @testset "build_line_mask rejects a cap below the halo" begin
            wavs = collect(5000.0:0.01:5010.0)
            flux = triangle_spectrum(wavs, (5002.0,))
            # halo at 5002 Å with v_broad = 4085 m/s and 5σ is ~0.34 Å, above this cap
            entries = [(λ=5002.0, label="A", flag=MASK_NLTE, max_extent=0.1)]

            @test_throws AssertionError build_line_mask(wavs, flux, entries;
                n_sigma_halo=5.0, depth_thresh=0.02, min_core_depth=0.05,
                v_broad=4085.0)
        end

        @testset "build_line_mask skips out-of-window entries silently" begin
            wavs = collect(5000.0:0.01:5010.0)
            flux = triangle_spectrum(wavs, (5002.0,))
            # the chunked caller passes one narrow window at a time, so most curated lines
            # fall outside it; warning on those would flood a production run
            entries = [(λ=3000.0, label="far blue", flag=MASK_NLTE,   max_extent=5.0),
                       (λ=9000.0, label="far red",  flag=MASK_CHROMO, max_extent=5.0)]

            out = @test_logs build_line_mask(wavs, flux, entries; n_sigma_halo=5.0,
                depth_thresh=0.02, min_core_depth=0.05, v_broad=4085.0)

            @test isempty(out.regions)
            @test all(out.mask .== 0x00)
        end

        @testset "report_line_mask prints one row per region" begin
            regions = [(label="A", λ_lo=5001.5, λ_hi=5002.5, flag=MASK_NLTE,
                        n_pix=101, core_depth=0.8, thresh=0.4, capped=false)]
            out = sprint(io -> report_line_mask(regions; n_read=3, io=io))
            @test occursin("A", out)
            @test occursin("1 of 3", out)   # applied count reported against rows read
        end
    end
end
