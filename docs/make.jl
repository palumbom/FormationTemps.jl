# add relative load path
push!(LOAD_PATH,"../src/")

using Documenter, FormationTemps

# sync the readme and the landing page
docs_base = basename(pwd()) == "docs" ? "." : "./docs"
readme_path = joinpath(docs_base, "..", "README.md")
target_path = joinpath(docs_base, "src", "index.md")
readme_text = read(readme_path, String)
readme_text = replace(readme_text, "./docs/src/" => "./")
readme_text = replace(readme_text, "<img src=\"docs/src/assets/logo.png\" height=\"48\">" => "")

function replace_admonition(text, github_style, documenter_style)
    github_marker = startswith(github_style, "[!") ? github_style : "[!$(github_style)]"
    escaped_marker = replace(github_marker, r"([\\.^$|?*+()[{])" => s"\\\1")
    pattern = Regex("(?m)^> $(escaped_marker)\\s*\\n(?:> ?.*\\n?)+")
    replace(text, pattern => m -> begin
        lines = split(m, '\n'; keepempty=true)
        body = String[]
        for line in lines[2:end]
            if startswith(line, ">")
                stripped = replace(line, r"^>\s?" => "")
                push!(body, stripped)
            elseif !isempty(line)
                push!(body, line)
            end
        end
        while !isempty(body) && isempty(body[end])
            pop!(body)
        end
        if isempty(body)
            return documenter_style
        end
        indented = join(["    " * l for l in body], "\n")
        return documenter_style * "\n" * indented
    end)
end

readme_text = replace_admonition(readme_text, "WARNING", "!!! warning")
readme_text = replace_admonition(readme_text, "CAUTION", "!!! danger")
# avoid duplicate "Parallelization" slug (parallelization.md already owns it)
readme_text = replace(readme_text, "## Parallelization" => "## Parallelization Overview")
write(target_path, readme_text)

# Check the static figures both ways: a reference with no file publishes a broken image,
# and a file with no reference means a figure was regenerated but never wired into a page
# (or was superseded and left behind). Runs after the README sync so figures referenced
# only from the landing page count as used.
function check_static_assets(src_dir)
    static_dir = joinpath(src_dir, "static")
    isdir(static_dir) || return
    on_disk = Set(readdir(static_dir))

    referenced = Set{String}()
    missing_refs = Tuple{String,String}[]
    for page in filter(f -> endswith(f, ".md"), readdir(src_dir))
        text = read(joinpath(src_dir, page), String)
        for m in eachmatch(r"static/([A-Za-z0-9_.\-]+)", text)
            asset = m.captures[1]
            push!(referenced, asset)
            asset in on_disk || push!(missing_refs, (page, asset))
        end
    end

    orphans = sort(collect(setdiff(on_disk, referenced)))
    isempty(missing_refs) && isempty(orphans) && return

    msg = "docs/src/static is out of sync with the pages:"
    for (page, asset) in missing_refs
        msg *= "\n  missing file: $page references static/$asset"
    end
    for asset in orphans
        msg *= "\n  unreferenced: static/$asset is on disk but no page uses it"
    end
    error(msg * "\n(reference the file from a page, or delete it)")
end

check_static_assets(joinpath(docs_base, "src"))

# set pages
Introduction = "Quickstart" => "index.md"
License = "License" => "license.md"
Index = "Full Index" => "longlist.md"
Guides = "Guides" => ["Basic Tutorial" => "tutorial.md",
                      "Contribution Functions" => "cont_func.md",
                      "Integration Methods" => "methods.md",
                      "Broadening & Convolutions" => "convolutions.md",
                      "Large Linelists" => "chunked.md",
                      "Parallelization" => "parallelization.md",
                      "Python Tutorial" => "python.md"]
Internals = "Public Functions" => "internals.md"

pages = [Introduction, Guides, Internals, Index, License]

# makedocs
makedocs(;
    modules=[FormationTemps],
    authors="Michael Palumbo",
    # repo="https://github.com/palumbom/GRASS/blob/{commit}{path}#{line}",
    sitename="FormationTemps.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://palumbom.github.io/FormationTemps.jl",
        assets=String[],
    ),
    pages=pages,
)

# deploydocs
deploydocs(;
    devbranch="main",
    repo="github.com/palumbom/FormationTemps.jl.git",
    push_preview=true
)
