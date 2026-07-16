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
