# add relative load path
push!(LOAD_PATH,"../src/")

using Documenter, FormationTemps

# sync the readme and the landing page
docs_base = basename(pwd()) == "docs" ? "." : "./docs"
readme_path = joinpath(docs_base, "..", "README.md")
target_path = joinpath(docs_base, "src", "index.md")
readme_text = read(readme_path, String)
readme_text = replace(readme_text, "./docs/src/" => "./")
write(target_path, readme_text)

# set pages
Introduction = "Quickstart" => "index.md"
License = "License" => "license.md"
Index = "Full Index" => "longlist.md"
Guides = "Guides" => ["Basic Tutorial" => "tutorial.md",
                      "Contribution Functions" => "cont_func.md",
                      "Python Tutorial" => "pycall.md"]
Internals = "Public Functions" => "internals.md"

pages = [Introduction, Guides, Internals, License, Index]

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
