# add relative load path
push!(LOAD_PATH,"../src/")

using Documenter, FormationTemps

# set pages
Introduction = "Quickstart" => "quickstart.md"
License = "License" => "license.md"
Index = "Index" => "longlist.md"
pages = [Introduction, License, Index]

# makdocs
makedocs(;
    modules=[FormationTemps],
    authors="Michael Palumbo",
    # repo="https://github.com/palumbom/GRASS/blob/{commit}{path}#{line}",
    sitename="FormationTemps.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://palumbom.github.io/FormationTemps",
        assets=String[],
    ),
    pages=pages,
)

deploydocs(;
    repo="github.com/palumbom/FormationTemps",
)
