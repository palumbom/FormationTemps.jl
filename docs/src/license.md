# License

```@eval
using Markdown, FormationTemps
license_file = joinpath(pkgdir(FormationTemps), "LICENSE")
if !isfile(license_file)
    license_file = joinpath(pkgdir(FormationTemps), "...", "LICENSE")
end
Markdown.parse_file(license_file)
```
