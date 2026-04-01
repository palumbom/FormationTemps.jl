# set absolute path to solar data
const moddir = abspath(joinpath(@__DIR__, ".."))
const datdir = abspath(joinpath(moddir, "data"))
const plotdir = abspath(joinpath(moddir, "figures"))

@assert isdir(moddir)
mkpath(datdir)
mkpath(plotdir)