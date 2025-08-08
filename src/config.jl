# Pkg manager
using Pkg

# set absolute path to solar data
const moddir = abspath(joinpath(@__DIR__, ".."))
const datdir = abspath(joinpath(moddir, "data/"))
const plotdir = abspath(joinpath(moddir, "plots/"))

@assert isdir(moddir)

if !isdir(datdir); mkdir(datdir); end;
if !isdir(plotdir); mkdir(plotdir); end;

@assert isdir(datdir)
@assert isdir(plotdir)