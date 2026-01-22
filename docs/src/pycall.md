# Python Tutorial

!!! warning
    Calling FormationTemps.jl from Python is currently somewhat fragile and a work in progress. 


## Installation 
First, some Python dependencies will need to be installed from the included ```pyproject.toml```. If you have the [```uv``` tool](https://docs.astral.sh/uv/) installed, you can simply run:

```bash
cd FormationTemps.jl
uv sync
```

Before FormationTemps.jl is used from Python for the first time, a few things will need to be configured in Python:

```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "setup_pycall.py"), String)
Markdown.parse("```python\n" * code * "\n```")
``` 

This script can also be run by ```uv``` via:

```bash
uv run FormationTemps.jl/docs/src/examples.setup_pycall.py
```

## Basic Usage

Once installed, FormationTemps.jl can be loaded in Python via:

```python
from juliacall import Main as jl
jl.seval("using FormationTemps")
FT = jl.FormationTemps
```

A simple formation temperature spectrum (following the Julia example shown in [the basic tutorial](@ref "Basic Tutorial")) can then be calculated like:

```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "simple.py"), String)
Markdown.parse("```python\n" * code * "\n```")
``` 
