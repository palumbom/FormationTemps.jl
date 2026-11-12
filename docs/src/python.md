# Python Tutorial

FormationTemps.jl can be called from Python using [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). This requires a working Julia installation (v1.12+) and Python 3.12+ with [uv](https://docs.astral.sh/uv/) (recommended) or pip. We strongly advise against Conda.

## Prerequisites

- Julia 1.12+: install from [julialang.org](https://julialang.org/downloads/) or via [juliaup](https://github.com/JuliaLang/juliaup).
  The install steps below ask juliapkg for Julia 1.12; if the `julia` on your PATH is newer,
  juliapkg prints a warning and uses it anyway, which works with 1.13.
- Python 3.12+ with [uv](https://docs.astral.sh/uv/) (recommended) or pip.
- A LaTeX installation, to run the example below as written: it uses the package's plot style,
  which renders text with LaTeX.

## Installing the Python interface

### From a Local Clone

If you have cloned the FormationTemps.jl repository:

```bash
cd FormationTemps.jl
uv sync
uv run python -c "import ssl; print(ssl.OPENSSL_VERSION)"   # must be 3.5 or newer
uv run python deps/setup.py
```

!!! note "If OpenSSL is older than 3.5"
    Julia 1.12 can only be loaded into a Python linked against OpenSSL 3.5 or newer. Older uv
    releases install Pythons with OpenSSL 3.0; update uv and its Python, then rebuild the
    environment:
    ```bash
    uv self update            # or update uv the way you installed it
    uv python upgrade 3.12
    rm -rf .venv && uv sync
    ```
    Any other Python 3.12+ that qualifies also works: `uv sync --python /path/to/python3`.
    `deps/setup.py` checks this and stops with the same advice.

This installs FormationTemps.jl as a development package, meaning changes you make to
the Julia source code are immediately reflected without reinstalling.

### From the Julia Package Registry

If you want to use the released version of FormationTemps.jl without cloning
the repository:

```bash
python -c "import ssl; print(ssl.OPENSSL_VERSION)"   # must be 3.5 or newer; see the note above
pip install juliacall juliapkg numpy matplotlib
```

Then set the following environment variables in your shell profile (e.g.
`~/.bashrc`, `~/.zshrc`) so they persist across sessions:

```bash
export PYTHON_JULIAPKG_EXE=$(which julia)
export JULIA_NUM_THREADS=1
```

Finally, run the following once in Python to register the Julia dependency:

```python
import juliapkg
juliapkg.require_julia("~1.12")
juliapkg.add("FormationTemps", "03bcd87b-2230-4045-a5fa-95a5fcdd1ff8", version="^3")
juliapkg.add("Korg", "acafc109-a718-429c-b0e5-afd7f8c7ae46")
juliapkg.resolve()
```

This pulls the latest compatible versions of FormationTemps.jl and Korg.jl from the
[Julia General registry](https://github.com/JuliaRegistries/General). Korg is added explicitly
because the examples call it directly, and Julia only loads packages the environment lists.

!!! warning
    `PYTHON_JULIAPKG_EXE` is required on macOS with juliaup (see Troubleshooting below).
    `JULIA_NUM_THREADS=1` prevents GC crashes in the PythonCall bridge — CPU
    multithreading is not available when calling from Python.
    The local clone setup script (`deps/setup.py`) handles both automatically.

## Basic Usage

Once installed, FormationTemps.jl can be loaded in Python via:

```python
from juliacall import Main as jl
jl.seval("using Korg")
jl.seval("using FormationTemps")
FT = jl.FormationTemps
Korg = jl.Korg
```

A complete example computing a formation temperature spectrum (following the
[Basic Tutorial](@ref "Basic Tutorial")) is shown below:

```@eval
using Markdown
code = read(joinpath(pwd(), "examples", "simple.py"), String)
Markdown.parse("```python\n" * code * "\n```")
```

## Troubleshooting

!!! details "\"not a valid Julia executable\" on macOS"
    juliaup installs Julia 1.12+ as `.app` bundles that `juliapkg` cannot find.
    The setup script handles this automatically for local clones. For registry
    installs, set `PYTHON_JULIAPKG_EXE` before importing juliacall:
    ```bash
    export PYTHON_JULIAPKG_EXE=$(which julia)
    ```

!!! details "\"julia compat entries have empty intersection\" (OpenSSL conflict)"
    Python is linked against an OpenSSL older than 3.5. Check with
    `python -c "import ssl; print(ssl.OPENSSL_VERSION)"` and see
    the note under [From a Local Clone](@ref) above.

!!! details "Julia GC crash (SIGBUS/segfault) during long computations"
    Julia's multi-threaded garbage collector can conflict with PythonCall's runtime
    bridge, causing hard crashes. The setup script sets `JULIA_NUM_THREADS=1` to
    avoid this. For registry installs, set it in your shell:
    ```bash
    export JULIA_NUM_THREADS=1
    ```
    This disables Julia's CPU multithreading but avoids the GC contention. GPU
    acceleration is unaffected — pass `use_gpu=True` to benefit from GPU parallelism
    even with a single Julia thread. If CPU multithreading is needed, run the
    computation in pure Julia and load the results in Python. See
    [Parallelization](parallelization.md) for details.

## Tips

!!! tip "Performance"
    The first `import juliacall` is slow — Julia compiles code on first use.
    Subsequent calls in the same session are fast. Precompilation happens once
    per environment.

!!! tip "Getting speed without threads"
    CPU multithreading is unavailable from Python (see above), which leaves two options.
    Reach for `method=jl.Symbol("quadrature")` first: it solves the radiative transfer once
    per μ node instead of once per surface tile, so it recovers most of the cost of the
    missing threads without changing the physics — see [Integration Methods](@ref). Beyond
    that, pass `use_gpu=True` if you have a CUDA-capable GPU configured with Julia's
    [CUDA.jl](https://cuda.juliagpu.org/stable/); the two compose.

!!! tip "Type mapping between Python and Julia"
    Keyword arguments map directly to Julia kwargs (e.g.,
    `FT.calc_formation_temp(star, linelist, use_gpu=False)`).
    One exception: `method` takes a Julia `Symbol`, and a Python `str` does not convert to
    one — pass `method=jl.Symbol("quadrature")` rather than `method="quadrature"`.
    Julia arrays can be converted to numpy with `numpy.asarray(result.wavs)`,
    which is zero-copy for contiguous `Float64` arrays. Python `True`/`False`
    map to Julia `true`/`false` automatically. Note that juliacall uses 0-based
    indexing from the Python side.

!!! tip "Matplotlib backend"
    Interactive backends (e.g., `macosx`, `Qt5Agg`) can crash under juliacall due to
    event loop conflicts. Use `matplotlib.use("Agg")` before importing `pyplot` and
    save figures to files instead of calling `plt.show()`.
