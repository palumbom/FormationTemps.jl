# Python Tutorial

!!! note
    FormationTemps.jl can be called from Python using
    [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).
    This requires a working Julia installation (v1.12+).

## Prerequisites

- **Julia 1.12+**: install from [julialang.org](https://julialang.org/downloads/) or via
  [juliaup](https://github.com/JuliaLang/juliaup).
- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) (recommended) or pip. We strongly advise against Conda.

## Installation

### From a Local Clone

If you have cloned the FormationTemps.jl repository:

```bash
cd FormationTemps.jl
uv sync
uv run python deps/setup.py
```

This installs FormationTemps.jl as a development package, meaning changes you make to
the Julia source code are immediately reflected without reinstalling.

### From the Julia Package Registry

If you want to use the released version of FormationTemps.jl without cloning
the repository:

```bash
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
juliapkg.add("FormationTemps", "03bcd87b-2230-4045-a5fa-95a5fcdd1ff8", version="^1")
juliapkg.resolve()
```

This pulls the latest compatible version of FormationTemps.jl from the
[Julia General registry](https://github.com/JuliaRegistries/General).

!!! warning
    `PYTHON_JULIAPKG_EXE` is required on macOS with juliaup (see Troubleshooting).
    `JULIA_NUM_THREADS=1` prevents GC crashes in the PythonCall bridge.
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
    juliacall requires that Julia and Python share a compatible OpenSSL version.
    Julia 1.12 needs OpenSSL 3.5+, but some Python builds (notably uv's standalone
    builds) ship OpenSSL 3.0, causing the resolver to reject Julia 1.12 entirely.
    Check your Python's OpenSSL version:
    ```python
    import ssl; print(ssl.OPENSSL_VERSION)
    ```
    If it reports < 3.5, you need a Python built against a newer OpenSSL. Recreate
    your venv pointing at one:
    ```bash
    rm -rf .venv
    uv sync --python /path/to/python3
    ```
    Where `/path/to/python3` is a Python with OpenSSL 3.5+. Options include:

    - **macOS**: Homebrew Python (`/opt/homebrew/bin/python3`) links against Homebrew's OpenSSL, which is typically up to date.
    - **Linux**: System Python from recent distros (Ubuntu 24.04+, Fedora 39+) ships OpenSSL 3.1+. On older distros, install a newer OpenSSL and rebuild Python via [pyenv](https://github.com/pyenv/pyenv).
    - **Windows**: The [python.org installer](https://www.python.org/downloads/) bundles its own OpenSSL. Recent releases (3.13.4+) include OpenSSL 3.5.

    uv's standalone Python builds (`uv python install`) currently bundle OpenSSL 3.0
    and will **not** work.

!!! details "Julia GC crash (SIGBUS/segfault) during long computations"
    Julia's multi-threaded garbage collector can conflict with PythonCall's runtime
    bridge, causing hard crashes. The setup script sets `JULIA_NUM_THREADS=1` to
    avoid this. For registry installs, set it in your shell:
    ```bash
    export JULIA_NUM_THREADS=1
    ```
    This disables Julia's thread parallelism but avoids the GC contention. If you
    need multi-threaded Julia, consider running the computation in pure Julia and
    loading the results in Python.

## Tips

!!! tip "Performance"
    The first `import juliacall` is slow — Julia compiles code on first use.
    Subsequent calls in the same session are fast. Precompilation happens once
    per environment.

!!! tip "GPU support"
    Pass `use_gpu=True` to `calc_formation_temp` if you have a CUDA-capable GPU
    configured with Julia's [CUDA.jl](https://cuda.juliagpu.org/stable/).

!!! tip "Type mapping between Python and Julia"
    - **Keyword arguments** map directly to Julia kwargs:
      `FT.calc_formation_temp(star, linelist, use_gpu=False, convolve=True, u1=0.43, u2=0.31)`.
    - **Arrays**: use `numpy.asarray(result.wavs)` to convert Julia arrays to numpy.
      This is zero-copy for contiguous `Float64` arrays.
    - **Booleans**: Python `True`/`False` map to Julia `true`/`false` automatically.
    - **Indexing**: juliacall uses 0-based indexing from the Python side. When slicing
      Julia arrays from Python, indices are shifted by one relative to Julia. For exact
      Julia-equivalent slicing, use `jl.seval("array[1:10]")`.

!!! tip "Matplotlib backend"
    Interactive backends (e.g. `macosx`, `Qt5Agg`) can crash under juliacall due to
    event loop conflicts. Use `matplotlib.use("Agg")` before importing `pyplot` and
    save figures to files instead of calling `plt.show()`.
