"""One-time setup for using FormationTemps.jl from Python via juliacall."""
import os, shutil, tomllib
from pathlib import Path

# point juliapkg at the julia on PATH (the juliaup shim) so it does not
# probe juliaup internals where .app bundles confuse its binary detection.
# must be set BEFORE importing juliapkg, which reads the env var at import time.
julia_exe = shutil.which("julia")
if julia_exe is not None:
    os.environ.setdefault("PYTHON_JULIAPKG_EXE", julia_exe)

import juliapkg

# fallback UUID for registry installs (no Project.toml available)
FALLBACK_UUID = "03bcd87b-2230-4045-a5fa-95a5fcdd1ff8"

def _find_repo_root():
    """Return (root, uuid) if running inside a FormationTemps.jl clone, else (None, None)."""
    anchor = Path(__file__).resolve().parent
    for parent in (anchor, *anchor.parents):
        candidate = parent / "Project.toml"
        if candidate.exists():
            try:
                data = tomllib.loads(candidate.read_text())
                if data.get("name") == "FormationTemps":
                    return parent, data["uuid"]
            except (OSError, tomllib.TOMLDecodeError, KeyError):
                continue
    return None, None

repo, uuid = _find_repo_root()

# FormationTemps.jl requires Julia 1.12+; override juliacall's default (^1.10.3)
juliapkg.require_julia("~1.12")

if repo is not None:
    print(f"Local clone detected at {repo} -- installing as dev package.")
    juliapkg.add("FormationTemps", uuid, dev=True, path=str(repo))
else:
    print("No local clone found -- installing FormationTemps from Julia package registry.")
    juliapkg.add("FormationTemps", FALLBACK_UUID, version="^1")

# install a .pth file into the venv so that PYTHON_JULIAPKG_EXE is set
# on every Python startup in this environment (before juliapkg is imported).
# .pth files with `import` lines are executed by site.py at startup.
# this works around juliapkg not finding Julia inside macOS .app bundles.
if julia_exe is not None:
    import site
    site_dir = Path(site.getsitepackages()[0])
    pth = site_dir / "juliapkg_exe.pth"
    pth.write_text(
        "import os, shutil; "
        "_jl = shutil.which('julia'); "
        "os.environ.setdefault('PYTHON_JULIAPKG_EXE', _jl) if _jl else None; "
        "os.environ.setdefault('JULIA_NUM_THREADS', '1')\n"
    )
    print(f"Wrote {pth} (sets PYTHON_JULIAPKG_EXE and JULIA_NUM_THREADS on every Python startup in this venv)")

# resolve triggers the actual Julia Pkg install (this may take several minutes on first run)
print("Resolving Julia dependencies (this may take a few minutes the first time)...")
juliapkg.resolve()
print("Setup complete. You can now use: from juliacall import Main as jl")
