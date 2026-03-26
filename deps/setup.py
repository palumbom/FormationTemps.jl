"""One-time setup for using FormationTemps.jl from Python via juliacall."""
import os, sys, shutil, tomllib
from pathlib import Path

# --- formatting helpers (ANSI colors, no dependencies) ---
_bold = "\033[1m"
_green = "\033[32m"
_yellow = "\033[33m"
_cyan = "\033[36m"
_reset = "\033[0m"

def _info(msg):
    print(f"{_cyan}{_bold}[setup]{_reset} {msg}", flush=True)

def _warn(msg):
    print(f"{_yellow}{_bold}[setup]{_reset} {msg}", flush=True)

def _ok(msg):
    print(f"{_green}{_bold}[setup]{_reset} {msg}", flush=True)

# --- locate julia before importing juliapkg (reads env at import time) ---
# on macOS with juliaup, Julia 1.12+ installs as .app bundles that juliapkg
# cannot find. pointing at the juliaup shim on PATH works around this.
julia_exe = shutil.which("julia")
if julia_exe is not None:
    os.environ.setdefault("PYTHON_JULIAPKG_EXE", julia_exe)
else:
    _warn("julia not found on PATH. Install Julia 1.12+ and ensure it is on your PATH.")
    sys.exit(1)

import juliapkg

# --- detect local clone vs registry install ---
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

# --- configure juliapkg ---
# FormationTemps.jl requires Julia 1.12+; override juliacall's default (^1.10.3)
juliapkg.require_julia("~1.12")

if repo is not None:
    _info(f"Local clone detected at {_bold}{repo}{_reset}")
    _info("Installing FormationTemps.jl as a dev package...")
    juliapkg.add("FormationTemps", uuid, dev=True, path=str(repo))
else:
    _info("No local clone found -- installing from Julia package registry.")
    juliapkg.add("FormationTemps", FALLBACK_UUID, version="^1")

# --- write .pth file for persistent env vars ---
# .pth files with `import` lines are executed by site.py on every Python startup.
# this sets PYTHON_JULIAPKG_EXE (macOS .app bundle workaround) and
# JULIA_NUM_THREADS=1 (avoids GC crash in PythonCall bridge).
import site
site_dir = Path(site.getsitepackages()[0])
pth = site_dir / "juliapkg_exe.pth"
pth.write_text(
    "import os, shutil; "
    "_jl = shutil.which('julia'); "
    "os.environ.setdefault('PYTHON_JULIAPKG_EXE', _jl) if _jl else None; "
    "os.environ.setdefault('JULIA_NUM_THREADS', '1')\n"
)
_info(f"Wrote {pth}")

# --- resolve Julia dependencies ---
_info("Resolving Julia dependencies (this may take a few minutes the first time)...")
juliapkg.resolve()
_ok("Setup complete!")
_ok("Try running: `uv run python docs/src/examples/simple.py`")
