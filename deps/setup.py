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

# --- check OpenSSL before anything slow ---
# Julia 1.12 ships OpenSSL 3.5, and juliacall refuses a Julia whose OpenSSL is newer than
# Python's, so an older Python fails later with "'julia' compat entries have empty
# intersection". Older uv releases install Pythons linked against OpenSSL 3.0.
import ssl
if ssl.OPENSSL_VERSION_INFO < (3, 5):
    _warn(f"This Python is linked against {ssl.OPENSSL_VERSION}; Julia 1.12 from Python needs OpenSSL 3.5+.")
    _warn("Update uv and its Python, then rebuild the environment:")
    _warn("    uv self update && uv python upgrade 3.12 && rm -rf .venv && uv sync")
    _warn("or use another Python 3.12+ that qualifies: uv sync --python /path/to/python3")
    sys.exit(1)

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
KORG_UUID = "acafc109-a718-429c-b0e5-afd7f8c7ae46"

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
# Pin juliacall to Julia 1.12.x, overriding its ^1.10.3 default. The Julia package allows 1.12+,
# but this Python route is tested only on 1.12; widen the pin once newer releases are tested.
juliapkg.require_julia("~1.12")

if repo is not None:
    _info(f"Local clone detected at {_bold}{repo}{_reset}")
    _info("Installing FormationTemps.jl as a dev package...")
    juliapkg.add("FormationTemps", uuid, dev=True, path=str(repo))
else:
    _info("No local clone found -- installing from Julia package registry.")
    juliapkg.add("FormationTemps", FALLBACK_UUID, version="^3")

# Korg too: Julia loads only direct dependencies of the environment, and callers need
# `using Korg` for linelists and atmospheres. The version follows FormationTemps' compat.
juliapkg.add("Korg", KORG_UUID)

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
