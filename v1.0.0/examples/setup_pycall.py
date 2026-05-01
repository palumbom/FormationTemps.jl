import os
import shutil
import subprocess
import sys
from pathlib import Path
import tomllib
import textwrap
import pdb

root = Path(__file__).resolve().parents[3]
project_toml = root / "Project.toml"
uuid = tomllib.loads(project_toml.read_text())["uuid"]

julia = os.environ.get("JULIA_EXE") or shutil.which("julia")
if julia is None:
    raise RuntimeError("Julia executable not found in PATH. Set JULIA_EXE to override.")

julia_env = Path(os.environ.get("PYTHON_JULIAPKG_PROJECT", root / ".julia_env"))
env_project = julia_env / "Project.toml"
needs_install = True
if env_project.exists():
    env_deps = tomllib.loads(env_project.read_text()).get("deps", {})
    needs_install = not {"FormationTemps", "PythonCall"}.issubset(env_deps.keys())

if needs_install:
    julia_env.mkdir(parents=True, exist_ok=True)
    script = textwrap.dedent(
        f"""
        import Pkg
        Pkg.activate(raw"{julia_env}")
        Pkg.add(Pkg.PackageSpec(name="PythonCall", uuid="6099a3de-0909-46bc-b1f4-468b9a2dfc0d"))
        Pkg.add(Pkg.PackageSpec(name="FormationTemps", uuid="{uuid}", url=raw"https://github.com/palumbom/FormationTemps.jl.git"))
        Pkg.precompile()
        """
    ).strip()
    env = os.environ.copy()
    env.setdefault("JULIA_PYTHONCALL_EXE", sys.executable or "")
    subprocess.run(
        [julia, "--startup-file=no", "--project=" + str(julia_env), "-e", script],
        check=True,
        env=env,
    )

# Avoid juliapkg rewriting the environment (and its OpenSSL_jll compat).
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(julia_env))
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")