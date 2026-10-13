"""Isolated virtualenvs for official harnesses.

Harnesses pin conflicting dependencies (e.g. inspect-ai needs ``openai>=3.4``
while LiteLLM pins ``openai<3``), so each one lives in its own ``uv`` venv under
``harnesses/<name>/.venv`` with exact pinned versions recorded in
``harnesses/<name>/manifest.json``. ``ensure_venv`` is idempotent: it rebuilds
only when the pin set changes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("BENCHLAB_HARNESS_DIR", "harnesses"))


class HarnessInstallError(RuntimeError):
    pass


def harness_root(name: str, root: Path | None = None) -> Path:
    return (root or DEFAULT_ROOT) / name


def _uv() -> str:
    exe = shutil.which("uv")
    if not exe:
        raise HarnessInstallError("`uv` is required to create harness environments")
    return exe


def ensure_venv(
    name: str,
    packages: list[str],
    python: str = "3.12",
    root: Path | None = None,
    extra_index_args: list[str] | None = None,
) -> Path:
    """Create/refresh ``harnesses/<name>/.venv`` with exactly ``packages`` (pip
    specifiers, e.g. ``"inspect-ai==0.3.279"``). Returns the venv's ``bin`` dir."""
    base = harness_root(name, root)
    venv = base / ".venv"
    manifest = base / "manifest.json"
    want = {"python": python, "packages": sorted(packages)}
    if venv.exists() and manifest.exists():
        try:
            have = json.loads(manifest.read_text())
            if have.get("python") == python and have.get("packages") == want["packages"]:
                return venv / "bin"
        except json.JSONDecodeError:
            pass
    base.mkdir(parents=True, exist_ok=True)
    uv = _uv()
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    try:
        subprocess.run(
            [uv, "venv", "--python", python, "--quiet", "--allow-existing", str(venv)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--quiet",
                "--python",
                str(venv / "bin" / "python"),
                *(extra_index_args or []),
                *packages,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise HarnessInstallError(
            f"failed to build harness env {name!r}: {e.stderr.strip()[-2000:] or e.stdout.strip()[-2000:]}"
        ) from e
    installed = frozen_versions(venv / "bin")
    manifest.write_text(
        json.dumps({**want, "installed": installed, "venv": str(venv)}, indent=2, sort_keys=True)
    )
    return venv / "bin"


def frozen_versions(bin_dir: Path, names: list[str] | None = None) -> dict[str, str]:
    """``{dist: version}`` for the venv (optionally filtered to ``names``)."""
    out = subprocess.run(
        [
            str(bin_dir / "python"),
            "-c",
            "import importlib.metadata as m, json;"
            "print(json.dumps({d.metadata['Name'].lower(): d.version for d in m.distributions()}))",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return {}
    allv = json.loads(out.stdout or "{}")
    if names:
        return {n: allv.get(n.lower().replace("_", "-"), allv.get(n.lower())) for n in names}
    return allv


def run_python(
    bin_dir: Path, code: str, env: dict[str, str] | None = None, timeout: float = 600
) -> str:
    """Run a snippet with the harness venv's python and return stdout."""
    out = subprocess.run(
        [str(bin_dir / "python"), "-I", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=timeout,
    )
    if out.returncode != 0:
        raise HarnessInstallError(out.stderr.strip()[-3000:])
    return out.stdout


def describe() -> str:  # pragma: no cover - convenience
    return f"harness envs under {DEFAULT_ROOT.resolve()} (python {sys.version_info.major}.{sys.version_info.minor} host)"
