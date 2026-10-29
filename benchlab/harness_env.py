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

from benchlab import paths


class HarnessInstallError(RuntimeError):
    pass


def harness_root(name: str, root: Path | None = None) -> Path:
    # read at call time so tests can point it at a temp dir
    return (root or paths.harnesses()) / name


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
                return (venv / "bin").resolve()
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
    return (venv / "bin").resolve()


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
    return f"harness envs under {paths.harnesses().resolve()} (python {sys.version_info.major}.{sys.version_info.minor} host)"


def ensure_repo_env(
    name: str,
    url: str,
    ref: str,
    root: Path | None = None,
    python: str = "3.12",
    sync_args: list[str] | None = None,
) -> tuple[Path, Path]:
    """Clone ``url`` at ``ref`` into ``harnesses/<name>/repo`` and ``uv sync`` it
    (the official install for repo-based harnesses such as tau2-bench). Returns
    ``(repo_dir, venv_bin_dir)``. Re-clones when ``ref`` changes."""
    base = harness_root(name, root)
    repo = base / "repo"
    manifest = base / "manifest.json"
    venv_bin = repo / ".venv" / "bin"
    if repo.exists() and manifest.exists():
        try:
            have = json.loads(manifest.read_text())
            if have.get("url") == url and have.get("ref") == ref and (venv_bin / "python").exists():
                return repo.resolve(), venv_bin.resolve()
        except json.JSONDecodeError:
            pass
    if repo.exists():
        shutil.rmtree(repo)
    base.mkdir(parents=True, exist_ok=True)
    uv = _uv()
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = ".venv"
    try:
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", ref, url, str(repo)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:  # ``ref`` may be a commit sha: full clone then checkout
            subprocess.run(
                ["git", "clone", "--quiet", url, str(repo)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--quiet", ref],
                check=True,
                capture_output=True,
                text=True,
            )
        subprocess.run(
            [uv, "sync", "--quiet", "--python", python, *(sync_args or [])],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(repo),
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise HarnessInstallError(
            f"failed to set up harness repo {name!r} at {ref}: {e.stderr.strip()[-2000:] or e.stdout.strip()[-2000:]}"
        ) from e
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    manifest.write_text(
        json.dumps(
            {
                "url": url,
                "ref": ref,
                "commit": commit,
                "python": python,
                "installed": frozen_versions(venv_bin),
                "venv": str(repo / ".venv"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return repo.resolve(), venv_bin.resolve()


def repo_commit(repo: Path) -> str | None:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip() or None if out.returncode == 0 else None


def ensure_repo(
    name: str, url: str, ref: str, subdir: str = "repo", root: Path | None = None
) -> Path:
    """Clone ``url`` at ``ref`` into ``harnesses/<name>/<subdir>`` (no venv). Re-clones
    when the recorded ref changes. Returns the absolute repo path."""
    base = harness_root(name, root)
    repo = base / subdir
    marker = base / f"{subdir}.ref.json"
    if repo.exists() and marker.exists():
        try:
            have = json.loads(marker.read_text())
            if have.get("url") == url and have.get("ref") == ref and (repo / ".git").exists():
                return repo.resolve()
        except json.JSONDecodeError:
            pass
    if repo.exists():
        shutil.rmtree(repo)
    base.mkdir(parents=True, exist_ok=True)
    try:
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", ref, url, str(repo)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:  # a commit sha: full clone then checkout
            subprocess.run(
                ["git", "clone", "--quiet", url, str(repo)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--quiet", ref],
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        raise HarnessInstallError(f"failed to clone {url}@{ref}: {e.stderr.strip()[-1500:]}") from e
    marker.write_text(json.dumps({"url": url, "ref": ref, "commit": repo_commit(repo)}, indent=2))
    return repo.resolve()


def ensure_pip_venv_in_repo(
    repo: Path, python: str, packages: list[str], editable_no_deps: bool = True
) -> Path:
    """``repo/.venv`` with ``packages`` plus the repo installed ``--no-deps -e`` (for
    harnesses whose declared deps include GPU stacks we do not need). Returns bin dir."""
    venv = repo / ".venv"
    manifest = repo.parent / f"{repo.name}.venv.json"
    want = {"python": python, "packages": sorted(packages), "editable_no_deps": editable_no_deps}
    if (venv / "bin" / "python").exists() and manifest.exists():
        try:
            if json.loads(manifest.read_text()).get("want") == want:
                return (venv / "bin").resolve()
        except json.JSONDecodeError:
            pass
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
        py = str(venv / "bin" / "python")
        subprocess.run(
            [uv, "pip", "install", "--quiet", "--python", py, *packages],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        edit = ["--no-deps"] if editable_no_deps else []
        subprocess.run(
            [uv, "pip", "install", "--quiet", "--python", py, *edit, "-e", str(repo)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise HarnessInstallError(
            f"failed to build venv in {repo}: {e.stderr.strip()[-2000:]}"
        ) from e
    manifest.write_text(
        json.dumps(
            {"want": want, "installed": frozen_versions(venv / "bin")}, indent=2, sort_keys=True
        )
    )
    return (venv / "bin").resolve()
