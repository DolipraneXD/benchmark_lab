"""API keys in ``.env``, set from the control panel.

``.env`` is the file BenchLab already reads keys from, so a key saved here works for
panel jobs and terminal runs alike. Safety rules:

* only variables that a model in the config names in ``api_key_env`` can be written,
* the file is kept readable by its owner only (0600) and written atomically,
* no backup copy is made (it would be a second copy of every key),
* values are never sent back to the page: only the last 4 characters are shown.

Each model gets its own variable, named after its config and id (``key_name``), so
adding a model never overwrites another model's key, even across the two configs.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path

from dotenv import dotenv_values

from benchlab.redact import register_secret_env

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# characters that would need quoting in .env, or that no API key contains
BAD_VALUE = re.compile(r"[\s'\"\\#]")


PREFIX = "BENCHLAB_KEY_"


def key_name(config_file: str, model_id: str) -> str:
    """The model's own key variable: BENCHLAB_KEY_<ID> for models.yaml,
    BENCHLAB_KEY_<TAG>_<ID> for models.<tag>.yaml. Model ids are unique in a config,
    so the name is unique across every config sharing this .env."""
    stem = Path(config_file).name.removesuffix(".yaml").removesuffix(".yml")
    tag = stem.removeprefix("models").strip("._-")
    tag = re.sub(r"[^A-Za-z0-9]+", "_", tag).upper()
    return PREFIX + (f"{tag}_" if tag else "") + model_id.upper()


def clean_value(value: str) -> str:
    """The pasted key, checked. Raises ValueError with a sentence the page can show."""
    value = value.strip()
    if not value:
        raise ValueError("The key is empty.")
    if BAD_VALUE.search(value):
        raise ValueError(
            "The key contains spaces, quotes, # or a backslash. Paste only the key itself."
        )
    if NAME.fullmatch(value) and value.isupper() and "_" in value and len(value) < 40:
        raise ValueError("Paste the key itself here, not the variable name.")
    return value


def _line_for(name: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*(export\s+)?{re.escape(name)}\s*=")


def lock(path: Path) -> bool:
    """Make ``path`` owner-only if group or others can read it. True if it changed."""
    if not path.exists():
        return False
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        path.chmod(0o600)
        return True
    return False


def in_file(path: Path) -> dict[str, str]:
    """Variables written in ``path`` (empty values left out)."""
    if not path.exists():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v}


def masked(value: str | None) -> str:
    if not value:
        return ""
    return "••••" + value[-4:] if len(value) >= 12 else "••••"


def _write(path: Path, lines: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".env-", suffix=".tmp")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _check(name: str, allowed: set[str]) -> None:
    if not NAME.fullmatch(name):
        raise ValueError("That is not a variable name.")
    if name not in allowed:
        raise ValueError(
            f"{name} is not used by any model in this config. "
            "Set it as the Key variable of a model first."
        )


def set_key(path: Path, name: str, value: str, allowed: set[str]) -> None:
    """Write ``name=value`` into ``path`` (replacing an existing line) and make it
    live for this process, so checks and new jobs use it right away."""
    _check(name, allowed)
    value = clean_value(value)
    pat = _line_for(name)
    out, done = [], False
    for line in _read(path):
        if pat.match(line):
            if not done:
                out.append(f"{name}={value}\n")
                done = True
            continue  # drop duplicates of the same variable
        out.append(line)
    if not done:
        out.append(f"{name}={value}\n")
    _write(path, out)
    register_secret_env(name)
    os.environ[name] = value


def remove_key(path: Path, name: str, allowed: set[str]) -> None:
    """Delete ``name`` from ``path``. A value set in your shell is left alone."""
    _check(name, allowed)
    old = in_file(path).get(name)
    pat = _line_for(name)
    lines = _read(path)
    kept = [line for line in lines if not pat.match(line)]
    if len(kept) == len(lines):
        raise ValueError(f"{name} is not in .env.")
    _write(path, kept)
    if old is not None and os.environ.get(name) == old:
        os.environ.pop(name, None)
