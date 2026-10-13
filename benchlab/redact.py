"""Scrub API keys from anything we log or print.

Two layers: every secret value that benchlab has loaded from the environment is
replaced verbatim, and common key shapes (``sk-...``, ``Bearer ...``) are masked
as a fallback for keys we did not load ourselves.
"""

from __future__ import annotations

import os
import re
from typing import Any

_SECRET_ENV_NAMES: set[str] = set()
_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(r"(x-api-key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
]


def register_secret_env(name: str | None) -> None:
    """Remember an env var name whose value must never appear in output."""
    if name:
        _SECRET_ENV_NAMES.add(name)


def _secret_values() -> list[str]:
    vals = []
    for name in _SECRET_ENV_NAMES:
        v = os.environ.get(name)
        if v and len(v) >= 6:
            vals.append(v)
    # longest first so partial overlaps don't leave fragments behind
    return sorted(vals, key=len, reverse=True)


def redact(text: Any) -> str:
    """Return ``text`` as a string with every known secret masked."""
    s = str(text)
    for v in _secret_values():
        s = s.replace(v, "***REDACTED***")
    for pat in _PATTERNS:
        s = pat.sub(lambda m: m.group(1) + "***", s)
    return s


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings inside dicts/lists (for JSON logs)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
