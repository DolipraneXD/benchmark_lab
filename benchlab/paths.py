"""Where BenchLab keeps things, relative to the project folder.

``data/``       inputs that belong in git: app-task cases, fixed task subsets, your own
                retrieval documents.
``workspace/``  everything BenchLab makes on this machine (git-ignored): run folders,
                results and reports, installed benchmark harnesses, config backups.
                ``BENCHLAB_WORKSPACE`` moves it elsewhere, e.g. to a bigger disk.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path("data")


def workspace() -> Path:
    # read at call time so tests (and users) can point it somewhere else
    return Path(os.environ.get("BENCHLAB_WORKSPACE", "workspace"))


def harnesses() -> Path:
    """Installed official harnesses (one venv / clone per benchmark)."""
    override = os.environ.get("BENCHLAB_HARNESS_DIR")
    return Path(override) if override else workspace() / "harnesses"


def backups() -> Path:
    return workspace() / "backups"
