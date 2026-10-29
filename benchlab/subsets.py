"""Fixed task subsets: picked once with a seed, saved to ``subsets/<name>.json``
and reused for every model so all models see the same tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchlab import paths

SUBSETS_DIR = paths.DATA / "subsets"


def subset_path(name: str, root: Path | None = None) -> Path:
    return (root or SUBSETS_DIR) / f"{name}.json"


def load_subset(name: str, root: Path | None = None) -> dict[str, Any] | None:
    p = subset_path(name, root)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pick_subset(
    name: str,
    all_ids: list[str],
    n: int,
    seed: int,
    root: Path | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Return the saved subset if it exists, else sample ``n`` ids with ``seed``,
    save and return it. Sorting first makes the draw independent of input order."""
    existing = load_subset(name, root)
    if existing is not None:
        return existing
    pool = sorted(set(all_ids))
    if n >= len(pool):
        ids = pool
    else:
        rng = random.Random(seed)
        ids = sorted(rng.sample(pool, n))
    data = {
        "name": name,
        "n": len(ids),
        "seed": seed,
        "pool_size": len(pool),
        "source": source,
        "ids": ids,
    }
    p = subset_path(name, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return data
