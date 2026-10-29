"""Run folders: ``workspace/runs/<timestamp>_<model>_<bench>/`` with config.json,
state.json (resume), result.json and the raw harness output."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchlab import __version__
from benchlab.redact import redact_obj
from benchlab.usage import UsageLog


def utc_stamp(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=UTC).strftime("%Y%m%d-%H%M%S")


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(redact_obj(data), f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class RunConfig:
    model_id: str
    bench: str
    model: dict[str, Any]  # ModelSpec.public_dict() (no key values)
    request_params: dict[str, Any]
    per_model_exceptions: dict[str, Any]
    harness: dict[str, Any]  # {"name", "version"/"commit", "package"}
    subset: dict[str, Any]  # {"name", "ids": [...], "seed", "n"}
    trials: int
    smoke: bool = False
    benchlab_version: str = __version__
    benchlab_commit: str | None = None
    created_at: str = ""
    user_model: str | None = None  # τ²-bench simulator etc.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorCounts:
    timeouts: int = 0
    api_errors: int = 0
    malformed_tool_calls: int = 0
    rate_limits: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return (
            self.timeouts
            + self.api_errors
            + self.malformed_tool_calls
            + self.rate_limits
            + self.other
        )


@dataclass
class NormalizedResult:
    """What every adapter's ``parse()`` returns and what ``result.json`` holds."""

    bench: str
    model_id: str
    metrics: dict[str, float | None]  # e.g. {"accuracy": 74.2, "pass^1": ..}; percentages 0-100
    primary_metric: str
    n_tasks: int
    trials: int
    errors: ErrorCounts = field(default_factory=ErrorCounts)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    wall_time_s: float = 0.0
    harness: dict[str, Any] = field(default_factory=dict)
    note: str = ""  # harness, subset size, trials, deviations - goes to tracker.json
    per_task: list[dict[str, Any]] = field(default_factory=list)
    tracker: dict[str, dict[str, str]] = field(default_factory=dict)  # key -> {"v","note"}
    warnings: list[str] = field(default_factory=list)

    @property
    def primary_value(self) -> float | None:
        return self.metrics.get(self.primary_metric)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["errors"]["total"] = self.errors.total
        return d


class RunDir:
    CONFIG = "config.json"
    STATE = "state.json"
    RESULT = "result.json"
    USAGE = "usage.jsonl"
    RAW = "raw"

    def __init__(self, path: Path):
        self.path = Path(path)

    # -- creation / discovery ----------------------------------------------
    @classmethod
    def create(cls, root: str | Path, model_id: str, bench: str) -> RunDir:
        root = Path(root)
        p = root / f"{utc_stamp()}_{model_id}_{bench}"
        i = 1
        while p.exists():  # two runs in the same second
            p = root / f"{utc_stamp()}_{model_id}_{bench}-{i}"
            i += 1
        (p / cls.RAW).mkdir(parents=True)
        return cls(p)

    @classmethod
    def find_resumable(cls, root: str | Path, model_id: str, bench: str) -> RunDir | None:
        """Newest run for (model, bench) that has a config but no result."""
        root = Path(root)
        if not root.exists():
            return None
        cands = sorted(
            [
                d
                for d in root.iterdir()
                if d.is_dir()
                and d.name.endswith(f"_{model_id}_{bench}")
                or (d.is_dir() and f"_{model_id}_{bench}-" in d.name)
            ],
            key=lambda d: d.name,
            reverse=True,
        )
        for d in cands:
            if (d / cls.CONFIG).exists() and not (d / cls.RESULT).exists():
                return cls(d)
        return None

    @classmethod
    def list_runs(cls, root: str | Path) -> list[RunDir]:
        root = Path(root)
        if not root.exists():
            return []
        return [cls(d) for d in sorted(root.iterdir()) if d.is_dir() and (d / cls.CONFIG).exists()]

    # -- files ----------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        d = self.path / self.RAW
        d.mkdir(exist_ok=True)
        return d

    @property
    def usage_log(self) -> UsageLog:
        return UsageLog(self.path / self.USAGE)

    @property
    def name(self) -> str:
        return self.path.name

    def write_config(self, cfg: RunConfig) -> None:
        if not cfg.created_at:
            cfg.created_at = datetime.now(UTC).isoformat(timespec="seconds")
        if cfg.benchlab_commit is None:
            cfg.benchlab_commit = git_commit(Path(__file__).resolve().parent.parent)
        atomic_write_json(self.path / self.CONFIG, asdict(cfg))

    def read_config(self) -> dict[str, Any]:
        return json.loads((self.path / self.CONFIG).read_text())

    def has_result(self) -> bool:
        return (self.path / self.RESULT).exists()

    def write_result(self, res: NormalizedResult) -> None:
        atomic_write_json(self.path / self.RESULT, res.to_dict())

    def read_result(self) -> dict[str, Any]:
        return json.loads((self.path / self.RESULT).read_text())

    # -- resume state -----------------------------------------------------------
    def state(self) -> dict[str, Any]:
        p = self.path / self.STATE
        if not p.exists():
            return {"completed": [], "failed": [], "started_at": time.time(), "elapsed_s": 0.0}
        return json.loads(p.read_text())

    def save_state(self, st: dict[str, Any]) -> None:
        atomic_write_json(self.path / self.STATE, st)

    def mark_done(self, task_id: str, ok: bool = True) -> None:
        st = self.state()
        key = "completed" if ok else "failed"
        if task_id not in st[key]:
            st[key].append(task_id)
        self.save_state(st)

    def completed_ids(self) -> set[str]:
        return set(self.state().get("completed", []))

    def remaining(self, all_ids: list[str]) -> list[str]:
        done = self.completed_ids()
        return [t for t in all_ids if t not in done]

    def add_elapsed(self, seconds: float) -> None:
        st = self.state()
        st["elapsed_s"] = float(st.get("elapsed_s", 0.0)) + seconds
        self.save_state(st)
