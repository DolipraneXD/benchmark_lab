"""Background jobs: a panel "Start" runs the very same ``benchlab run`` command the
CLI would, detached in its own process group so it outlives the panel.

Files per job in ``<runs_dir>/.jobs/``: ``<id>.json`` (what was started, written
once), ``<id>.log`` (stdout+stderr), ``<id>.exit`` (exit code, written by the
launcher when the command ends). Stop = SIGINT to the process group, which the
CLI turns into "progress saved, re-run to resume".
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchlab.runs import RunDir, utc_stamp
from benchlab.usage import UsageLog

JOBS_DIR = ".jobs"
EXIT_MEANING = {0: "done", 1: "failed", 2: "config error", 3: "budget reached", 130: "stopped"}

_children: dict[int, subprocess.Popen] = {}  # launchers started by this panel process (reaped)


@dataclass
class Job:
    id: str
    argv: list[str]
    config: str
    pairs: list[list[str]]  # [[model_id, bench], ...] from the plan
    started_at: float
    pid: int = 0
    budget_usd: float | None = None
    smoke: bool = False
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def jobs_dir(runs_dir: str | Path) -> Path:
    d = Path(runs_dir) / JOBS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paths(runs_dir: str | Path, job_id: str) -> tuple[Path, Path, Path]:
    d = jobs_dir(runs_dir)
    return d / f"{job_id}.json", d / f"{job_id}.log", d / f"{job_id}.exit"


def start(runs_dir: str | Path, job: Job, cwd: str | Path | None = None) -> Job:
    jpath, log, _ = _paths(runs_dir, job.id)
    jpath.write_text(json.dumps(asdict(job), indent=2))
    p = subprocess.Popen(
        [sys.executable, "-m", "benchlab.ui.launcher", str(jpath.resolve())],
        cwd=cwd,
        env=job_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # own process group: survives the panel, stoppable as a group
    )
    _children[p.pid] = p
    job.pid = p.pid
    jpath.write_text(json.dumps(asdict(job), indent=2))
    return job


def job_env() -> dict[str, str]:
    """The panel's environment with its own venv's bin dir first on PATH, as `uv run` or an
    activated venv would give: harness tools such as the `litellm` proxy CLI live there,
    and the panel may have been started as `.venv/bin/benchlab ui` without activation."""
    env = dict(os.environ)
    bin_dir = str(Path(sys.executable).parent)
    parts = env.get("PATH", "").split(os.pathsep)
    env["PATH"] = os.pathsep.join([bin_dir, *[p for p in parts if p and p != bin_dir]])
    return env


def new_id() -> str:
    return f"{utc_stamp()}-{secrets.token_hex(2)}"


def load(runs_dir: str | Path, job_id: str) -> Job:
    jpath, _, _ = _paths(runs_dir, job_id)
    return Job(**json.loads(jpath.read_text()))


def list_jobs(runs_dir: str | Path) -> list[Job]:
    out = []
    for p in sorted(jobs_dir(runs_dir).glob("*.json"), reverse=True):
        try:
            out.append(Job(**json.loads(p.read_text())))
        except Exception:  # noqa: BLE001 - half-written file
            continue
    return out


def exit_code(runs_dir: str | Path, job: Job) -> int | None:
    _, _, epath = _paths(runs_dir, job.id)
    try:
        return int(epath.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _alive(pid: int) -> bool:
    if pid in _children:
        return _children[pid].poll() is None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status(runs_dir: str | Path, job: Job) -> str:
    """running | done | failed | config error | budget reached | stopped | lost | exit N"""
    rc = exit_code(runs_dir, job)
    if rc is not None:
        return EXIT_MEANING.get(rc, f"exit {rc}")
    if job.pid and _alive(job.pid):
        return "running"
    return "lost"  # killed without the launcher writing an exit code (reboot, kill -9)


def stop(runs_dir: str | Path, job: Job) -> bool:
    if status(runs_dir, job) != "running":
        return False
    try:
        os.killpg(job.pid, signal.SIGINT)
    except ProcessLookupError:
        return False
    return True


def log_tail(runs_dir: str | Path, job: Job, max_bytes: int = 20000) -> str:
    _, log, _ = _paths(runs_dir, job.id)
    if not log.exists():
        return ""
    with open(log, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read().decode("utf-8", errors="replace")
    return data if size <= max_bytes else "…\n" + data.split("\n", 1)[-1]


def running_pairs(runs_dir: str | Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for j in list_jobs(runs_dir):
        if status(runs_dir, j) == "running":
            out |= {(m, b) for m, b in j.pairs}
    return out


def run_dir_for(runs_dir: str | Path, job: Job, model_id: str, bench: str) -> RunDir | None:
    """The run folder this job is writing for (model, bench): the newest one for the pair
    that was created or touched after the job started (a resumed run keeps its old name)."""
    root = Path(runs_dir)
    if not root.exists():
        return None
    cands = [
        d
        for d in root.iterdir()
        if d.is_dir()
        and (d.name.endswith(f"_{model_id}_{bench}") or f"_{model_id}_{bench}-" in d.name)
    ]
    for d in sorted(cands, key=lambda d: d.name, reverse=True):
        touched = max(
            (
                p.stat().st_mtime
                for p in (d / "config.json", d / "state.json", d / "usage.jsonl")
                if p.exists()
            ),
            default=0.0,
        )
        if touched >= job.started_at - 2:
            return RunDir(d)
    return None


def progress(runs_dir: str | Path, job: Job) -> list[dict[str, Any]]:
    """Per (model, bench): done / planned tasks, usage so far, result when finished."""
    rows = []
    for model_id, bench in job.pairs:
        rd = run_dir_for(runs_dir, job, model_id, bench)
        row: dict[str, Any] = {"model": model_id, "bench": bench, "run": None}
        if rd is not None:
            cfg = rd.read_config() if (rd.path / RunDir.CONFIG).exists() else {}
            st = rd.state()
            planned = (cfg.get("subset") or {}).get("n")
            u = UsageLog(rd.path / RunDir.USAGE).summarize().as_dict()
            row.update(
                run=rd.name,
                done=len(st.get("completed", [])),
                planned=planned,
                usage=u,
                elapsed_s=st.get("elapsed_s"),
            )
            if rd.has_result():
                res = rd.read_result()
                row["result"] = {
                    "metric": res["primary_metric"],
                    "value": res["metrics"].get(res["primary_metric"]),
                    "warnings": res.get("warnings") or [],
                }
        rows.append(row)
    return rows


def age(job: Job) -> float:
    return time.time() - job.started_at
