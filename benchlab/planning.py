"""Which (model, bench) pairs a selection runs, with task counts and rough cost and time
estimates. Shared by ``benchlab run`` (plan table, ``--dry-run``) and ``benchlab ui``."""

from __future__ import annotations

import os
from typing import Any, NamedTuple

from benchlab import registry
from benchlab.config import BenchConfig, ModelSpec
from benchlab.redact import redact
from benchlab.runs import RunDir

# Typical wall seconds for one task attempt at concurrency 1, from runs against
# mid-speed API models. Only a starting point: once a (model, bench) pair has run,
# its measured time per attempt is used instead.
TYPICAL_SECONDS = {
    "gpqa": 25,  # one chain-of-thought answer
    "bfcl": 4,  # one short tool-call turn
    "t2air": 150,  # multi-turn conversation with a simulated customer
    "t2ret": 150,
    "t2tel": 200,
    "tb": 600,  # agent loop in a Docker container
    "swe": 900,
    "gaia": 300,  # browsing agent
    "bcomp": 300,
    "ttft": 4,
    "autobench": 180,
    "lcb": 60,
    "aider": 120,
    "mteb_ret": 120,
    "mteb_rr": 60,
    "beir": 300,
    "own": 60,
    "serving": 300,
    "apptasks": 15,
}
SETUP_SECONDS = 60  # harness start-up per pair (venv, proxy, containers already pulled)


class PlanRow(NamedTuple):
    model: ModelSpec
    bench: registry.BenchInfo
    adapter_cls: Any
    n: int
    trials: int
    est_usd: float | None
    notes: list[str]
    est_s: float | None = None
    time_source: str = ""  # "last run" | "typical"


def measured_seconds_per_attempt(cfg: BenchConfig, model_id: str, bench: str) -> float | None:
    """Wall seconds per task attempt in the newest finished run of (model, bench)."""
    for rd in reversed(RunDir.list_runs(cfg.settings.runs_dir)):
        if not rd.name.endswith(f"_{model_id}_{bench}") or not rd.has_result():
            continue
        try:
            res = rd.read_result()
        except Exception:  # noqa: BLE001 - half-written folder
            continue
        attempts = int(res.get("n_tasks") or 0) * int(res.get("trials") or 1)
        wall = float(res.get("wall_time_s") or 0)
        if attempts and wall:
            return wall / attempts
    return None


def estimate_seconds(
    cfg: BenchConfig, m: ModelSpec, b: registry.BenchInfo, n: int, k: int
) -> tuple[float | None, str]:
    """Rough wall time for one pair: the model's own pace on this bench if it has run
    it before, else a typical pace spread over its parallel requests."""
    attempts = (n or 0) * k
    if not attempts:
        return None, ""
    measured = measured_seconds_per_attempt(cfg, m.id, b.key)
    if measured is not None:
        return SETUP_SECONDS + measured * attempts, "last run"
    per = TYPICAL_SECONDS.get(b.key, 60)
    parallel = max(1, min(m.max_concurrency, attempts))
    return SETUP_SECONDS + per * attempts / parallel, "typical"


def plan_rows(
    cfg: BenchConfig,
    specs: list[ModelSpec],
    benches: list[registry.BenchInfo],
    trials: int | None,
    subset: str | None,
    smoke: bool,
):
    rows = []
    for b in benches:
        adapter_cls = registry.load_adapter(b)
        k = trials if trials is not None else (b.trials_default or 1)
        if trials is None and b.key.startswith("t2") and cfg.settings.tau2_trials:
            k = cfg.settings.tau2_trials  # profile: lite -> 1
        for m in specs:
            if not registry.applies_to(b, m):
                continue
            n = b.n_tasks_default
            notes = []
            if adapter_cls is not None:
                try:
                    plan = adapter_cls(cfg).plan(m, subset, k, smoke)
                    n, tin, tout, notes = (
                        plan.n_tasks,
                        plan.est_tokens_in_per_task,
                        plan.est_tokens_out_per_task,
                        plan.notes,
                    )
                except Exception as e:  # noqa: BLE001
                    tin, tout = b.tokens_in_per_task, b.tokens_out_per_task
                    notes = [f"plan failed: {redact(e)}"]
            else:
                tin, tout = b.tokens_in_per_task, b.tokens_out_per_task
                if smoke:
                    n = min(n or 3, 3)
                notes = ["adapter not integrated yet"]
            est = m.cost_usd(tin * n * k, tout * n * k) if n else None
            secs, source = estimate_seconds(cfg, m, b, n, k) if adapter_cls else (None, "")
            rows.append(PlanRow(m, b, adapter_cls, n, k, est, notes, secs, source))
    return rows


def _no_key(m: ModelSpec) -> bool:
    return bool(m.api_key_env) and not os.environ.get(m.api_key_env)


def blocked_reason(cfg: BenchConfig, m: ModelSpec, b: registry.BenchInfo) -> str | None:
    """Why (m, b) can't run because a key is missing: the model's own, or that of the
    helper model the bench needs (τ²-bench customer, BrowseComp grader). None = fine."""
    if _no_key(m):
        return f"{m.id} has no API key ({m.api_key_env})."
    helpers = []
    if b.key.startswith("t2"):
        helpers.append(("τ²-bench customer (user simulator)", cfg.settings.tau2_user_model))
    if b.key == "bcomp":
        grader = cfg.settings.grader_model or cfg.settings.baseline_model
        helpers.append(("BrowseComp grader", grader))
    for role, hid in helpers:
        try:
            h = cfg.get(hid)
        except KeyError:
            return f"The {role} is {hid}, which is not in this config."
        if _no_key(h):
            return (
                f"The {role} is {h.id}, which has no API key ({h.api_key_env}). "
                "Save its key, or choose another model for that role."
            )
    return None


def fmt_duration(s: float | None) -> str:
    """1h05m / 12m30s / 45s."""
    if s is None:
        return "-"
    s = int(s)
    h, rem = divmod(s, 3600)
    mi, sec = divmod(rem, 60)
    return f"{h}h{mi:02d}m" if h else (f"{mi}m{sec:02d}s" if mi else f"{sec}s")
