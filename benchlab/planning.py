"""Which (model, bench) pairs a selection runs, with task counts and a rough cost
estimate. Shared by ``benchlab run`` (plan table, ``--dry-run``) and ``benchlab ui``."""

from __future__ import annotations

import os

from benchlab import registry
from benchlab.config import BenchConfig, ModelSpec
from benchlab.redact import redact


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
            rows.append((m, b, adapter_cls, n, k, est, notes))
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
