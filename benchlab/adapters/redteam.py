"""Red-team results imported from an external lab (promptfoo red-team CSV export).

Nothing is run here: the lab ran the attacks and its graders decided pass/fail.
BenchLab only copies the CSV into a run folder, counts the lab's verdicts and
writes the usual ``result.json`` so the report and tracker pick it up like any
other bench (``benchlab import-redteam``).

Expected columns (promptfoo "report" CSV): Test ID, Plugin, Plugin ID,
Strategy, Target, Prompt, Response, Pass, Score, Reason, Timestamp. A row whose
``Pass`` is ``Fail`` with an empty response *and* an empty reason was never
graded (the target returned nothing); it is counted as an API error, not as a
failed attack, exactly like an errored sample in the other adapters.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from benchlab.config import BenchConfig, ModelSpec
from benchlab.runs import ErrorCounts, NormalizedResult, RunConfig, RunDir

KEY = "redteam"
REQUIRED_COLUMNS = ("Test ID", "Plugin ID", "Strategy", "Target", "Response", "Pass", "Reason")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)  # responses can be long
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: not a promptfoo red-team CSV, missing columns {missing}")
    return rows


def verdict(row: dict[str, str]) -> str:
    """'pass' | 'fail' | 'error' from the lab's own Pass column."""
    p = (row.get("Pass") or "").strip().lower()
    if p.startswith("pass"):
        return "pass"
    if not (row.get("Response") or "").strip() and not (row.get("Reason") or "").strip():
        return "error"
    return "fail"


def normalize(
    rows: list[dict[str, str]], model_id: str, tracker_id: str, harness: dict[str, Any]
) -> NormalizedResult:
    by_plugin: dict[str, Counter] = {}
    by_group: dict[str, Counter] = {}
    total: Counter = Counter()
    per_task = []
    for r in rows:
        v = verdict(r)
        plugin = r["Plugin ID"].strip()
        by_plugin.setdefault(plugin, Counter())[v] += 1
        by_group.setdefault(plugin.split(":", 1)[0], Counter())[v] += 1
        total[v] += 1
        per_task.append(
            {
                "id": r["Test ID"],
                "plugin": plugin,
                "strategy": r.get("Strategy", ""),
                "verdict": v,
                "score": r.get("Score") or None,
                "reason": (r.get("Reason") or "")[:300],
            }
        )

    def rate(c: Counter) -> float | None:
        graded = c["pass"] + c["fail"]
        return c["pass"] / graded * 100 if graded else None

    n = len(rows)
    metrics: dict[str, float | None] = {
        "pass_rate": rate(total),
        "pass_rate_errors_as_fail": total["pass"] / n * 100,
        "n_pass": float(total["pass"]),
        "n_fail": float(total["fail"]),
        "n_error": float(total["error"]),
    }
    for g, c in sorted(by_group.items()):
        metrics[f"pass_rate_{g}"] = rate(c)
    for p, c in sorted(by_plugin.items()):
        metrics[f"plugin:{p}"] = rate(c)

    warnings = []
    if total["error"]:
        warnings.append(
            f"{total['error']}/{n} tests got no response from the target and were never graded; "
            "excluded from pass_rate, counted as failed in pass_rate_errors_as_fail"
        )
    thinking = sum(1 for r in rows if (r.get("Response") or "").lstrip().startswith("Thinking:"))
    if thinking:
        warnings.append(
            f"{thinking}/{n} responses start with visible 'Thinking:' text; the lab graded the "
            "reasoning too, so some failures (e.g. profanity) may come from the reasoning, not the answer"
        )
    targets = sorted({r["Target"] for r in rows})
    strategies = sorted({r.get("Strategy", "") for r in rows})
    note = (
        f"external red team ({harness.get('source_file')}), promptfoo CSV, target {', '.join(targets)}, "
        f"{len(by_plugin)} plugins, strategies {strategies}, {n} tests, pass = attack resisted per the "
        f"lab's graders; {total['error']} ungraded (no response) excluded; not run or re-scored by BenchLab"
    )
    tracker = {}
    if metrics["pass_rate"] is not None:
        tracker[f"{tracker_id}__{KEY}"] = {"v": f"{metrics['pass_rate']:.1f}", "note": note}
    return NormalizedResult(
        bench=KEY,
        model_id=model_id,
        metrics=metrics,
        primary_metric="pass_rate",
        n_tasks=n,
        trials=1,
        errors=ErrorCounts(api_errors=total["error"]),
        harness=harness,
        note=note,
        per_task=per_task,
        tracker=tracker,
        warnings=warnings,
    )


def import_csv(cfg: BenchConfig, model: ModelSpec, path: Path) -> tuple[RunDir, NormalizedResult]:
    """Create a run folder for ``model`` from the lab's CSV and write result.json."""
    path = Path(path)
    rows = read_rows(path)
    harness = {
        "name": "external red team (promptfoo CSV)",
        "source_file": path.name,
        "sha256": _sha256(path),
        "targets": sorted({r["Target"] for r in rows}),
        "evaluated_at": sorted({r.get("Timestamp", "") for r in rows})[-1],
        "imported": True,
    }
    run_dir = RunDir.create(cfg.settings.runs_dir, model.id, KEY)
    shutil.copy2(path, run_dir.raw_dir / path.name)
    run_dir.write_config(
        RunConfig(
            model_id=model.id,
            bench=KEY,
            model=model.public_dict(),
            request_params={},
            per_model_exceptions={},
            harness=harness,
            subset={"name": "lab", "ids": None, "n": len(rows)},
            trials=1,
            extra={"imported": True},
        )
    )
    res = normalize(rows, model.id, model.tracker, harness)
    run_dir.save_state({"completed": [r["Test ID"] for r in rows], "failed": [], "elapsed_s": 0.0})
    run_dir.write_result(res)
    return run_dir, res
