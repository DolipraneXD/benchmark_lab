"""results/results.csv: one row per (model, bench, metric)."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from benchlab.runs import NormalizedResult, RunDir

COLUMNS = [
    "model",
    "bench",
    "metric",
    "value",
    "n_tasks",
    "trials",
    "cost_usd",
    "tokens_in",
    "tokens_out",
    "wall_time_s",
    "errors_total",
    "timeouts",
    "api_errors",
    "malformed_tool_calls",
    "run_dir",
    "date",
]


def append_result(results_dir: str | Path, res: NormalizedResult, run_dir: RunDir) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / "results.csv"
    new = not p.exists()
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for metric, value in res.metrics.items():
            w.writerow(
                {
                    "model": res.model_id,
                    "bench": res.bench,
                    "metric": metric,
                    "value": "" if value is None else f"{value:.4f}",
                    "n_tasks": res.n_tasks,
                    "trials": res.trials,
                    "cost_usd": "" if res.cost_usd is None else f"{res.cost_usd:.4f}",
                    "tokens_in": res.tokens_in,
                    "tokens_out": res.tokens_out,
                    "wall_time_s": f"{res.wall_time_s:.1f}",
                    "errors_total": res.errors.total,
                    "timeouts": res.errors.timeouts,
                    "api_errors": res.errors.api_errors,
                    "malformed_tool_calls": res.errors.malformed_tool_calls,
                    "run_dir": str(run_dir.path),
                    "date": date,
                }
            )
    return p


def read_results(results_dir: str | Path) -> list[dict[str, str]]:
    p = Path(results_dir) / "results.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
