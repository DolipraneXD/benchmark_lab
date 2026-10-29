"""Stage 8: ``benchlab report`` -> ``report.html`` + ``results/tracker.json``.

Reads every finished run under ``runs/`` (the newest non-smoke run per
(model, bench) wins; smoke runs are used only when nothing else exists and are
flagged), merges the adapters' tracker entries, derives the aggregate keys the
tracker expects (``cost``, ``srv_match``) and renders one
static HTML page: models x benchmarks with the delta against the baseline,
cost per task, serving / TTFT / retrieval tables and a "suspicious results"
list. Nothing here re-scores anything: values come straight from result.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchlab import __version__
from benchlab.config import BenchConfig, ModelSpec
from benchlab.registry import BENCHES, BenchInfo
from benchlab.runs import RunDir

CORE_SCORE_BENCHES = ("gpqa", "bfcl", "t2air", "t2ret", "t2tel", "tb", "swe", "gaia")
NON_SCORE_BENCHES = {"ttft", "serving"}  # seconds / tokens per second, not percentages
TAU2_BENCHES = ("t2air", "t2ret", "t2tel")
TOOL_BENCHES = (
    "bfcl",
    "t2air",
    "t2ret",
    "t2tel",
)  # where a self-hosted gap usually means a chat-template/tool-parser problem
COMPUTED_KEYS = {"cost", "srv_match"}  # derived here


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    model_id: str
    bench: str
    run: RunDir
    config: dict[str, Any]
    result: dict[str, Any]

    @property
    def smoke(self) -> bool:
        return bool(self.config.get("smoke"))

    @property
    def primary(self) -> float | None:
        v = self.result["metrics"].get(self.result["primary_metric"])
        return None if v is None else float(v)

    @property
    def n_tasks(self) -> int:
        return int(self.result.get("n_tasks") or 0)

    @property
    def trials(self) -> int:
        return int(self.result.get("trials") or 1)

    @property
    def cost(self) -> float | None:
        c = self.result.get("cost_usd")
        return None if c is None else float(c)

    @property
    def errors(self) -> dict[str, int]:
        return self.result.get("errors") or {}

    @property
    def error_total(self) -> int:
        e = self.errors
        return (
            int(e.get("timeouts", 0))
            + int(e.get("api_errors", 0))
            + int(e.get("malformed_tool_calls", 0))
        )

    @property
    def error_rate(self) -> float:
        n = max(1, self.n_tasks * self.trials)
        return self.error_total / n

    @property
    def cost_per_task(self) -> float | None:
        if self.cost is None or self.n_tasks == 0:
            return None
        return self.cost / (self.n_tasks * self.trials)

    @property
    def solved(self) -> float | None:
        """Estimated solved task-attempts: n_tasks * trials * primary/100 (score benches only)."""
        p = self.primary
        if p is None or self.bench in NON_SCORE_BENCHES:
            return None
        return self.n_tasks * self.trials * p / 100.0

    @property
    def date(self) -> str:
        return (self.config.get("created_at") or "")[:10]


@dataclass
class Collected:
    entries: dict[tuple[str, str], Entry] = field(
        default_factory=dict
    )  # (model_id, bench) -> newest usable
    smoke_only: set[tuple[str, str]] = field(default_factory=set)
    all_runs: list[Entry] = field(default_factory=list)

    def get(self, model_id: str, bench: str) -> Entry | None:
        return self.entries.get((model_id, bench))

    def benches_with_results(self) -> list[str]:
        have = {b for (_, b) in self.entries}
        return [k for k in BENCHES if k in have] + sorted(have - set(BENCHES))


def collect(cfg: BenchConfig, include_smoke: bool = True) -> Collected:
    col = Collected()
    for r in RunDir.list_runs(cfg.settings.runs_dir):
        if not r.has_result():
            continue
        try:
            c, res = r.read_config(), r.read_result()
        except Exception:  # noqa: BLE001 - half-written folder
            continue
        e = Entry(c["model_id"], c["bench"], r, c, res)
        col.all_runs.append(e)
    # newest first (folder names start with the UTC stamp)
    for e in sorted(col.all_runs, key=lambda e: e.run.name, reverse=True):
        key = (e.model_id, e.bench)
        cur = col.entries.get(key)
        if cur is None:
            if e.smoke and not include_smoke:
                continue
            col.entries[key] = e
            if e.smoke:
                col.smoke_only.add(key)
        elif cur.smoke and not e.smoke:  # a real run beats a newer smoke run
            col.entries[key] = e
            col.smoke_only.discard(key)
    return col


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #
def _f1(v: float) -> str:
    return f"{v:.1f}"


def _tracker_id(cfg: BenchConfig, model_id: str) -> str:
    try:
        return cfg.get(model_id).tracker
    except KeyError:
        return model_id


def _score(col: Collected, model_id: str, bench: str) -> float | None:
    e = col.get(model_id, bench)
    return e.primary if e else None


def build_tracker(cfg: BenchConfig, col: Collected) -> dict[str, dict[str, str]]:
    tracker: dict[str, dict[str, str]] = {}
    # 1. what the adapters wrote (oldest first so the newest usable run wins)
    for e in sorted(col.entries.values(), key=lambda e: e.run.name):
        for k, v in (e.result.get("tracker") or {}).items():
            note = v.get("note", "")
            if e.smoke and "SMOKE" not in note:
                note += "; SMOKE"
            tracker[k] = {"v": str(v.get("v")), "note": note}
    for m in cfg.reference_models():
        for b, v in m.published.items():
            tracker[f"{m.tracker}__{b}"] = {
                "v": _f1(float(v)),
                "note": "published score (reference model, not measured by BenchLab)",
            }
    for m in cfg.chat_models():
        tid = m.tracker
        # 2. <id>__cost: total USD over the core benches that have a result
        core = [col.get(m.id, b) for b in CORE_SCORE_BENCHES]
        core = [e for e in core if e]
        priced = [e for e in core if e.cost is not None]
        if priced:
            total = sum(e.cost for e in priced)
            parts = ", ".join(f"{e.bench} ${e.cost:.2f}" for e in priced)
            missing = [e.bench for e in core if e.cost is None]
            note = f"total USD over core benches from the usage logs: {parts}"
            if missing:
                note += f"; no cost for {missing} (unpriced model or self-hosted)"
            tracker[f"{tid}__cost"] = {"v": f"{total:.2f}", "note": note}
        # 3. self-hosted vs API counterpart: <id>__srv_match
        if m.self_hosted:
            entry = _srv_match_entry(cfg, col, m)
            if entry:
                tracker[f"{tid}__srv_match"] = entry
    return dict(sorted(tracker.items()))


def _sanity_entry(cfg: BenchConfig, col: Collected, m: ModelSpec) -> dict[str, str]:
    thr = cfg.settings.sanity_threshold_pts
    if not m.published:
        return {
            "v": "n/a",
            "note": "fill `published: {bench: score}` on the model in models.yaml to enable the sanity check",
        }
    diffs = []
    missing = []
    for b, pub in m.published.items():
        ours = _score(col, m.id, b)
        if ours is None:
            missing.append(b)
        else:
            diffs.append((b, ours, float(pub), ours - float(pub)))
    if not diffs:
        return {"v": "n/a", "note": f"no results yet for published benches {sorted(m.published)}"}
    worst = max(diffs, key=lambda d: abs(d[3]))
    parts = "; ".join(f"{b}: ours {o:.1f} vs published {p:.1f} ({d:+.1f})" for b, o, p, d in diffs)
    off = [b for b, _, _, d in diffs if abs(d) > thr]
    verdict = (
        f"OFF by >{thr:g} pts on {off} - check chat template / tool parser / quantization / subset"
        if off
        else f"within {thr:g} pts of published"
    )
    note = f"{verdict}; {parts}" + (f"; no result yet for {missing}" if missing else "")
    return {"v": f"{worst[3]:+.1f}", "note": note}


def _common_core(col: Collected, a: ModelSpec, b: ModelSpec | None) -> list[str]:
    out = []
    for bench in CORE_SCORE_BENCHES:
        if _score(col, a.id, bench) is None:
            continue
        if b is not None and _score(col, b.id, bench) is None:
            continue
        out.append(bench)
    return out


def _srv_match_entry(cfg: BenchConfig, col: Collected, m: ModelSpec) -> dict[str, str] | None:
    """Self-hosted vs API counterpart (compare_to) on common core benches; else vs published."""
    if m.compare_to:
        other = cfg.get(m.compare_to)
        benches = _common_core(col, m, other)
        if not benches:
            return {
                "v": "n/a",
                "note": f"no core bench has results for both {m.id} and {other.id} yet",
            }
        gaps = [(b, _score(col, m.id, b) - _score(col, other.id, b)) for b in benches]
        mean = sum(g for _, g in gaps) / len(gaps)
        note = f"self-hosted minus API ({other.id}) in points, mean over {benches}: " + ", ".join(
            f"{b} {g:+.1f}" for b, g in gaps
        )
        if mean < -cfg.settings.self_hosted_gap_pts:
            note += "; self-hosted clearly below the API: check chat template, tool-call parser, quantization, max context"
        return {"v": f"{mean:+.1f}", "note": note}
    if m.published:
        e = _sanity_entry(cfg, col, m)
        e["note"] = (
            "vs published scores (no API counterpart configured via compare_to); " + e["note"]
        )
        return e
    return {
        "v": "n/a",
        "note": "set compare_to (API counterpart id) or published scores on the model to compute the gap",
    }


# --------------------------------------------------------------------------- #
# suspicious results
# --------------------------------------------------------------------------- #
def suspicious(cfg: BenchConfig, col: Collected) -> list[dict[str, str]]:
    s = cfg.settings
    out: list[dict[str, str]] = []

    def add(e: Entry | None, model: str, bench: str, why: str, hint: str = "") -> None:
        out.append(
            {
                "model": model,
                "bench": bench,
                "why": why,
                "hint": hint,
                "run": e.run.name if e else "",
            }
        )

    baseline = cfg.baseline
    for (mid, bench), e in sorted(col.entries.items()):
        try:
            m = cfg.get(mid)
        except KeyError:
            m = None
        if (mid, bench) in col.smoke_only:
            add(
                e,
                mid,
                bench,
                "only a --smoke run exists (2-3 tasks)",
                "run the full bench before comparing",
            )
        if e.error_rate > s.suspicious_error_rate:
            er = e.errors
            add(
                e,
                mid,
                bench,
                f"error rate {e.error_rate:.0%}: timeouts {er.get('timeouts', 0)}, API errors {er.get('api_errors', 0)}, malformed tool calls {er.get('malformed_tool_calls', 0)}",
                "errored tasks count as failed; fix the endpoint / raise timeouts and re-run",
            )
        p = e.primary
        if (
            p is not None
            and bench not in NON_SCORE_BENCHES
            and e.n_tasks >= 5
            and (p <= 0.0 or p >= 100.0)
        ):
            add(
                e,
                mid,
                bench,
                f"score {p:.1f} on {e.n_tasks} tasks",
                "all-or-nothing scores usually mean a harness/format problem, not model quality",
            )
        imported = bool((e.config.get("extra") or {}).get("imported"))  # no usage log by design
        if (
            m
            and m.has_prices
            and e.cost is None
            and bench not in NON_SCORE_BENCHES
            and e.n_tasks
            and not imported
        ):
            add(
                e,
                mid,
                bench,
                "no cost recorded although the model has prices",
                "usage.jsonl empty? the harness may have bypassed the proxy",
            )
        for w in e.result.get("warnings") or []:
            add(e, mid, bench, f"adapter warning: {w}")
        if m and m.published.get(bench) is not None and p is not None:
            d = p - float(m.published[bench])
            if abs(d) > s.sanity_threshold_pts:
                add(
                    e,
                    mid,
                    bench,
                    f"{d:+.1f} pts vs published {float(m.published[bench]):.1f}",
                    "subset/prompt/temperature differences first; then chat template, tool parser, quantization",
                )
        if m and m.self_hosted and bench in TOOL_BENCHES and p is not None and mid != baseline.id:
            bl = _score(col, baseline.id, bench)
            if bl is not None and p < bl - 20:
                add(
                    e,
                    mid,
                    bench,
                    f"{p:.1f} vs baseline {bl:.1f} on a tool-calling bench",
                    "self-hosted << API usually means wrong chat template, missing --tool-call-parser / --enable-auto-tool-choice, or a lossy quantization",
                )
    for m in cfg.chat_models():
        if m.self_hosted and m.compare_to:
            other = cfg.get(m.compare_to)
            for b in _common_core(col, m, other):
                gap = _score(col, m.id, b) - _score(col, other.id, b)
                if gap < -s.self_hosted_gap_pts:
                    add(
                        col.get(m.id, b),
                        m.id,
                        b,
                        f"{gap:+.1f} pts below the API counterpart {other.id}",
                        "chat template / tool parser / quantization / context length on the self-hosted box",
                    )
    return out


# --------------------------------------------------------------------------- #
# comparison: what the panel's Results page and report.html both show
# --------------------------------------------------------------------------- #
LOWER_BETTER = {"ttft"}  # seconds: smaller is better


def _bench_info(key: str) -> BenchInfo:
    return BENCHES.get(key) or BenchInfo(key, key, 4, (), key, "")


def comparison(cfg: BenchConfig, col: Collected, failures: int = 0) -> dict[str, Any]:
    """Models x benchmarks with scores, the delta vs the baseline, tests passed / failed,
    cost and time; a per-model summary; and up to ``failures`` failed tests per cell
    (prompts and answers) for the exported report."""
    from benchlab import samples

    base_id = cfg.baseline.id
    benches = [_bench_info(b) for b in col.benches_with_results()]
    models = [
        m for m in cfg.models if not m.reference and any(col.get(m.id, b.key) for b in benches)
    ]
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for m in models:
        row = cells.setdefault(m.id, {})
        for b in benches:
            e = col.get(m.id, b.key)
            if e is None:
                continue
            base = col.get(base_id, b.key)
            try:
                detail = samples.load(e.run)
            except Exception:  # noqa: BLE001 - a broken raw file must not break the page
                detail = {"counts": {"pass": 0, "fail": 0, "error": 0}, "samples": []}
            attempts = max(1, e.n_tasks * e.trials)
            toks = int(e.result.get("tokens_in") or 0) + int(e.result.get("tokens_out") or 0)
            row[b.key] = {
                "value": e.primary,
                "metric": e.result.get("primary_metric"),
                "base": None if m.id == base_id or base is None else base.primary,
                "is_base": m.id == base_id,
                "pct": b.key not in NON_SCORE_BENCHES,
                "lower": b.key in LOWER_BETTER,
                "n": e.n_tasks,
                "trials": e.trials,
                "cost": e.cost,
                "wall": float(e.result.get("wall_time_s") or 0) or None,
                "errors": e.error_total,
                "smoke": e.smoke,
                "run": e.run.name,
                "date": e.date,
                "warnings": e.result.get("warnings") or [],
                "note": e.result.get("note") or "",
                "tokens_per_task": toks / attempts if toks else None,
                "counts": detail["counts"],
                "failures": [s for s in detail["samples"] if s["status"] != "pass"][:failures],
            }
    summary = {}
    for m in models:
        row = cells.get(m.id, {})
        pct = {k: c for k, c in row.items() if c["pct"] and c["value"] is not None}
        deltas = [c["value"] - c["base"] for c in pct.values() if c["base"] is not None]
        counts = {
            k: sum(c["counts"].get(k, 0) for c in row.values()) for k in ("pass", "fail", "error")
        }
        costs = [c["cost"] for c in row.values() if c["cost"] is not None]
        summary[m.id] = {
            "benches": len(row),
            "mean": sum(c["value"] for c in pct.values()) / len(pct) if pct else None,
            "delta": sum(deltas) / len(deltas) if deltas else None,
            "delta_n": len(deltas),
            "counts": counts,
            "cost": sum(costs) if costs else None,
            "wall": sum(c["wall"] or 0 for c in row.values()) or None,
            "errors": sum(c["errors"] for c in row.values()),
            "smoke": any(c["smoke"] for c in row.values()),
        }
    return {
        "models": models,
        "benches": benches,
        "cells": cells,
        "summary": summary,
        "baseline": cfg.baseline,
        "references": [m for m in cfg.reference_models() if m.published],
    }


def render_html(
    cfg: BenchConfig,
    col: Collected,
    tracker: dict[str, dict[str, str]],
    susp: list[dict[str, str]],
    failures: int = 25,
) -> str:
    """A self-contained report.html with the panel's look (styles and script inlined)."""
    from benchlab.ui import render

    env = render.standalone_env()
    return env.get_template("report.html").render(
        cmp=comparison(cfg, col, failures=failures),
        susp=susp,
        tracker=tracker,
        cfg=cfg,
        col=col,
        live=False,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
        css=(render.STATIC / "app.css").read_text(encoding="utf-8"),
        js=(render.STATIC / "app.js").read_text(encoding="utf-8"),
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
@dataclass
class ReportOutput:
    html_path: Path
    tracker_path: Path
    tracker: dict[str, dict[str, str]]
    suspicious: list[dict[str, str]]
    collected: Collected


def build_report(
    cfg: BenchConfig, out_html: str | Path = "report.html", include_smoke: bool = True
) -> ReportOutput:
    col = collect(cfg, include_smoke=include_smoke)
    tracker = build_tracker(cfg, col)
    susp = suspicious(cfg, col)
    results_dir = Path(cfg.settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tpath = results_dir / "tracker.json"
    tpath.write_text(json.dumps(tracker, indent=2, ensure_ascii=False) + "\n")
    (results_dir / "suspicious.json").write_text(
        json.dumps(susp, indent=2, ensure_ascii=False) + "\n"
    )
    hpath = Path(out_html)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    hpath.write_text(render_html(cfg, col, tracker, susp), encoding="utf-8")
    return ReportOutput(hpath, tpath, tracker, susp, col)
