"""Stage 8: ``benchlab report`` -> ``report.html`` + ``results/tracker.json``.

Reads every finished run under ``runs/`` (the newest non-smoke run per
(model, bench) wins; smoke runs are used only when nothing else exists and are
flagged), merges the adapters' tracker entries, derives the aggregate keys the
tracker expects (``cost``, ``gpt_sanity``, ``s5_api``/``s5_self``/``s5_tok``,
``ds_cps``/``ds_lps``, ``srv_match``, ``claude__coder_*``) and renders one
static HTML page: models x benchmarks with the delta against the baseline,
cost per task, serving / TTFT / retrieval tables and a "suspicious results"
list. Nothing here re-scores anything: values come straight from result.json.
"""

from __future__ import annotations

import html
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
CODER_BENCHES = ("lcb", "aider", "swe", "tb")
NON_SCORE_BENCHES = {"ttft", "serving"}  # seconds / tokens per second, not percentages
TAU2_BENCHES = ("t2air", "t2ret", "t2tel")
TOOL_BENCHES = (
    "bfcl",
    "t2air",
    "t2ret",
    "t2tel",
)  # where a self-hosted gap usually means a chat-template/tool-parser problem
COMPUTED_KEYS = {
    "cost",
    "gpt_sanity",
    "srv_match",
    "s5_api",
    "s5_self",
    "s5_tok",
    "ds_cps",
    "ds_lps",
}  # derived here


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
    baseline = cfg.baseline
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
            if m.id == "step5self":
                pass  # shares tracker id with step5; keep the API cost (priced) rather than overwrite
            else:
                tracker[f"{tid}__cost"] = {"v": f"{total:.2f}", "note": note}
        # 3. coder aliases for the baseline: claude__coder_lcb etc.
        if m.id == baseline.id:
            for b in CODER_BENCHES:
                src = tracker.get(f"{tid}__{b}")
                if src:
                    tracker[f"{tid}__coder_{b}"] = dict(src)
        # 4. sanity vs published (gpt_sanity for gptoss; any model with `published`)
        if "gpt_sanity" in m.extras:
            tracker[f"{tid}__gpt_sanity"] = _sanity_entry(cfg, col, m)
        # 5. self-hosted vs API counterpart: <id>__srv_match
        if m.self_hosted:
            entry = _srv_match_entry(cfg, col, m)
            if entry:
                tracker[f"{tid}__srv_match"] = entry
        # 6. cost / latency per solved task (ds_cps, ds_lps)
        if "ds_cps" in m.extras or "ds_lps" in m.extras:
            cps, lps = _per_solved(col, m)
            if "ds_cps" in m.extras:
                tracker[f"{tid}__ds_cps"] = cps
            if "ds_lps" in m.extras:
                tracker[f"{tid}__ds_lps"] = lps
    # 7. Step 5: API vs self-hosted aggregates
    api = next((m for m in cfg.chat_models() if "s5_api" in m.extras), None)
    own = next((m for m in cfg.chat_models() if "s5_self" in m.extras), None)
    if api:
        tracker[f"{api.tracker}__s5_api"] = _core_mean_entry(col, api, own)
    if own:
        tracker[f"{own.tracker}__s5_self"] = _core_mean_entry(col, own, api)
    if api and own:
        tracker[f"{own.tracker}__s5_tok"] = _tokens_per_task_entry(col, own, api)
    elif own or api:
        mm = own or api
        tracker[f"{mm.tracker}__s5_tok"] = _tokens_per_task_entry(col, mm, None)
    # 8. benches without an integration: explicit n/a so the tracker is complete
    known = {m.id for m in cfg.models}
    for b in BENCHES.values():
        if b.adapter is None and b.key not in COMPUTED_KEYS:
            for mid in b.only_models:
                if mid not in known:
                    continue
                k = f"{_tracker_id(cfg, mid)}__{b.tracker_key}"
                if k not in tracker:
                    tracker[k] = {"v": "n/a", "note": f"not integrated: {b.notes or b.harness}"}
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


def _core_mean_entry(col: Collected, m: ModelSpec, other: ModelSpec | None) -> dict[str, str]:
    benches = _common_core(col, m, other)
    if not benches:
        return {
            "v": "n/a",
            "note": "no core results yet" + (f" shared with {other.id}" if other else ""),
        }
    vals = [_score(col, m.id, b) for b in benches]
    mean = sum(vals) / len(vals)
    note = f"mean of core scores over {benches}: " + ", ".join(
        f"{b} {v:.1f}" for b, v in zip(benches, vals, strict=True)
    )
    if other:
        ov = [_score(col, other.id, b) for b in benches]
        om = sum(ov) / len(ov)
        note += f"; {other.id} mean {om:.1f} on the same benches (gap {mean - om:+.1f})"
    return {"v": _f1(mean), "note": note}


def _tokens_per_task_entry(col: Collected, m: ModelSpec, other: ModelSpec | None) -> dict[str, str]:
    benches = _common_core(col, m, other)
    es = [col.get(m.id, b) for b in benches]
    es = [e for e in es if e and e.n_tasks]
    if not es:
        return {"v": "n/a", "note": "no core results yet"}
    tot = sum(int(e.result.get("tokens_out") or 0) for e in es)
    n = sum(e.n_tasks * e.trials for e in es)
    v = tot / n
    note = f"output tokens per task-attempt over {benches} ({m.id})"
    if other:
        oes = [col.get(other.id, b) for b in benches]
        oes = [e for e in oes if e and e.n_tasks]
        if oes:
            ov = sum(int(e.result.get("tokens_out") or 0) for e in oes) / sum(
                e.n_tasks * e.trials for e in oes
            )
            note += f"; {other.id}: {ov:.0f} tokens/task (ratio {v / ov:.2f}x)" if ov else ""
    return {"v": f"{v:.0f}", "note": note}


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


def _per_solved(col: Collected, m: ModelSpec) -> tuple[dict[str, str], dict[str, str]]:
    es = [col.get(m.id, b) for b in CORE_SCORE_BENCHES]
    es = [e for e in es if e and e.solved is not None]
    solved = sum(e.solved for e in es)
    benches = [e.bench for e in es]
    if not es or solved <= 0:
        na = {"v": "n/a", "note": "no solved core tasks yet"}
        return na, na
    priced = [e for e in es if e.cost is not None]
    if priced:
        ps = sum(e.solved for e in priced)
        cps = {
            "v": f"{sum(e.cost for e in priced) / ps:.4f}",
            "note": f"USD per solved task-attempt over {[e.bench for e in priced]} (cost from usage logs / estimated solved = n*trials*score)",
        }
    else:
        cps = {"v": "n/a", "note": "no priced runs (set price_in/out_per_mtok)"}
    wall = sum(float(e.result.get("wall_time_s") or 0) for e in es)
    lps = {
        "v": f"{wall / solved:.1f}",
        "note": f"wall seconds per solved task-attempt over {benches} (run wall time / estimated solved)",
    }
    return cps, lps


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
        if m and m.has_prices and e.cost is None and bench not in NON_SCORE_BENCHES and e.n_tasks:
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
# HTML
# --------------------------------------------------------------------------- #
CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1b1b1b;background:#fff}
h1{margin:0 0 4px} h2{margin:32px 0 8px;font-size:1.15rem} .muted{color:#666;font-size:.9rem}
table{border-collapse:collapse;font-size:.85rem;margin-top:8px} th,td{border:1px solid #ddd;padding:4px 8px;text-align:right;vertical-align:top}
th{background:#f3f3f3;position:sticky;top:0} td:first-child,th:first-child{text-align:left;font-weight:600;white-space:nowrap}
.pos{color:#0a7a2f} .neg{color:#b3261e} .sub{display:block;color:#777;font-size:.72rem;font-weight:400}
.base{background:#fbfbe6} .smoke{outline:2px dashed #e0a800} .na{color:#bbb}
ul.susp li{margin:4px 0} .hint{color:#555} code{background:#f3f3f3;padding:0 3px}
.wrap{overflow-x:auto}
"""


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _delta_cell(
    v: float | None, base: float | None, lower_better: bool = False, nd: int = 1
) -> str:
    if v is None:
        return '<span class="na">-</span>'
    if base is None:
        return f"{v:.{nd}f}"
    d = v - base
    good = d < 0 if lower_better else d > 0
    cls = "pos" if good else ("neg" if d != 0 else "")
    return f'{v:.{nd}f} <span class="{cls}">({d:+.{nd}f})</span>'


def _cell(
    e: Entry | None,
    base: Entry | None,
    lower_better: bool = False,
    nd: int = 1,
    baseline: bool = False,
) -> str:
    if e is None:
        return '<td class="na">-</td>'
    cls = []
    if baseline:
        cls.append("base")
    if e.smoke:
        cls.append("smoke")
    v = e.primary
    body = _delta_cell(v, None if baseline else (base.primary if base else None), lower_better, nd)
    sub = f"n={e.n_tasks}" + (f"×{e.trials}" if e.trials > 1 else "")
    if e.cost is not None:
        sub += f" ${e.cost:.2f}"
    if e.error_total:
        sub += f" err={e.error_total}"
    if e.smoke:
        sub += " SMOKE"
    return f'<td class="{" ".join(cls)}">{body}<span class="sub">{_esc(sub)}</span></td>'


def render_html(
    cfg: BenchConfig, col: Collected, tracker: dict[str, dict[str, str]], susp: list[dict[str, str]]
) -> str:
    baseline = cfg.baseline
    chat = cfg.chat_models()
    benches = col.benches_with_results()
    score_benches = [
        b
        for b in benches
        if b not in NON_SCORE_BENCHES
        and BENCHES.get(b, BenchInfo(b, b, 4, (), b, "")).suites != ("retrieval",)
        and b not in ("mteb_ret", "mteb_rr", "beir", "own")
    ]
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    H: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>BenchLab report</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>BenchLab report</h1><div class='muted'>generated {now} · benchlab {__version__} · baseline <b>{_esc(baseline.name)}</b> ({_esc(baseline.id)}) · "
        f"{len(col.entries)} (model, bench) results from {len(col.all_runs)} finished runs in <code>{_esc(cfg.settings.runs_dir)}</code>. "
        "Cells: score (delta vs baseline in points); n = tasks×trials, $ = run cost from the usage logs, err = timeouts+API errors+malformed tool calls. "
        "Dashed outline = smoke run only.</div>",
    ]
    # ---- main table
    H.append("<h2>Scores: models × benchmarks</h2><div class='wrap'><table><tr><th>model</th>")
    for b in score_benches:
        bi = BENCHES.get(b)
        H.append(
            f"<th title='{_esc(bi.harness if bi else '')}'>{_esc(bi.display if bi else b)}<span class='sub'>{_esc(b)}</span></th>"
        )
    H.append("<th>TTFT p50 s</th><th>core cost $</th></tr>")
    for m in chat:
        is_base = m.id == baseline.id
        H.append(
            f"<tr><td{' class=base' if is_base else ''}>{_esc(m.name)}<span class='sub'>{_esc(m.id)}{' · self-hosted' if m.self_hosted else ''}</span></td>"
        )
        for b in score_benches:
            H.append(_cell(col.get(m.id, b), col.get(baseline.id, b), baseline=is_base))
        H.append(
            _cell(
                col.get(m.id, "ttft"),
                col.get(baseline.id, "ttft"),
                lower_better=True,
                nd=2,
                baseline=is_base,
            )
        )
        cost = tracker.get(f"{m.tracker}__cost", {}).get("v") if not m.compare_to else None
        H.append(f"<td>{_esc(cost) if cost else '<span class=na>-</span>'}</td></tr>")
    H.append("</table></div>")
    # ---- cost per task
    H.append(
        "<h2>Cost per task (USD per task-attempt) and tokens</h2><div class='wrap'><table><tr><th>model</th>"
    )
    for b in score_benches:
        H.append(f"<th>{_esc(b)}</th>")
    H.append("</tr>")
    for m in chat:
        H.append(f"<tr><td>{_esc(m.id)}</td>")
        for b in score_benches:
            e = col.get(m.id, b)
            if e is None or e.cost_per_task is None:
                H.append("<td class=na>-</td>")
            else:
                tok = (
                    int(e.result.get("tokens_in") or 0) + int(e.result.get("tokens_out") or 0)
                ) / max(1, e.n_tasks * e.trials)
                H.append(
                    f"<td>{e.cost_per_task:.4f}<span class='sub'>{tok:,.0f} tok/task</span></td>"
                )
        H.append("</tr>")
    H.append("</table></div>")
    # ---- serving
    srv = [(m, col.get(m.id, "serving")) for m in chat if col.get(m.id, "serving")]
    if srv:
        H.append(
            "<h2>Serving (self-hosted)</h2><div class='wrap'><table><tr><th>model</th><th>quant</th><th>GPUs</th><th>GPU mem GB</th><th>TTFT p50 s (c=1)</th><th>tok/s per request</th><th>total tok/s c=1</th><th>c=8</th><th>c=32</th><th>vs API (pts)</th><th>tool</th></tr>"
        )
        for m, e in srv:

            def t(k: str, tid: str = m.tracker) -> str:
                return _esc(tracker.get(f"{tid}__{k}", {}).get("v", "-"))

            H.append(
                f"<tr><td>{_esc(m.id)}</td><td>{t('srv_quant')}</td><td>{t('srv_gpus')}</td><td>{t('srv_mem')}</td><td>{t('srv_ttft')}</td><td>{t('srv_tps')}</td><td>{t('srv_c1')}</td><td>{t('srv_c8')}</td><td>{t('srv_c32')}</td><td>{t('srv_match')}</td><td>{_esc((e.result.get('harness') or {}).get('name', ''))}</td></tr>"
            )
        H.append("</table></div>")
    # ---- retrieval
    ret = [
        (m, b, col.get(m.id, b))
        for m in cfg.models
        if m.kind != "chat"
        for b in ("mteb_ret", "mteb_rr", "beir", "own")
        if col.get(m.id, b)
    ]
    if ret:
        H.append(
            "<h2>Retrieval</h2><div class='wrap'><table><tr><th>model</th><th>bench</th><th>metrics</th><th>tracker</th></tr>"
        )
        for m, b, e in ret:
            mets = ", ".join(
                f"{k}={v:.1f}"
                for k, v in e.result["metrics"].items()
                if isinstance(v, (int, float))
            )
            tk = ", ".join(f"{k}={v['v']}" for k, v in (e.result.get("tracker") or {}).items())
            H.append(
                f"<tr><td>{_esc(m.id)}</td><td>{_esc(b)}</td><td style='text-align:left'>{_esc(mets)}</td><td style='text-align:left'>{_esc(tk)}</td></tr>"
            )
        H.append("</table></div>")
    # ---- suspicious
    H.append(f"<h2>Suspicious results ({len(susp)})</h2>")
    if susp:
        H.append("<ul class='susp'>")
        for s in susp:
            H.append(
                f"<li><b>{_esc(s['model'])}</b> / {_esc(s['bench'])}: {_esc(s['why'])}"
                + (f" <span class='hint'>→ {_esc(s['hint'])}</span>" if s["hint"] else "")
                + (f" <span class='muted'>[{_esc(s['run'])}]</span>" if s["run"] else "")
                + "</li>"
            )
        H.append("</ul>")
    else:
        H.append("<div class='muted'>nothing flagged</div>")
    # ---- derived tracker keys
    derived = {
        k: v
        for k, v in tracker.items()
        if k.split("__", 1)[-1]
        in ("cost", "gpt_sanity", "srv_match", "s5_api", "s5_self", "s5_tok", "ds_cps", "ds_lps")
    }
    if derived:
        H.append(
            "<h2>Derived tracker keys</h2><div class='wrap'><table><tr><th>key</th><th>value</th><th>note</th></tr>"
        )
        for k, v in derived.items():
            H.append(
                f"<tr><td>{_esc(k)}</td><td>{_esc(v['v'])}</td><td style='text-align:left'>{_esc(v['note'])}</td></tr>"
            )
        H.append("</table></div>")
    # ---- run index
    H.append(
        "<h2>Runs used</h2><div class='wrap'><table><tr><th>model</th><th>bench</th><th>run</th><th>date</th><th>harness</th><th>primary</th><th>note</th></tr>"
    )
    for (mid, b), e in sorted(col.entries.items()):
        h = e.result.get("harness") or {}
        hv = h.get("version") or h.get("commit") or h.get("guidellm") or h.get("tag") or ""
        H.append(
            f"<tr><td>{_esc(mid)}</td><td>{_esc(b)}</td><td style='text-align:left'><code>{_esc(e.run.name)}</code></td><td>{_esc(e.date)}</td><td style='text-align:left'>{_esc(h.get('name') or h.get('package') or '')} {_esc(hv)}</td><td>{'-' if e.primary is None else f'{e.primary:.2f}'}</td><td style='text-align:left'>{_esc(e.result.get('note', ''))}</td></tr>"
        )
    H.append("</table></div></body></html>")
    return "\n".join(H)


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
