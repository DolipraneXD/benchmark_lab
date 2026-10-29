"""TTFT probe for every model (API ones too): a small fixed prompt set, streamed
through the LiteLLM layer, time-to-first-token p50/p95 in seconds.

Not a harness benchmark: it is BenchLab's own measurement, deliberately tiny and
identical for every model (same prompts, same order, same max_tokens, one
untimed warm-up request). The tracker key is ``<id>__ttft`` = p50 TTFT seconds.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import MissingKeyError, ModelSpec
from benchlab.llm import LLMClient
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

# Fixed prompt set: short conversational turns of the kind a voice assistant sees.
PROMPTS = [
    "What time zone is Paris in?",
    "Give me one sentence about the water cycle.",
    "Translate 'good morning' into Spanish.",
    "Name three primary colors.",
    "What is 17 times 6?",
    "Suggest a quick breakfast idea.",
    "Summarize the rules of tic-tac-toe in two sentences.",
    "What's the capital of Canada?",
    "Write a one-line birthday wish for a colleague.",
    "Explain what an API is in one sentence.",
]
MAX_TOKENS = 64
SYSTEM = "You are a concise assistant. Answer briefly."


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[k]


class TTFTAdapter(BenchAdapter):
    key = "ttft"
    display = "TTFT probe"
    stage = 1
    needs_proxy = False  # goes through the LiteLLM SDK directly, logged to usage.jsonl
    needs_docker = False
    tokens_in_per_task = 60
    tokens_out_per_task = MAX_TOKENS

    def prompts(self, smoke: bool) -> list[str]:
        n = 3 if smoke else self.cfg.settings.ttft_prompts
        return PROMPTS[: max(1, min(n, len(PROMPTS)))]

    def prepare(self) -> dict[str, Any]:
        return {
            "name": "benchlab-ttft",
            "prompts": len(PROMPTS),
            "repeats": self.cfg.settings.ttft_repeats,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "warmup": 1,
        }

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        p = self.prompts(smoke)
        reps = 1 if smoke else self.cfg.settings.ttft_repeats
        return RunPlan(
            len(p),
            reps,
            None,
            self.tokens_in_per_task,
            self.tokens_out_per_task,
            [
                f"{len(p)} fixed prompts x {reps} streamed repeats + 1 warm-up, max_tokens {MAX_TOKENS}"
            ],
        )

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        try:
            model.api_key()
        except MissingKeyError as e:
            raise HarnessNotReady(str(e)) from e
        prompts = self.prompts(smoke)
        reps = 1 if smoke else self.cfg.settings.ttft_repeats
        client = LLMClient(self.cfg.settings, usage_log=run_dir.usage_log)
        done = run_dir.completed_ids()
        rows: list[dict[str, Any]] = []
        raw = run_dir.raw_dir
        samples_path = raw / "samples.jsonl"
        if samples_path.exists():
            rows = [
                json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()
            ]

        def msgs(p: str):
            return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]

        async def go():
            # warm-up (connection, model load on self-hosted boxes), untimed
            await client.acomplete(
                model,
                msgs("Say hi."),
                stream=True,
                max_tokens=8,
                call_type="stream",
                tags={"ttft": "warmup"},
            )
            for i, p in enumerate(prompts):
                for r in range(reps):
                    tid = f"p{i}r{r}"
                    if tid in done:
                        continue
                    res = await client.acomplete(
                        model,
                        msgs(p),
                        stream=True,
                        max_tokens=MAX_TOKENS,
                        call_type="stream",
                        tags={"ttft": tid},
                    )
                    row = {
                        "id": tid,
                        "prompt": i,
                        "repeat": r,
                        "ok": res.ok,
                        "status": res.status,
                        "ttft_s": res.ttft_s,
                        "latency_s": res.latency_s,
                        "tokens_out": res.tokens_out,
                        "error": res.error_type,
                    }
                    rows.append(row)
                    with open(samples_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    run_dir.mark_done(tid, ok=res.ok)
                    if budget is not None:
                        budget.add(res.cost_usd)

        asyncio.run(go())
        return raw

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        p = run_dir.raw_dir / "samples.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"no TTFT samples at {p}")
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        ok = [r for r in rows if r.get("ok") and r.get("ttft_s") is not None]
        ttfts = [float(r["ttft_s"]) for r in ok]
        errors = ErrorCounts()
        for r in rows:
            if r.get("ok"):
                continue
            st = r.get("status")
            if st == "timeout":
                errors.timeouts += 1
            elif st == "rate_limit":
                errors.rate_limits += 1
            else:
                errors.api_errors += 1
        # tokens/s per request (decode rate after the first token)
        tps = []
        for r in ok:
            dec = float(r["latency_s"]) - float(r["ttft_s"])
            if r.get("tokens_out") and dec > 0:
                tps.append((int(r["tokens_out"]) - 1) / dec)
        metrics: dict[str, float | None] = {
            "ttft_p50_s": pct(ttfts, 0.5),
            "ttft_p95_s": pct(ttfts, 0.95),
            "ttft_mean_s": statistics.fmean(ttfts) if ttfts else None,
            "latency_p50_s": pct([float(r["latency_s"]) for r in ok], 0.5),
            "tokens_per_s_p50": pct(tps, 0.5),
            "n_samples": float(len(rows)),
            "n_ok": float(len(ok)),
        }
        usage = run_dir.usage_log.summarize(cfg["model_id"])
        warnings = []
        if rows and len(ok) / len(rows) < 0.9:
            warnings.append(f"only {len(ok)}/{len(rows)} streamed requests succeeded")
        if ttfts and pct(ttfts, 0.95) and pct(ttfts, 0.95) > 3 * (pct(ttfts, 0.5) or 1e-9):
            warnings.append("TTFT p95 > 3x p50: unstable endpoint or cold starts")
        model_spec = None
        try:
            model_spec = self.cfg.get(cfg["model_id"])
        except Exception:  # noqa: BLE001
            pass
        tracker_id = model_spec.tracker if model_spec else cfg["model_id"]
        note = (
            f"benchlab TTFT probe: {len({r['prompt'] for r in rows})} fixed prompts x "
            f"{max((r['repeat'] for r in rows), default=0) + 1} streamed repeats, max_tokens {MAX_TOKENS}, "
            f"1 warm-up excluded, p50 seconds (p95 in result.json)"
            f"{'; via ' + model_spec.base_url if model_spec and model_spec.base_url else ''}"
        )
        if cfg.get("smoke"):
            note += "; SMOKE"
        tracker = {}
        if metrics["ttft_p50_s"] is not None:
            tracker[f"{tracker_id}__ttft"] = {"v": f"{metrics['ttft_p50_s']:.2f}", "note": note}
        return NormalizedResult(
            bench=self.key,
            model_id=cfg["model_id"],
            metrics=metrics,
            primary_metric="ttft_p50_s",
            n_tasks=len({r["prompt"] for r in rows}),
            trials=max((r["repeat"] for r in rows), default=0) + 1,
            errors=errors,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=usage.cost_usd if (usage.n_requests and usage.cost_known) else None,
            wall_time_s=run_dir.state().get("elapsed_s", 0.0)
            or time.time() - run_dir.state().get("started_at", time.time()),
            harness=cfg.get("harness") or {},
            note=note,
            per_task=rows,
            tracker=tracker,
            warnings=warnings,
        )
