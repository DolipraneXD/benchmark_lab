"""Retrieval pipeline: MTEB retrieval/reranking subsets, BEIR (± reranker) and the
user's own data - all scored by ``mteb`` (pinned 2.24.12) in its own venv.

Verified against mteb 2.24.12: ``mteb.get_tasks``/``mteb.evaluate`` with a
``ResultCache`` in the run folder (resume at task granularity),
``OpenAIAPIEncodeWrapper`` (``<host>/v1/embeddings``) and
``OpenAIAPIRerankWrapper`` (``<host>/v1/rerank``) for vLLM/TEI endpoints,
``SentenceTransformerEncoderWrapper`` / ``CrossEncoderWrapper`` for local
weights, and ``AbsTaskRetrieval.convert_to_reranking(predictions, top_k)`` to
re-rank the embedding stage's top-k with the cross-encoder. BEIR is mteb's
``BEIR`` benchmark (same datasets/qrels, nDCG@10). The own-data set is a
custom ``AbsTaskRetrieval`` built from ``data/retrieval/queries.jsonl`` +
``retrieval/docs/`` so it goes through exactly the same evaluator.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from benchlab.adapters._proc import run_logged
from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import ModelSpec
from benchlab.harness_env import HarnessInstallError, ensure_venv, frozen_versions
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

HARNESS_NAME = "retrieval"
MTEB_VERSION = "2.24.12"
PACKAGES = [f"mteb=={MTEB_VERSION}"]
RUNNER = Path(__file__).with_name("retrieval_runner.py")
SMOKE_RETRIEVAL_TASKS = ["NanoSciFactRetrieval", "NanoNFCorpusRetrieval"]
SMOKE_RERANKING_TASKS = ["SciDocsRR"]
BEIR_ALL = [
    "TRECCOVID",
    "NFCorpus",
    "NQ",
    "HotpotQA",
    "FiQA2018",
    "ArguAna",
    "Touche2020",
    "CQADupstackRetrieval",
    "QuoraRetrieval",
    "DBPedia",
    "SCIDOCS",
    "FEVER",
    "ClimateFEVER",
    "SciFact",
    "MSMARCO",
]


def endpoint_root(base_url: str) -> str:
    """mteb's OpenAI wrappers append ``/v1/embeddings`` themselves: strip a trailing /v1."""
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def model_job_spec(spec: ModelSpec) -> dict[str, Any]:
    prompts = (
        spec.extra_params.get("prompts")
        if isinstance(spec.extra_params.get("prompts"), dict)
        else None
    )
    if spec.base_url == "local":
        return {"kind": "local", "name": spec.id, "model": spec.model, "prompts": prompts}
    return {
        "kind": "openai",
        "name": spec.id,
        "model": spec.model,
        "endpoint_url": endpoint_root(spec.base_url or ""),
        "api_key_env": spec.api_key_env,
        "prompts": prompts,
    }


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(100.0 * sum(xs) / len(xs), 4) if xs else None


def task_metric(task_scores: dict[str, dict[str, float]], metric: str) -> float | None:
    """Mean of ``metric`` over the task's evaluated splits/subsets (as a fraction)."""
    vals = [s[metric] for s in task_scores.values() if metric in s]
    return sum(vals) / len(vals) if vals else None


class RetrievalAdapter(BenchAdapter):
    mode: str = ""  # mteb | reranking_tasks | own
    needs_proxy = False
    needs_docker = False
    stage = 4
    tokens_in_per_task = 0
    tokens_out_per_task = 0
    with_reranker: bool = True

    # -- harness -------------------------------------------------------------------
    def harness_bin(self) -> Path:
        try:
            return ensure_venv(HARNESS_NAME, PACKAGES, python="3.12")
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def reranker_spec(self) -> ModelSpec | None:
        rid = self.cfg.settings.reranker_model
        try:
            spec = self.cfg.get(rid)
        except KeyError:
            return None
        return spec if spec.kind == "reranker" else None

    def check_model(self, spec: ModelSpec, want_kind: str) -> None:
        if spec.kind != want_kind:
            raise HarnessNotReady(
                f"{self.key}: model {spec.id} is kind={spec.kind}, expected {want_kind}"
            )
        if spec.base_url != "local" and spec.api_key_env and not os.environ.get(spec.api_key_env):
            raise HarnessNotReady(
                f"{self.key}: env var {spec.api_key_env} for {spec.id} is not set"
            )

    def prepare(self) -> dict[str, Any]:
        bin_dir = self.harness_bin()
        rr = self.reranker_spec()
        info = {
            "name": "mteb",
            "mteb": frozen_versions(
                bin_dir, ["mteb", "sentence-transformers", "pytrec-eval-terrier"]
            ),
            "pinned": MTEB_VERSION,
            "mode": self.mode,
            "tasks": self.tasks(None, False),
            "rerank_top_k": self.cfg.settings.rerank_top_k,
            "reranker": (
                {
                    "id": rr.id,
                    "model": rr.model,
                    "endpoint": "local"
                    if rr.base_url == "local"
                    else endpoint_root(rr.base_url or ""),
                }
                if rr
                else None
            ),
        }
        return info

    # -- tasks -----------------------------------------------------------------------
    def tasks(self, subset: str | None, smoke: bool) -> list[str]:
        raise NotImplementedError

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        t = self.tasks(subset, smoke)
        notes = [f"mteb {MTEB_VERSION} {self.mode}: {', '.join(t) if t else 'own data'}"]
        rr = self.reranker_spec()
        notes.append(
            f"reranker: {rr.id} (top-{self.cfg.settings.rerank_top_k})"
            if (rr and self.with_reranker)
            else "no reranker configured (settings.reranker_model)"
        )
        return RunPlan(max(1, len(t)), 1, t or None, 0, 0, notes)

    # -- run -------------------------------------------------------------------------
    def job(
        self, model: ModelSpec, run_dir: RunDir, subset: str | None, smoke: bool
    ) -> dict[str, Any]:
        rr = self.reranker_spec() if self.with_reranker else None
        job = {
            "mode": self.mode,
            "embedding": model_job_spec(model),
            "reranker": model_job_spec(rr) if rr else None,
            "rerank_top_k": self.cfg.settings.rerank_top_k,
            "tasks": self.tasks(subset, smoke),
            "out_dir": str(run_dir.raw_dir.resolve()),
        }
        if self.mode == "own":
            d = Path(self.cfg.settings.own_retrieval_dir)
            job["own"] = {
                "queries": str((d / "queries.jsonl").resolve()),
                "docs": str((d / "docs").resolve()),
            }
        return job

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        self.check_model(model, "embedding")
        rr = self.reranker_spec()
        if rr and self.with_reranker:
            self.check_model(rr, "reranker")
        if self.mode == "own":
            d = Path(self.cfg.settings.own_retrieval_dir)
            if not (d / "queries.jsonl").exists() or not (d / "docs").is_dir():
                raise HarnessNotReady(
                    f"{self.key}: need {d}/queries.jsonl and {d}/docs/ (see retrieval/README.md)"
                )
        bin_dir = self.harness_bin()
        raw = run_dir.raw_dir
        job = self.job(model, run_dir, subset, smoke)
        (raw / "job.json").write_text(json.dumps(job, indent=2))
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = [str(bin_dir / "python"), str(RUNNER), str(raw / "job.json")]
        (raw / "command.txt").write_text(" ".join(cmd) + "\n")
        run_logged(cmd, env, raw / "retrieval.log", budget=budget, cwd=raw)
        return raw

    # -- parse -----------------------------------------------------------------------
    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        p = run_dir.raw_dir / "summary.json"
        if not p.exists():
            raise FileNotFoundError(f"no summary at {p}")
        return self.normalize(json.loads(p.read_text()), cfg, run_dir)

    def stage_scores(
        self, summary: dict[str, Any], stage: str, metric: str
    ) -> dict[str, float | None]:
        tasks = (summary.get("stages", {}).get(stage) or {}).get("tasks") or {}
        return {name: task_metric(scores, metric) for name, scores in tasks.items()}

    def normalize(
        self, summary: dict[str, Any], cfg: dict[str, Any], run_dir: RunDir
    ) -> NormalizedResult:
        raise NotImplementedError

    def _common(self, summary, cfg, run_dir, metrics, primary, tracker_entries, note_extra=""):
        errors = ErrorCounts()
        warnings = []
        for stage in ("embed", "rerank"):
            exc = (summary.get("stages", {}).get(stage) or {}).get("exceptions") or []
            errors.api_errors += len(exc)
            for e in exc:
                warnings.append(f"{stage}: {str(e)[:200]}")
        for e in summary.get("errors") or []:
            errors.other += 1
            warnings.append(str(e)[:200])
        n_tasks = len(summary.get("tasks") or [])
        h = cfg.get("harness") or {}
        rr = h.get("reranker") or {}
        secs = sum(
            float((summary.get("stages", {}).get(s) or {}).get("seconds") or 0)
            for s in ("embed", "rerank")
        )
        model_id = cfg["model_id"]
        try:
            tracker_id = self.cfg.get(model_id).tracker
        except Exception:  # noqa: BLE001
            tracker_id = model_id
        note = (
            f"mteb {summary.get('mteb_version')} {self.mode}, {n_tasks} task(s): {', '.join(summary.get('tasks') or [])}"
            f"{note_extra}; reranker {rr.get('id') + ' top-' + str(h.get('rerank_top_k')) if rr else 'none'}; "
            f"endpoint called directly (no proxy): cost n/a"
        )
        if cfg.get("smoke"):
            note += "; SMOKE"
        tracker = {}
        for key, value in tracker_entries.items():
            if value is not None:
                tracker[f"{tracker_id}__{key}"] = {"v": f"{value:.1f}", "note": note}
        harness = {
            "name": "mteb",
            "mteb_version": summary.get("mteb_version"),
            "mode": self.mode,
            "tasks": summary.get("tasks"),
            "reranker": rr or None,
            "rerank_top_k": h.get("rerank_top_k"),
            "summary": str(run_dir.raw_dir / "summary.json"),
        }
        return NormalizedResult(
            bench=self.key,
            model_id=model_id,
            metrics=metrics,
            primary_metric=primary,
            n_tasks=n_tasks,
            trials=1,
            errors=errors,
            tokens_in=0,
            tokens_out=0,
            cost_usd=None,
            wall_time_s=secs,
            harness=harness,
            note=note,
            per_task=[],
            tracker=tracker,
            warnings=warnings,
        )


class MTEBRetrievalAdapter(RetrievalAdapter):
    key = "mteb_ret"
    display = "MTEB retrieval subset"
    mode = "mteb"
    with_reranker = False  # the MTEB retrieval number is the embedding model's

    def tasks(self, subset, smoke):
        if smoke:
            return SMOKE_RETRIEVAL_TASKS
        if subset and subset not in ("all",):
            return [t.strip() for t in subset.split("+") if t.strip()]
        return list(self.cfg.settings.mteb_retrieval_tasks)

    def normalize(self, summary, cfg, run_dir):
        per = self.stage_scores(summary, "embed", "ndcg_at_10")
        metrics: dict[str, float | None] = {"ndcg_at_10_mean": _mean(list(per.values()))}
        for t, v in per.items():
            metrics[f"ndcg_at_10/{t}"] = None if v is None else round(100.0 * v, 4)
        return self._common(
            summary,
            cfg,
            run_dir,
            metrics,
            "ndcg_at_10_mean",
            {"mteb_ret": metrics["ndcg_at_10_mean"]},
            note_extra=", mean nDCG@10",
        )


class MTEBRerankingAdapter(RetrievalAdapter):
    key = "mteb_rr"
    display = "MTEB reranking subset"
    mode = "reranking_tasks"

    def tasks(self, subset, smoke):
        if smoke:
            return SMOKE_RERANKING_TASKS
        if subset and subset != "all":
            return [t.strip() for t in subset.split("+") if t.strip()]
        return list(self.cfg.settings.mteb_reranking_tasks)

    def normalize(self, summary, cfg, run_dir):
        emb = self.stage_scores(summary, "embed", "main_score")
        rr = self.stage_scores(summary, "rerank", "main_score")
        metrics: dict[str, float | None] = {
            "main_score_mean": _mean(list(emb.values())),
            "main_score_mean_reranker": _mean(list(rr.values())) if rr else None,
        }
        for t, v in emb.items():
            metrics[f"main_score/{t}"] = None if v is None else round(100.0 * v, 4)
        for t, v in rr.items():
            metrics[f"main_score_reranker/{t}"] = None if v is None else round(100.0 * v, 4)
        entries = {"mteb_rr": metrics["main_score_mean"]}
        res = self._common(
            summary,
            cfg,
            run_dir,
            metrics,
            "main_score_mean",
            entries,
            note_extra=", mean of task main scores (MAP) for the embedding model; cross-encoder result in result.json",
        )
        rspec = self.reranker_spec()
        if rspec and metrics["main_score_mean_reranker"] is not None:
            res.tracker[f"{rspec.tracker}__mteb_rr"] = {
                "v": f"{metrics['main_score_mean_reranker']:.1f}",
                "note": res.note + " (cross-encoder on the task candidates)",
            }
        return res


class BEIRAdapter(RetrievalAdapter):
    key = "beir"
    display = "BEIR nDCG@10 (± reranker)"
    mode = "mteb"

    def tasks(self, subset, smoke):
        if smoke:
            return SMOKE_RETRIEVAL_TASKS
        if subset == "all":
            return list(BEIR_ALL)
        if subset:
            return [t.strip() for t in subset.split("+") if t.strip()]
        return list(self.cfg.settings.beir_tasks)

    def normalize(self, summary, cfg, run_dir):
        emb = self.stage_scores(summary, "embed", "ndcg_at_10")
        rr = self.stage_scores(summary, "rerank", "ndcg_at_10")
        metrics: dict[str, float | None] = {
            "ndcg_at_10_mean": _mean(list(emb.values())),
            "ndcg_at_10_mean_reranked": _mean(list(rr.values())) if rr else None,
        }
        for t, v in emb.items():
            metrics[f"ndcg_at_10/{t}"] = None if v is None else round(100.0 * v, 4)
        for t, v in rr.items():
            metrics[f"ndcg_at_10_reranked/{t}"] = None if v is None else round(100.0 * v, 4)
        return self._common(
            summary,
            cfg,
            run_dir,
            metrics,
            "ndcg_at_10_mean",
            {"beir": metrics["ndcg_at_10_mean"], "beir_rr": metrics["ndcg_at_10_mean_reranked"]},
            note_extra=", mean nDCG@10 (BEIR datasets as packaged by mteb)",
        )


class OwnDataAdapter(RetrievalAdapter):
    key = "own"
    display = "own retrieval data"
    mode = "own"

    def tasks(self, subset, smoke):
        return []

    def normalize(self, summary, cfg, run_dir):
        e = (summary.get("stages", {}).get("embed") or {}).get("tasks") or {}
        r = (summary.get("stages", {}).get("rerank") or {}).get("tasks") or {}
        es = next(iter(e.values()), {})
        rs = next(iter(r.values()), {})

        def pct(scores, m):
            v = task_metric(scores, m)
            return None if v is None else round(100.0 * v, 4)

        metrics = {
            "recall_at_10": pct(es, "recall_at_10"),
            "ndcg_at_10": pct(es, "ndcg_at_10"),
            "recall_at_10_reranked": pct(rs, "recall_at_10"),
            "ndcg_at_10_reranked": pct(rs, "ndcg_at_10"),
            "recall_at_100": pct(es, "recall_at_100"),
        }
        entries = {
            "own_rec": metrics["recall_at_10"],
            "own_ndcg": metrics["ndcg_at_10"],
            "own_rec_rr": metrics["recall_at_10_reranked"],
            "own_ndcg_rr": metrics["ndcg_at_10_reranked"],
        }
        return self._common(
            summary,
            cfg,
            run_dir,
            metrics,
            "ndcg_at_10",
            entries,
            note_extra=f", own data from {self.cfg.settings.own_retrieval_dir}/ (queries.jsonl + docs/)",
        )
