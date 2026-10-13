"""Runs inside the *retrieval harness venv* (mteb). No benchlab imports here.

Usage: python retrieval_runner.py <job.json>

The job describes the embedding model, an optional reranker, the mteb tasks (or
the user's own retrieval data) and the output folders. Stage 1 evaluates the
embedding model and saves its top-k predictions; stage 2 (if a reranker is
given) re-ranks those top-k with the cross-encoder through mteb's own
``convert_to_reranking`` path. All scores come from mteb's retrieval evaluator
(pytrec_eval nDCG@k / recall@k). Results are written to ``<out>/summary.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def build_encoder(spec: dict):
    from mteb.models import ModelMeta, OpenAIAPIEncodeWrapper, SentenceTransformerEncoderWrapper

    kind = spec["kind"]
    if kind == "test_hash":  # deterministic test double (unit tests only)
        return HashEncoder(spec["name"])
    if kind == "local":
        model = SentenceTransformerEncoderWrapper(
            spec["model"], model_prompts=spec.get("prompts") or None
        )
    else:
        model = OpenAIAPIEncodeWrapper(
            endpoint_url=spec["endpoint_url"],
            model_name=spec["model"],
            api_key=os.environ.get(spec.get("api_key_env") or "", None) or None,
            prompt_dict=spec.get("prompts") or None,
            timeout=int(spec.get("timeout", 300)),
        )
    # a stable name/revision so mteb's result cache keys the run by benchlab id
    try:
        model.mteb_model_meta = ModelMeta.create_empty(
            {"name": spec["name"], "revision": spec.get("revision", "benchlab")}
        )
    except Exception:  # noqa: BLE001
        pass
    return model


def build_reranker(spec: dict):
    from mteb.models import CrossEncoderWrapper, ModelMeta, OpenAIAPIRerankWrapper

    kind = spec["kind"]
    if kind == "test_hash":
        return HashCrossEncoder(spec["name"])
    if kind == "local":
        model = CrossEncoderWrapper(spec["model"])
    else:
        model = OpenAIAPIRerankWrapper(
            endpoint_url=spec["endpoint_url"],
            model_name=spec["model"],
            api_key=os.environ.get(spec.get("api_key_env") or "", None) or None,
            timeout=int(spec.get("timeout", 300)),
        )
    try:
        model.mteb_model_meta = ModelMeta.create_empty(
            {"name": spec["name"], "revision": spec.get("revision", "benchlab")}
        )
    except Exception:  # noqa: BLE001
        pass
    return model


class HashEncoder:
    """Bag-of-words hashing embedder; only for tests of the pipeline itself."""

    def __init__(self, name: str):
        from mteb.models import ModelMeta

        self.mteb_model_meta = ModelMeta.create_empty({"name": name, "revision": "test"})

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kw):
        out = []
        for batch in inputs:
            for t in batch["text"]:
                v = np.zeros(256, dtype=np.float32)
                for w in str(t).lower().split():
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
                out.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(out)

    def similarity(self, a, b):
        return np.asarray(a) @ np.asarray(b).T

    def similarity_pairwise(self, a, b):
        return (np.asarray(a) * np.asarray(b)).sum(axis=1)


class HashCrossEncoder:
    def __init__(self, name: str):
        from mteb.models import ModelMeta

        self.mteb_model_meta = ModelMeta.create_empty({"name": name, "revision": "test"})

    def predict(
        self, inputs1, inputs2, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kw
    ):
        q = [t for b in inputs1 for t in b["text"]]
        d = [t for b in inputs2 for t in b["text"]]
        scores = []
        for a, b in zip(q, d, strict=True):
            sa, sb = set(str(a).lower().split()), set(str(b).lower().split())
            scores.append(len(sa & sb) / (len(sa) or 1))
        return np.array(scores, dtype=np.float32)


# --------------------------------------------------------------------------- #
# own-data task
# --------------------------------------------------------------------------- #
def make_own_task(queries_path: str, docs_dir: str, name: str = "BenchLabOwn"):
    from mteb import TaskMetadata
    from mteb.abstasks import AbsTaskRetrieval

    class OwnRetrieval(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=name,
            description="BenchLab: user-supplied queries.jsonl + docs/ retrieval set",
            type="Retrieval",
            dataset={"path": "local/benchlab-own", "revision": "local"},
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="ndcg_at_10",
            reference=None,
            date=None,
            domains=None,
            license=None,
        )

        def load_data(self, **kwargs):
            if self.data_loaded:
                return
            from datasets import Dataset

            corpus_ids, titles, texts = [], [], []
            ddir = Path(docs_dir)
            jsonl = ddir / "docs.jsonl"
            if jsonl.exists():  # {"id","text"[,"title"]} per line
                for line in jsonl.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    corpus_ids.append(str(o["id"]))
                    titles.append(str(o.get("title") or ""))
                    texts.append(str(o["text"]))
            else:  # one file per document, id = file name without extension
                for p in sorted(ddir.iterdir()):
                    if p.is_file() and not p.name.startswith("."):
                        corpus_ids.append(p.stem)
                        titles.append("")
                        texts.append(p.read_text(encoding="utf-8", errors="replace"))
            qids, qtexts, qrels = [], [], {}
            for i, line in enumerate(Path(queries_path).read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                o = json.loads(line)
                qid = str(o.get("id", f"q{i}"))
                qids.append(qid)
                qtexts.append(str(o["query"]))
                qrels[qid] = {str(d): 1 for d in o["relevant_doc_ids"]}
            missing = sorted({d for rel in qrels.values() for d in rel} - set(corpus_ids))
            if missing:
                raise ValueError(
                    f"queries.jsonl references {len(missing)} doc ids not in docs/: {missing[:10]}"
                )
            self.dataset = {
                "default": {
                    "test": {
                        "corpus": Dataset.from_dict(
                            {"id": corpus_ids, "title": titles, "text": texts}
                        ),
                        "queries": Dataset.from_dict({"id": qids, "text": qtexts}),
                        "relevant_docs": qrels,
                        "top_ranked": None,
                        "instructions": None,
                    }
                }
            }
            self.data_loaded = True

    return OwnRetrieval


def get_tasks(job: dict):
    import mteb

    if job["mode"] == "own":
        cls = make_own_task(job["own"]["queries"], job["own"]["docs"])
        return [cls()], lambda: [cls()]
    names = job["tasks"]
    return list(mteb.get_tasks(tasks=names)), lambda: list(mteb.get_tasks(tasks=names))


def pick_scores(task_result) -> dict:
    """Flatten a TaskResult into {split: {metric: value}} (main_score, ndcg_at_k, recall_at_k, map...)."""
    out = {}
    for split, entries in task_result.scores.items():
        for e in entries:
            subset = e.get("hf_subset", "default")
            key = split if subset in ("default", None) else f"{split}/{subset}"
            out[key] = {k: v for k, v in e.items() if isinstance(v, (int, float))}
    return out


def main(job_path: str) -> int:
    import mteb

    job = json.loads(Path(job_path).read_text())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mteb_version": mteb.__version__,
        "mode": job["mode"],
        "stages": {},
        "errors": [],
        "started": time.time(),
    }

    encoder = build_encoder(job["embedding"])
    tasks, fresh_tasks = get_tasks(job)
    summary["tasks"] = [t.metadata.name for t in tasks]
    cache1 = mteb.ResultCache(cache_path=str(out_dir / "cache_embed"))
    pred = out_dir / "predictions"
    t0 = time.time()
    res1 = mteb.evaluate(
        encoder,
        tasks,
        prediction_folder=str(pred),
        cache=cache1,
        show_progress_bar=False,
        co2_tracker=False,
        raise_error=False,
        overwrite_strategy="only-missing",
    )
    summary["stages"]["embed"] = {
        "seconds": time.time() - t0,
        "tasks": {tr.task_name: pick_scores(tr) for tr in res1.task_results},
        "exceptions": [str(e) for e in (res1.exceptions or [])]
        if hasattr(res1, "exceptions")
        else [],
    }

    rr = job.get("reranker")
    if rr:
        reranker = build_reranker(rr)
        top_k = int(job.get("rerank_top_k", 100))
        t0 = time.time()
        if job["mode"] == "reranking_tasks":
            # mteb reranking tasks carry their own candidates: score them with the cross-encoder directly
            tasks2 = fresh_tasks()
        else:
            tasks2 = []
            for t in fresh_tasks():
                try:
                    tasks2.append(t.convert_to_reranking(str(pred), top_k=top_k))
                except Exception as e:  # noqa: BLE001
                    summary["errors"].append(
                        f"{t.metadata.name}: no stage-1 predictions to rerank ({e})"
                    )
        if tasks2:
            cache2 = mteb.ResultCache(cache_path=str(out_dir / "cache_rerank"))
            res2 = mteb.evaluate(
                reranker,
                tasks2,
                cache=cache2,
                show_progress_bar=False,
                co2_tracker=False,
                raise_error=False,
                overwrite_strategy="only-missing",
            )
            summary["stages"]["rerank"] = {
                "seconds": time.time() - t0,
                "top_k": top_k,
                "tasks": {tr.task_name: pick_scores(tr) for tr in res2.task_results},
                "exceptions": [str(e) for e in (res2.exceptions or [])]
                if hasattr(res2, "exceptions")
                else [],
            }
    summary["finished"] = time.time()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
