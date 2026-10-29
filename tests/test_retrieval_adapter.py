"""Retrieval pipeline tests: job building, parsing, and (when the harness venv exists)
an end-to-end run of the mteb runner on own data with deterministic test models."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from benchlab.adapters.retrieval import (
    RUNNER,
    BEIRAdapter,
    MTEBRerankingAdapter,
    MTEBRetrievalAdapter,
    OwnDataAdapter,
    endpoint_root,
    model_job_spec,
)
from benchlab.config import load_config
from benchlab.runs import RunConfig, RunDir

VENV_PY = Path("workspace/harnesses/retrieval/.venv/bin/python")


def _cfg_with_reranker(models_yaml, tmp_path):
    text = (
        models_yaml.read_text()
        + """
  - id: qrr
    name: Reranker
    kind: reranker
    provider: openai_compatible
    base_url: http://127.0.0.1:8011/v1/
    model: Qwen/Qwen3-Reranker-4B
    api_key_env: MOCK_API_KEY
"""
    )
    p = tmp_path / "models_rr.yaml"
    p.write_text(text)
    return load_config(p)


def test_endpoint_root_and_job_spec(models_yaml, tmp_path):
    assert (
        endpoint_root("http://h:8010/v1") == "http://h:8010"
        and endpoint_root("http://h:8010/v1/") == "http://h:8010"
    )
    assert endpoint_root("http://h:8010") == "http://h:8010"
    cfg = _cfg_with_reranker(models_yaml, tmp_path)
    emb = model_job_spec(cfg.get("qemb"))
    assert (
        emb["kind"] == "openai"
        and emb["endpoint_url"].endswith("/v1") is False
        and emb["api_key_env"] == "MOCK_API_KEY"
    )
    cfg.get("qemb").base_url = "local"
    assert model_job_spec(cfg.get("qemb"))["kind"] == "local"


def test_tasks_and_job(models_yaml, tmp_path):
    cfg = _cfg_with_reranker(models_yaml, tmp_path)
    cfg.settings.own_retrieval_dir = str(tmp_path / "retrieval")
    ret = MTEBRetrievalAdapter(cfg)
    assert ret.tasks(None, False) == cfg.settings.mteb_retrieval_tasks and ret.tasks(
        None, True
    ) == ["NanoSciFactRetrieval", "NanoNFCorpusRetrieval"]
    assert ret.tasks("SciFact+ArguAna", False) == ["SciFact", "ArguAna"]
    beir = BEIRAdapter(cfg)
    assert (
        len(beir.tasks("all", False)) == 15 and beir.tasks(None, False) == cfg.settings.beir_tasks
    )
    rd = RunDir.create(tmp_path / "runs", "qemb", "beir")
    job = beir.job(cfg.get("qemb"), rd, None, False)
    assert job["mode"] == "mteb" and job["reranker"]["name"] == "qrr" and job["rerank_top_k"] == 100
    assert (
        MTEBRetrievalAdapter(cfg).job(cfg.get("qemb"), rd, None, False)["reranker"] is None
    )  # MTEB number is embedding-only
    own = OwnDataAdapter(cfg).job(cfg.get("qemb"), rd, None, False)
    assert own["mode"] == "own" and own["own"]["queries"].endswith("queries.jsonl")
    plan = beir.plan(cfg.get("qemb"), None, 1, False)
    assert plan.n_tasks == len(cfg.settings.beir_tasks) and "reranker: qrr" in plan.notes[1]


def _summary(mode, tasks, embed, rerank=None):
    s = {
        "mteb_version": "2.24.12",
        "mode": mode,
        "tasks": tasks,
        "errors": [],
        "stages": {"embed": {"seconds": 10.0, "tasks": embed, "exceptions": []}},
    }
    if rerank is not None:
        s["stages"]["rerank"] = {"seconds": 5.0, "top_k": 100, "tasks": rerank, "exceptions": []}
    return s


def _run_dir(tmp_path, cfg, bench, summary, smoke=False):
    rd = RunDir.create(tmp_path / "runs", "qemb", bench)
    (rd.raw_dir / "summary.json").write_text(json.dumps(summary))
    rd.write_config(
        RunConfig(
            model_id="qemb",
            bench=bench,
            model={},
            request_params={},
            per_model_exceptions={},
            harness={"reranker": {"id": "qrr"}, "rerank_top_k": 100},
            subset={},
            trials=1,
            smoke=smoke,
        )
    )
    return rd


def test_parse_beir_and_own(models_yaml, tmp_path):
    cfg = _cfg_with_reranker(models_yaml, tmp_path)
    s = _summary(
        "mteb",
        ["SciFact", "NFCorpus"],
        {
            "SciFact": {"test": {"ndcg_at_10": 0.70, "recall_at_10": 0.9}},
            "NFCorpus": {"test": {"ndcg_at_10": 0.30}},
        },
        {"SciFact": {"test": {"ndcg_at_10": 0.76}}, "NFCorpus": {"test": {"ndcg_at_10": 0.36}}},
    )
    res = BEIRAdapter(cfg).parse(_run_dir(tmp_path, cfg, "beir", s))
    assert res.metrics["ndcg_at_10_mean"] == pytest.approx(50.0) and res.metrics[
        "ndcg_at_10_mean_reranked"
    ] == pytest.approx(56.0)
    assert (
        res.metrics["ndcg_at_10/SciFact"] == pytest.approx(70.0)
        and res.n_tasks == 2
        and res.cost_usd is None
    )
    assert res.tracker["qemb__beir"]["v"] == "50.0" and res.tracker["qemb__beir_rr"]["v"] == "56.0"
    assert "reranker qrr top-100" in res.note
    own = _summary(
        "own",
        ["BenchLabOwn"],
        {"BenchLabOwn": {"test": {"ndcg_at_10": 0.5, "recall_at_10": 0.8, "recall_at_100": 1.0}}},
        {"BenchLabOwn": {"test": {"ndcg_at_10": 0.6, "recall_at_10": 0.8}}},
    )
    res = OwnDataAdapter(cfg).parse(_run_dir(tmp_path, cfg, "own", own))
    assert {k: v["v"] for k, v in res.tracker.items()} == {
        "qemb__own_rec": "80.0",
        "qemb__own_ndcg": "50.0",
        "qemb__own_rec_rr": "80.0",
        "qemb__own_ndcg_rr": "60.0",
    }
    rr = _summary(
        "reranking_tasks",
        ["SciDocsRR"],
        {"SciDocsRR": {"test": {"main_score": 0.8}}},
        {"SciDocsRR": {"test": {"main_score": 0.85}}},
    )
    res = MTEBRerankingAdapter(cfg).parse(_run_dir(tmp_path, cfg, "mteb_rr", rr))
    assert (
        res.tracker["qemb__mteb_rr"]["v"] == "80.0" and res.tracker["qrr__mteb_rr"]["v"] == "85.0"
    )
    # exceptions are counted as errors and surfaced
    bad = _summary("mteb", ["SciFact"], {}, None)
    bad["stages"]["embed"]["exceptions"] = ["ConnectionError: refused"]
    res = MTEBRetrievalAdapter(cfg).parse(_run_dir(tmp_path, cfg, "mteb_ret", bad))
    assert (
        res.errors.api_errors == 1 and res.metrics["ndcg_at_10_mean"] is None and res.tracker == {}
    )


@pytest.mark.skipif(not VENV_PY.exists(), reason="retrieval harness venv not built")
def test_runner_end_to_end_own_data(tmp_path):
    """Real mteb evaluation (official scorer) on tiny own data with deterministic test models."""
    docs = tmp_path / "retrieval" / "docs"
    docs.mkdir(parents=True)
    (docs / "d1.txt").write_text("the cat sat on the mat")
    (docs / "d2.txt").write_text("dogs bark loudly at night")
    (docs / "d3.txt").write_text("quantum physics of cats and particles")
    (tmp_path / "retrieval" / "queries.jsonl").write_text(
        '{"id":"q1","query":"cat on a mat","relevant_doc_ids":["d1"]}\n{"id":"q2","query":"barking dogs","relevant_doc_ids":["d2"]}\n'
    )
    job = {
        "mode": "own",
        "embedding": {"kind": "test_hash", "name": "hash-enc"},
        "reranker": {"kind": "test_hash", "name": "hash-ce"},
        "rerank_top_k": 100,
        "tasks": [],
        "out_dir": str(tmp_path / "out"),
        "own": {"queries": str(tmp_path / "retrieval" / "queries.jsonl"), "docs": str(docs)},
    }
    (tmp_path / "job.json").write_text(json.dumps(job))
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    out = subprocess.run(
        [str(VENV_PY), str(RUNNER), str(tmp_path / "job.json")],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    emb = s["stages"]["embed"]["tasks"]["BenchLabOwn"]["test"]
    rr = s["stages"]["rerank"]["tasks"]["BenchLabOwn"]["test"]
    assert emb["ndcg_at_10"] == 1.0 and emb["recall_at_10"] == 1.0 and rr["ndcg_at_10"] == 1.0
    assert (tmp_path / "out" / "predictions" / "BenchLabOwn_predictions.json").exists()
    assert list(
        (tmp_path / "out" / "cache_embed").rglob("BenchLabOwn.json")
    )  # mteb result cache -> resume
