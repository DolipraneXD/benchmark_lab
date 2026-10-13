"""Unit tests for the Inspect adapters (command building, subsets, log parsing)."""

import json
from pathlib import Path

import pytest

from benchlab.adapters.base import HarnessNotReady
from benchlab.adapters.inspect_bench import (
    BFCL_COUNTS,
    BFCL_SUBSETS,
    BFCLAdapter,
    BrowseCompAdapter,
    GAIAAdapter,
    GPQAAdapter,
    InspectAdapter,
    _ns,
    classify_error_message,
)
from benchlab.config import load_config
from benchlab.runs import RunConfig, RunDir

FIXTURES = Path(__file__).parent / "fixtures"


def _cmd(adapter, model, subset=None, trials=1, smoke=False):
    return " ".join(
        adapter.build_command(model, Path("logs"), subset, trials, smoke, "http://p/v1")
    )


def test_gpqa_command_and_per_model_params(models_yaml):
    cfg = load_config(models_yaml)
    a = GPQAAdapter(cfg)
    c = _cmd(a, cfg.get("mock"))
    assert "eval-set inspect_evals/gpqa_diamond" in c
    assert "--model openai/mock --model-base-url http://p/v1 -M responses_api=false" in c
    assert "--epochs 1" in c and "--temperature 0.0" in c and "--max-tokens 4096" in c
    assert "--max-connections 4" in c and "--no-fail-on-error" in c and "--log-format eval" in c
    # claude: temperature omitted (per-model exception), everything else identical
    c2 = _cmd(a, cfg.get("claude"))
    assert "--temperature" not in c2 and "--max-tokens 4096" in c2
    assert "--epochs 3" in _cmd(a, cfg.get("mock"), trials=3)
    assert "--limit 3" in _cmd(a, cfg.get("mock"), smoke=True)
    assert "--limit 20" in _cmd(a, cfg.get("mock"), subset="20")
    assert a.plan(cfg.get("mock"), None, 1, False).n_tasks == 198
    assert a.plan(cfg.get("mock"), None, 1, True).n_tasks == 3


def test_bfcl_categories_and_counts(models_yaml):
    cfg = load_config(models_yaml)
    a = BFCLAdapter(cfg)
    assert sum(BFCL_COUNTS[c] for c in BFCL_SUBSETS["single_turn"]) == 3641
    assert sum(BFCL_COUNTS[c] for c in BFCL_SUBSETS["leaderboard"]) == 3641 + 800
    c = _cmd(a, cfg.get("mock"))
    assert "-T categories=['simple_python','simple_java'" in c and "multi_turn" not in c
    assert "multi_turn_base" in _cmd(a, cfg.get("mock"), subset="leaderboard")
    assert "-T categories=['irrelevance','sql']" in _cmd(
        a, cfg.get("mock"), subset="irrelevance+sql"
    )
    with pytest.raises(ValueError, match="unknown BFCL categories"):
        a.plan(cfg.get("mock"), "nope_cat", 1, False)
    p = a.plan(cfg.get("mock"), None, 1, True)
    assert p.n_tasks == 3 and "smoke" in " ".join(p.notes)
    assert a.plan(cfg.get("mock"), "multi_turn", 1, False).n_tasks == 800


def test_gaia_subset_and_browsecomp_grader(models_yaml, monkeypatch, tmp_path):
    cfg = load_config(models_yaml)
    g = GAIAAdapter(cfg)
    g.subsets_root = tmp_path / "subsets"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(g, "harness_bin", lambda: tmp_path / "fake-bin")
    with pytest.raises(HarnessNotReady, match="HF_TOKEN"):
        g.prepare()
    # dataset unreachable -> plan still works and says the subset is picked on first run
    plan = g.plan(cfg.get("mock"), None, 1, False)
    assert plan.n_tasks == 50 and plan.subset_ids is None and "first run" in plan.notes[0]
    # subset of ids is picked once with a fixed seed, saved, and reused
    monkeypatch.setattr(g, "_all_validation_ids", lambda: [f"id{i}" for i in range(165)])
    plan = g.plan(cfg.get("mock"), None, 1, False)
    assert plan.n_tasks == 50 and len(plan.subset_ids) == 50
    saved = json.loads((tmp_path / "subsets" / "gaia_validation_50.json").read_text())
    assert saved["ids"] == plan.subset_ids and saved["seed"] == 20261013
    monkeypatch.setattr(
        g, "_all_validation_ids", lambda: (_ for _ in ()).throw(AssertionError("must reuse"))
    )
    assert g.plan(cfg.get("mock"), None, 1, False).subset_ids == plan.subset_ids
    c = _cmd(g, cfg.get("mock"), subset="50")
    assert "-T instance_ids=['id" in c and "--limit" not in c and "-T split=validation" in c
    assert "--limit 3" in _cmd(g, cfg.get("mock"), smoke=True)

    b = BrowseCompAdapter(cfg)
    c = _cmd(b, cfg.get("mock"), subset="25")
    # grader is the fixed baseline model via the proxy, never the model under test
    assert (
        "-T scorer_model=openai/claude" in c
        and "-T num_samples=25" in c
        and "-T with_browsing=true" in c
    )
    assert "--limit" not in c and b.tracker_key == "gaia"


def test_classify_error_message():
    assert classify_error_message("Request timed out after 300s") == "timeout"
    assert classify_error_message("RateLimitError: 429 Too Many Requests") == "rate_limit"
    assert (
        classify_error_message("Error parsing tool call arguments: invalid JSON")
        == "malformed_tool_call"
    )
    assert classify_error_message("APIConnectionError: connection refused") == "api_error"
    assert classify_error_message("something odd") == "other"


def _run_dir_with_dump(tmp_path, dump: dict, model_id="mock", bench="bfcl", smoke=True):
    rd = RunDir.create(tmp_path / "runs", model_id, bench)
    (rd.raw_dir / "logs").mkdir()
    (rd.raw_dir / "logs" / "x.eval").write_bytes(b"placeholder")
    (rd.raw_dir / "log_dump.json").write_text(json.dumps(dump))
    rd.write_config(
        RunConfig(
            model_id=model_id,
            bench=bench,
            model={},
            request_params={"temperature": 0.0, "max_tokens": 4096},
            per_model_exceptions={},
            harness={},
            subset={},
            trials=1,
            smoke=smoke,
        )
    )
    return rd


def test_normalize_synthetic_log_counts_errors(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    dump = {
        "status": "success",
        "eval": {
            "task": "inspect_evals/gpqa_diamond",
            "task_version": "3-D",
            "task_args": {},
            "model": "openai/mock",
            "packages": {"inspect_evals": "0.24.0", "inspect_ai": "0.3.279"},
            "revision": None,
            "config": {"epochs": 1},
        },
        "results": {
            "total_samples": 4,
            "completed_samples": 3,
            "scores": [
                {
                    "name": "choice",
                    "metrics": {"accuracy": {"value": 2 / 3}, "stderr": {"value": 0.27}},
                }
            ],
        },
        "stats": {
            "started_at": "2026-10-13T10:00:00+00:00",
            "completed_at": "2026-10-13T10:05:00+00:00",
            "model_usage": {"openai/mock": {"input_tokens": 1000, "output_tokens": 400}},
        },
        "samples": [
            {
                "id": "q1",
                "epoch": 1,
                "scores": {"choice": {"value": "C"}},
                "error": None,
                "messages": [],
            },
            {
                "id": "q2",
                "epoch": 1,
                "scores": {"choice": {"value": "I"}},
                "error": None,
                "messages": [],
            },
            {
                "id": "q3",
                "epoch": 1,
                "scores": {"choice": {"value": "C"}},
                "error": None,
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": "f", "parse_error": "bad json"}],
                    }
                ],
            },
            {
                "id": "q4",
                "epoch": 1,
                "scores": None,
                "error": {"message": "Request timed out"},
                "messages": [],
            },
        ],
    }
    rd = _run_dir_with_dump(tmp_path, dump, bench="gpqa", smoke=False)
    a = GPQAAdapter(cfg)
    monkeypatch.setattr(
        a, "harness_bin", lambda: tmp_path / "fake-bin"
    )  # dump is cached -> no venv needed
    res = a.parse(rd)
    assert res.metrics["accuracy"] == pytest.approx(66.6667, abs=1e-3)
    assert res.metrics["accuracy_errors_as_wrong"] == pytest.approx(50.0, abs=1e-3)
    assert (
        res.errors.timeouts == 1 and res.errors.malformed_tool_calls == 1 and res.errors.total == 2
    )
    assert res.n_tasks == 4 and res.trials == 1 and res.wall_time_s == 300.0
    assert res.tokens_in == 1000 and res.cost_usd == pytest.approx(1000 * 1e-6 + 400 * 2e-6)
    assert res.warnings and "errored" in res.warnings[0]
    assert res.tracker == {"mock__gpqa": {"v": "66.7", "note": res.note}}
    assert res.note.startswith("inspect_evals 0.24.0 gpqa_diamond, 4 q, temp 0.0, 1 trial, cot")


@pytest.mark.skipif(
    not (FIXTURES / "bfcl_smoke_mock.log_dump.json").exists(), reason="fixture not present"
)
def test_parse_real_bfcl_log_dump(models_yaml, tmp_path, monkeypatch):
    """Dump of a real `.eval` log produced by inspect_evals/bfcl 0.24.0 against the mock server."""
    cfg = load_config(models_yaml)
    dump = json.loads((FIXTURES / "bfcl_smoke_mock.log_dump.json").read_text())
    rd = _run_dir_with_dump(tmp_path, dump)
    a = BFCLAdapter(cfg)
    monkeypatch.setattr(a, "harness_bin", lambda: tmp_path / "fake-bin")
    res = a.parse(rd)
    assert res.bench == "bfcl" and res.primary_metric == "accuracy"
    assert 0.0 <= res.metrics["accuracy"] <= 100.0
    assert (
        any(k.endswith("_acc") for k in res.metrics)
        and "accuracy_unweighted_categories" in res.metrics
    )
    assert res.n_tasks == 3 and res.trials == 1 and res.tokens_in > 0
    assert res.harness["inspect_evals"] == "0.24.0" and res.harness["task_version"]
    assert "inspect_evals 0.24.0 bfcl, 3 q, temp 0.0, 1 trial" in res.note and "SMOKE" in res.note
    assert list(res.tracker) == ["mock__bfcl"] and len(res.per_task) == 3


def test_ns_helper():
    n = _ns({"a": {"b": [1, {"c": 2}]}})
    assert n.a.b[1].c == 2 and InspectAdapter._plain(n) == {"a": {"b": [1, {"c": 2}]}}
