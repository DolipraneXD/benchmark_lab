import json
import time

import pytest

from benchlab import registry
from benchlab.config import load_config
from benchlab.results import append_result, read_results
from benchlab.runs import ErrorCounts, NormalizedResult, RunConfig, RunDir
from benchlab.usage import BudgetExceeded, BudgetGuard, UsageLog, UsageRecord
from tests.conftest import MOCK_KEY


def _rec(model="m", status="ok", cost=0.5, **kw):
    base = dict(
        ts=time.time(),
        model_id=model,
        litellm_model="openai/m",
        call_type="chat",
        status=status,
        latency_s=1.0,
        tokens_in=10,
        tokens_out=5,
        cost_usd=cost,
    )
    base.update(kw)
    return UsageRecord(**base)


def test_usage_log_summary_and_budget(tmp_path):
    log = UsageLog(tmp_path / "u.jsonl")
    log.append(_rec(cost=0.4))
    log.append(_rec(status="timeout", cost=None, tokens_in=0, tokens_out=0))
    log.append(_rec(status="malformed_tool_call", cost=0.1))
    log.append(_rec(model="other", cost=1.0, error_msg=f"key {MOCK_KEY} rejected"))
    s = log.summarize("m")
    assert s.n_requests == 3 and s.n_ok == 1 and s.n_timeouts == 1 and s.n_malformed_tool_calls == 1
    assert s.cost_usd == pytest.approx(0.5) and s.cost_known
    assert s.as_dict()["error_rate"] == pytest.approx(2 / 3, abs=1e-3)
    guard = BudgetGuard(1.2, logs=[log])
    with pytest.raises(BudgetExceeded):
        guard.check()
    assert BudgetGuard(None, logs=[log]).check() is None


def test_run_dir_lifecycle_and_resume(tmp_path):
    root = tmp_path / "runs"
    rd = RunDir.create(root, "mock", "gpqa")
    assert rd.path.name.endswith("_mock_gpqa") and (rd.path / "raw").is_dir()
    rd.write_config(
        RunConfig(
            model_id="mock",
            bench="gpqa",
            model={"id": "mock"},
            request_params={},
            per_model_exceptions={},
            harness={"name": "x", "version": "1"},
            subset={"ids": ["a", "b", "c"], "n": 3},
            trials=1,
        )
    )
    c = rd.read_config()
    assert c["benchlab_version"] and c["created_at"] and c["subset"]["ids"] == ["a", "b", "c"]
    rd.mark_done("a")
    rd.mark_done("b")
    rd.mark_done("x", ok=False)
    assert rd.remaining(["a", "b", "c"]) == ["c"]
    # resumable: config present, no result
    found = RunDir.find_resumable(root, "mock", "gpqa")
    assert found is not None and found.path == rd.path
    assert RunDir.find_resumable(root, "mock", "bfcl") is None
    res = NormalizedResult(
        bench="gpqa",
        model_id="mock",
        metrics={"accuracy": 74.0},
        primary_metric="accuracy",
        n_tasks=3,
        trials=1,
        errors=ErrorCounts(timeouts=1),
        cost_usd=0.01,
        note="n",
    )
    rd.write_result(res)
    assert RunDir.find_resumable(root, "mock", "gpqa") is None
    assert rd.read_result()["errors"]["total"] == 1
    p = append_result(tmp_path / "results", res, rd)
    rows = read_results(tmp_path / "results")
    assert (
        rows[0]["model"] == "mock" and rows[0]["value"] == "74.0000" and rows[0]["timeouts"] == "1"
    )
    assert p.name == "results.csv"


def test_registry_selection(models_yaml):
    cfg = load_config(models_yaml)
    assert [b.key for b in registry.select_benches(stage=1)] == ["gpqa", "bfcl", "apptasks"]
    assert [b.key for b in registry.select_benches(bench="bfcl,gpqa")] == ["gpqa", "bfcl"]
    core = {b.key for b in registry.select_benches(suite="core")}
    assert {"gpqa", "bfcl", "t2air", "tb", "swe", "gaia", "ttft"} <= core
    assert [b.key for b in registry.select_benches(suite="core", stage=3)] == ["tb", "swe"]
    with pytest.raises(ValueError):
        registry.select_benches()
    with pytest.raises(KeyError):
        registry.select_benches(bench="nope")
    # extras apply only to a model that opts in via `extras:`
    autobench = registry.get("autobench")
    assert registry.applies_to(autobench, cfg.get("mock"))  # mock opted in
    assert not registry.applies_to(autobench, cfg.get("claude"))
    assert registry.applies_to(registry.get("gpqa"), cfg.get("claude"))
    assert not registry.applies_to(registry.get("gpqa"), cfg.get("qemb"))
    # not-yet-built adapters resolve to None instead of crashing
    assert registry.load_adapter(registry.get("gpqa")).__name__ == "GPQAAdapter"
    assert registry.load_adapter(registry.get("redteam")) is None  # imported, never run


def test_proxy_config_has_no_secrets(models_yaml, tmp_path):
    from benchlab.proxy import build_proxy_config, write_proxy_config

    cfg = load_config(models_yaml)
    data = build_proxy_config(cfg)
    names = {e["model_name"] for e in data["model_list"]}
    assert {"mock", "claude", "qemb", "bed"} <= names
    mock = next(e for e in data["model_list"] if e["model_name"] == "mock")
    assert mock["litellm_params"]["api_key"] == "os.environ/MOCK_API_KEY"
    assert mock["litellm_params"]["input_cost_per_token"] == pytest.approx(1e-6)
    assert "benchlab.proxy_logger.benchlab_logger" in data["litellm_settings"]["callbacks"]
    p = write_proxy_config(cfg, tmp_path / "litellm_config.yaml")
    txt = p.read_text()
    assert MOCK_KEY not in txt and "sk-ant" not in txt
    assert json.dumps(data)  # serialisable


def test_uv_is_found_in_benchlabs_own_environment(monkeypatch):
    """Harness installs must not depend on the venv being on PATH (terminal runs)."""
    import sys
    from pathlib import Path

    from benchlab import harness_env

    monkeypatch.setenv("PATH", "/nonexistent")
    own = Path(sys.executable).parent / "uv"
    if own.exists():
        assert harness_env._uv() == str(own)
    lite = Path(sys.executable).parent / "litellm"  # the proxy CLI, same rule
    if lite.exists():
        assert harness_env.own_tool("litellm") == str(lite)


def test_proxy_retries_like_direct_calls(models_yaml):
    from benchlab.proxy import build_proxy_config

    cfg = load_config(models_yaml)
    data = build_proxy_config(cfg)
    assert data["litellm_settings"]["num_retries"] == cfg.settings.request_retries
    assert data["router_settings"] == {
        "num_retries": cfg.settings.request_retries,
        "retry_after": cfg.settings.retry_backoff_s,
    }
