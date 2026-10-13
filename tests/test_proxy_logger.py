"""Unit tests for the proxy usage callback (kwargs shaped like LiteLLM's)."""

from datetime import datetime, timedelta
from types import SimpleNamespace

from benchlab.proxy_logger import BenchLabProxyLogger, _record
from benchlab.usage import UsageLog
from tests.conftest import MOCK_KEY


def _kwargs(call_id="c1", stream=False, cost=1.6e-5, tin=17, tout=5, exc=None):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    slo = {
        "model_group": "mock",
        "model": "openai/mock-model",
        "prompt_tokens": tin,
        "completion_tokens": tout,
        "response_cost": cost,
        "completion_startTime": t0 + timedelta(seconds=0.3 if stream else 1.0),
        "stream": stream,
        "metadata": {
            "user_agent": "inspect/0.3",
            "requester_metadata": {
                "headers": {"x-benchlab-task": "t7", "authorization": f"Bearer {MOCK_KEY}"}
            },
        },
    }
    kw = {
        "model": "mock",
        "litellm_call_id": call_id,
        "stream": stream,
        "response_cost": cost,
        "standard_logging_object": slo,
    }
    if exc is not None:
        kw["exception"] = exc
    return kw, t0, t0 + timedelta(seconds=1.0)


def _resp(n_tools=0):
    msg = SimpleNamespace(tool_calls=[object()] * n_tools if n_tools else None)
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=17, completion_tokens=5),
        choices=[SimpleNamespace(message=msg)],
    )


def test_record_fields():
    kw, t0, t1 = _kwargs()
    r = _record(kw, _resp(n_tools=1), t0, t1, failed=False)
    assert r.model_id == "mock" and r.status == "ok" and r.n_tool_calls == 1
    assert r.latency_s == 1.0 and r.ttft_s is None  # not streamed -> no TTFT
    assert r.tokens_in == 17 and r.cost_usd == 1.6e-5
    assert r.tags["x-benchlab-task"] == "t7" and "authorization" not in r.tags
    assert MOCK_KEY not in r.to_json()


def test_record_stream_ttft_and_unknown_cost():
    kw, t0, t1 = _kwargs(stream=True, cost=0.0)
    r = _record(kw, _resp(), t0, t1, failed=False)
    assert r.ttft_s == 0.3
    assert r.cost_usd is None  # zero cost with real tokens = price unknown


def test_record_failure_classification():
    class Timeout(Exception):
        pass

    kw, t0, t1 = _kwargs(exc=Timeout(f"timed out, key {MOCK_KEY}"), tin=0, tout=0, cost=0.0)
    r = _record(kw, None, t0, t1, failed=True)
    assert r.status == "timeout" and r.error_type == "Timeout"
    assert MOCK_KEY not in (r.error_msg or "")


def test_dedupe_sync_and_async_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHLAB_USAGE_LOG", str(tmp_path / "u.jsonl"))
    lg = BenchLabProxyLogger()
    kw, t0, t1 = _kwargs(call_id="same")
    lg.log_success_event(kw, _resp(), t0, t1)
    import asyncio

    asyncio.run(lg.async_log_success_event(kw, _resp(), t0, t1))
    kw2, _, _ = _kwargs(call_id="other")
    lg.log_success_event(kw2, _resp(), t0, t1)
    lg.log_failure_event(kw2, None, t0, t1)  # same id, different outcome -> kept
    recs = list(UsageLog(tmp_path / "u.jsonl").records())
    assert [r["tags"]["call_id"] for r in recs] == ["same", "other", "other"]
