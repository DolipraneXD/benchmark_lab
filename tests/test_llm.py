import asyncio
import json

import pytest

from benchlab.check import WEATHER_TOOL, run_check
from benchlab.config import load_config
from benchlab.llm import LLMClient
from benchlab.usage import UsageLog
from tests.conftest import MOCK_KEY


def test_chat_and_usage_logging(models_yaml, tmp_path, mock_server):
    cfg = load_config(models_yaml)
    log = UsageLog(tmp_path / "u.jsonl")
    client = LLMClient(cfg.settings, usage_log=log, num_retries=0)
    res = client.complete(cfg.get("mock"), [{"role": "user", "content": "say pong"}])
    assert res.ok and res.content == "pong"
    assert res.tokens_in == 17 and res.tokens_out == 5 and res.usage_returned
    assert res.cost_usd == pytest.approx(17 * 1e-6 + 5 * 2e-6)
    recs = list(log.records())
    assert len(recs) == 1 and recs[0]["status"] == "ok" and recs[0]["model_id"] == "mock"
    # the request that reached the server carried our sampling params
    sent = mock_server.app.state.requests[-1]
    assert sent["temperature"] == 0.0 and sent["max_tokens"] == 4096
    # nothing in the log leaks the key
    assert MOCK_KEY not in (tmp_path / "u.jsonl").read_text()


def test_tool_call_parsing(models_yaml):
    cfg = load_config(models_yaml)
    client = LLMClient(cfg.settings, num_retries=0)
    res = client.complete(
        cfg.get("mock"), [{"role": "user", "content": "weather"}], tools=[WEATHER_TOOL]
    )
    assert res.ok and res.tool_calls[0].name == "get_weather"
    assert res.tool_calls[0].arguments == {"city": "Paris", "unit": "celsius"}
    bad = client.complete(
        cfg.get("badtools"), [{"role": "user", "content": "weather"}], tools=[WEATHER_TOOL]
    )
    assert (
        bad.status == "malformed_tool_call" and not bad.ok and bad.tool_calls[0].arguments is None
    )


def test_streaming_ttft(models_yaml):
    cfg = load_config(models_yaml)
    client = LLMClient(cfg.settings, num_retries=0)
    res = client.complete(cfg.get("mock"), [{"role": "user", "content": "count"}], stream=True)
    assert res.ok and res.ttft_s is not None and 0 < res.ttft_s < res.latency_s + 1e-6
    assert res.usage_returned and res.tokens_in == 17
    assert "1 2 3" in (res.content or "")


def test_error_classification(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    log = UsageLog(tmp_path / "u.jsonl")
    client = LLMClient(cfg.settings, usage_log=log, num_retries=0)
    res = client.complete(cfg.get("broken"), [{"role": "user", "content": "x"}])
    assert not res.ok and res.status == "error" and res.error_type
    res = client.complete(cfg.get("slow"), [{"role": "user", "content": "x"}], timeout=0.5)
    assert res.status == "timeout", (res.status, res.error_type, res.error_msg)
    s = log.summarize()
    assert s.n_requests == 2 and s.n_timeouts == 1 and s.n_errors == 1 and s.error_rate == 1.0


def test_async_concurrency(models_yaml):
    cfg = load_config(models_yaml)
    client = LLMClient(cfg.settings, num_retries=0)

    async def go():
        return await asyncio.gather(
            *[
                client.acomplete(cfg.get("mock"), [{"role": "user", "content": "pong"}])
                for _ in range(6)
            ]
        )

    results = asyncio.run(go())
    assert all(r.ok for r in results)


def test_check_ok_and_failures(models_yaml):
    cfg = load_config(models_yaml)
    rep = run_check(cfg, cfg.get("mock"), timeout=10)
    assert rep.ok and rep.tool_calling == "ok", rep.problems
    assert rep.stream.ttft_s is not None
    rep = run_check(cfg, cfg.get("badtools"), timeout=10)
    assert rep.tool_calling == "malformed" and not rep.ok
    rep = run_check(cfg, cfg.get("notools"), timeout=10)
    assert rep.tool_calling == "not_used"
    rep = run_check(cfg, cfg.get("nousage"), timeout=10)
    assert any("no token usage" in p for p in rep.problems)
    rep = run_check(cfg, cfg.get("broken"), timeout=10)
    assert rep.chat is not None and not rep.chat.ok and rep.tool is None
    rep = run_check(cfg, cfg.get("nokey"), timeout=10)
    assert rep.chat is None and "DEFINITELY_UNSET_KEY_ENV" in rep.problems[0]
    rep = run_check(cfg, cfg.get("qemb"), timeout=10)
    assert "kind=embedding" in rep.problems[0]


def test_sent_request_shape(models_yaml, mock_server):
    cfg = load_config(models_yaml)
    LLMClient(cfg.settings, num_retries=0).complete(
        cfg.get("mock"),
        [{"role": "user", "content": "weather"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    sent = mock_server.app.state.requests[-1]
    assert sent["tools"][0]["function"]["name"] == "get_weather" and sent["tool_choice"] == "auto"
    assert json.dumps(sent).count("sk-") == 0
