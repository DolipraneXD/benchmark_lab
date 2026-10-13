"""Shared fixtures: a tiny OpenAI-compatible mock server so the LiteLLM layer
is exercised for real (HTTP, streaming, tool calls, usage) without spending money."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

MOCK_KEY = "sk-mock-secret-key-1234567890"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_app() -> FastAPI:
    app = FastAPI()
    app.state.requests = []

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def models():
        return JSONResponse({"object": "list", "data": [{"id": m, "object": "model", "owned_by": "mock"}
                                                         for m in ("mock-model", "no-tools", "malformed-tools", "no-usage", "slow-model")]})

    @app.post("/v1/chat/completions")
    async def chat(req: Request, authorization: str | None = Header(default=None)):
        body = await req.json()
        app.state.requests.append(body)
        if authorization != f"Bearer {MOCK_KEY}":
            raise HTTPException(401, "bad key")
        model = body["model"]
        if model == "broken-model":
            raise HTTPException(500, "upstream exploded")
        if model == "slow-model":
            time.sleep(3)
        tools = body.get("tools")
        want_malformed = model == "malformed-tools"
        want_no_tools = model == "no-tools"
        tin, tout = 17, 5
        if tools and not want_no_tools:
            # call the first offered tool; get_weather gets real args, anything else `{}`
            first = tools[0]["function"]["name"]
            if first == "get_weather":
                args = (
                    '{"city": "Paris", "unit": "celsius"'
                    if want_malformed
                    else json.dumps({"city": "Paris", "unit": "celsius"})
                )
            else:
                args = "{}"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": first, "arguments": args},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            blob = json.dumps(body["messages"])
            if "pong" in blob:
                text = "pong"
            elif "###STOP###" in blob:
                text = "###STOP###"  # tau2 user simulator: end the conversation at once
            elif "ANSWER: $LETTER" in blob:
                text = "Let me think. ANSWER: A"  # multiple-choice evals (GPQA)
            else:
                text = "1 2 3 4 5 6 7 8 9 10"
            message = {"role": "assistant", "content": text}
            finish = "stop"
        usage = {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}
        if model == "no-usage":
            usage = None

        if body.get("stream"):

            def gen():
                base = {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                }
                time.sleep(0.05)  # measurable TTFT
                if message.get("tool_calls"):
                    delta = {
                        "role": "assistant",
                        "tool_calls": [dict(message["tool_calls"][0], index=0)],
                    }
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                **base,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                            }
                        )
                        + "\n\n"
                    )
                else:
                    for word in message["content"].split(" "):
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    **base,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": word + " "},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            + "\n\n"
                        )
                yield (
                    "data: "
                    + json.dumps(
                        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                    )
                    + "\n\n"
                )
                if usage and (body.get("stream_options") or {}).get("include_usage"):
                    yield "data: " + json.dumps({**base, "choices": [], "usage": usage}) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        if usage:
            resp["usage"] = usage
        return JSONResponse(resp)

    return app


class MockServer:
    def __init__(self):
        self.port = _free_port()
        self.app = make_app()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("mock server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture(scope="session")
def mock_server():
    s = MockServer().start()
    yield s
    s.stop()


@pytest.fixture
def models_yaml(tmp_path: Path, mock_server: MockServer, monkeypatch) -> Path:
    monkeypatch.setenv("MOCK_API_KEY", MOCK_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-tests-000000")
    monkeypatch.delenv("BENCHLAB_MODELS", raising=False)
    p = tmp_path / "models.yaml"
    p.write_text(f"""
settings:
  baseline_model: claude
  tau2_user_model: claude
  runs_dir: {tmp_path / "runs"}
  results_dir: {tmp_path / "results"}
  request_timeout_s: 20
models:
  - id: claude
    name: Claude Sonnet 5.5
    provider: anthropic
    model: claude-sonnet-5-5
    api_key_env: ANTHROPIC_API_KEY
    price_in_per_mtok: 2.0
    price_out_per_mtok: 10.0
    extra_params: {{temperature: null}}
  - id: mock
    name: Mock OK
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: mock-model
    api_key_env: MOCK_API_KEY
    price_in_per_mtok: 1.0
    price_out_per_mtok: 2.0
    max_concurrency: 4
    extra_params: {{temperature: 0.0}}
    suites: [core, coder]
    extras: [glm_aider, mimo_tool, ds_auto]
  - id: badtools
    name: Mock malformed tools
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: malformed-tools
    api_key_env: MOCK_API_KEY
  - id: notools
    name: Mock no tools
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: no-tools
    api_key_env: MOCK_API_KEY
    self_hosted: true
  - id: nousage
    name: Mock no usage
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: no-usage
    api_key_env: MOCK_API_KEY
  - id: broken
    name: Mock 500
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: broken-model
    api_key_env: MOCK_API_KEY
  - id: slow
    name: Mock slow
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: slow-model
    api_key_env: MOCK_API_KEY
  - id: nokey
    name: Mock missing key
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: mock-model
    api_key_env: DEFINITELY_UNSET_KEY_ENV
  - id: bed
    name: Bedrock example
    provider: bedrock
    model: us.anthropic.claude-sonnet-5-5
    region: us-east-1
  - id: qemb
    name: Emb
    kind: embedding
    provider: openai_compatible
    base_url: {mock_server.base_url}
    model: emb
    api_key_env: MOCK_API_KEY
""")
    return p


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch, tmp_path):
    # rich truncates table cells in CliRunner's narrow pseudo-terminal
    monkeypatch.setenv("COLUMNS", "300")
    # never let a test build or touch the real harness checkouts/venvs
    monkeypatch.setenv("BENCHLAB_HARNESS_DIR", str(tmp_path / "harnesses"))
