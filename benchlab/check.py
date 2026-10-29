"""``benchlab check --model X``: one chat request, one tool-call request, one
streamed request (for TTFT). Fails in seconds when an endpoint is broken."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchlab.config import BenchConfig, MissingKeyError, ModelSpec
from benchlab.llm import (
    CallResult,
    LLMClient,
    litellm_model_name,
    per_model_exceptions,
    request_params,
)
from benchlab.usage import UsageLog

CHAT_MESSAGES = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "Reply with exactly one word: pong"},
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

# Forced tool_choice is rejected by some current models (e.g. Claude Sonnet 5.5),
# so we use "auto" with an unambiguous instruction and judge the outcome.
TOOL_MESSAGES = [
    {
        "role": "user",
        "content": "What is the weather in Paris right now, in celsius? "
        "Use the get_weather tool; do not answer from memory.",
    },
]

STREAM_MESSAGES = [{"role": "user", "content": "Count from 1 to 20, separated by spaces."}]


@dataclass
class CheckReport:
    model_id: str
    litellm_model: str
    chat: CallResult | None = None
    tool: CallResult | None = None
    stream: CallResult | None = None
    tool_calling: str = "untested"  # ok | not_used | malformed | wrong_tool | error
    tool_detail: str = ""
    problems: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    exceptions: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def judge_tool_call(res: CallResult) -> tuple[str, str]:
    if not res.ok and res.status != "malformed_tool_call":
        return "error", f"{res.error_type}: {res.error_msg}"
    if not res.tool_calls:
        txt = (res.content or "").strip().replace("\n", " ")
        return "not_used", f"model answered in text instead: {txt[:80]!r}"
    bad = [t for t in res.tool_calls if not t.valid]
    if bad:
        return "malformed", f"arguments not valid JSON: {bad[0].arguments_raw[:80]!r}"
    tc = res.tool_calls[0]
    if tc.name != "get_weather":
        return "wrong_tool", f"called {tc.name!r}"
    if "city" not in (tc.arguments or {}):
        return "malformed", f"missing required arg 'city': {tc.arguments}"
    return "ok", f"get_weather({tc.arguments})"


def run_check(
    cfg: BenchConfig,
    spec: ModelSpec,
    timeout: float = 60.0,
    usage_log: UsageLog | None = None,
    with_stream: bool = True,
) -> CheckReport:
    rep = CheckReport(model_id=spec.id, litellm_model=litellm_model_name(spec))
    rep.params = request_params(spec, cfg.settings)
    rep.exceptions = per_model_exceptions(spec, cfg.settings)
    if spec.kind != "chat":
        rep.problems.append(f"{spec.id} is kind={spec.kind}; `check` only probes chat models")
        return rep
    try:
        spec.api_key()
    except MissingKeyError as e:
        rep.problems.append(str(e))
        return rep

    client = LLMClient(cfg.settings, usage_log=usage_log, retries=1)
    tags = {"check": True}

    rep.chat = client.complete(
        spec,
        CHAT_MESSAGES,
        timeout=timeout,
        call_type="chat",
        tags=tags,
        max_tokens=min(64, rep.params.get("max_tokens", 64)),
    )
    if not rep.chat.ok:
        rep.problems.append(f"chat request failed: {rep.chat.error_type}: {rep.chat.error_msg}")
        return rep  # no point continuing
    if not rep.chat.usage_returned:
        rep.problems.append("no token usage returned (cost tracking will be blind for this model)")

    rep.tool = client.complete(
        spec,
        TOOL_MESSAGES,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        timeout=timeout,
        call_type="tool",
        tags=tags,
    )
    rep.tool_calling, rep.tool_detail = judge_tool_call(rep.tool)
    if rep.tool_calling != "ok":
        rep.problems.append(f"tool calling {rep.tool_calling}: {rep.tool_detail}")

    if with_stream:
        rep.stream = client.complete(
            spec,
            STREAM_MESSAGES,
            stream=True,
            timeout=timeout,
            call_type="stream",
            tags=tags,
            max_tokens=128,
        )
        if not rep.stream.ok:
            rep.problems.append(
                f"streaming failed: {rep.stream.error_type}: {rep.stream.error_msg}"
            )
        elif not rep.stream.usage_returned:
            rep.problems.append(
                "streaming returned no usage (set include_usage on the server / vLLM)"
            )
    return rep
