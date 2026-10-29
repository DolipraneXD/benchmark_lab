"""The single LLM access layer: every direct request goes through LiteLLM here.

Harnesses talk to the LiteLLM *proxy* (see ``benchlab.proxy``), which is built
from the same ``ModelSpec`` mapping, so a model is described once in
models.yaml and used identically everywhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

from benchlab import providers
from benchlab.config import ModelSpec, Settings
from benchlab.redact import redact
from benchlab.usage import UsageLog, UsageRecord

litellm.suppress_debug_info = True
litellm.set_verbose = False
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(
    logging.ERROR
)  # e.g. "custom pricing not in cost map" warnings


# --------------------------------------------------------------------------- #
# ModelSpec -> LiteLLM parameters
# --------------------------------------------------------------------------- #
def litellm_model_name(spec: ModelSpec) -> str:
    """LiteLLM routing string for a model (provider prefix + model name)."""
    return providers.route(spec.provider, spec.model)


def endpoint_params(spec: ModelSpec, api_key: str | None) -> dict[str, Any]:
    """Where and how to reach the model: base URL, key, API version, region. ``api_key``
    is the key value (direct calls) or an ``os.environ/VAR`` reference (proxy config)."""
    prov = providers.get(spec.provider)
    p: dict[str, Any] = {"model": litellm_model_name(spec)}
    base = spec.base_url or prov.default_base_url
    if base:
        p["api_base"] = base
    if api_key:
        p["api_key"] = api_key
    elif spec.provider == "openai_compatible":
        p["api_key"] = "EMPTY"  # most self-hosted servers want some token
    if spec.api_version:
        p["api_version"] = spec.api_version
    if prov.needs_region:
        region = spec.region or os.environ.get("AWS_REGION_NAME") or os.environ.get("AWS_REGION")
        if region:
            p["aws_region_name"] = region
    if spec.extra_headers:
        p["extra_headers"] = dict(spec.extra_headers)
    if spec.has_prices:
        # explicit prices from models.yaml win over LiteLLM's built-in price map
        p["input_cost_per_token"] = spec.price_in_per_mtok / 1e6
        p["output_cost_per_token"] = spec.price_out_per_mtok / 1e6
    return p


def litellm_params(spec: ModelSpec, settings: Settings | None = None) -> dict[str, Any]:
    """Provider/auth parameters (the ``litellm_params`` of a proxy entry, or
    kwargs for ``litellm.completion``). The key value is read from the env at
    call time and must never be persisted by callers."""
    settings = settings or Settings()
    p = endpoint_params(spec, spec.api_key())
    p["timeout"] = settings.request_timeout_s
    return p


def request_params(spec: ModelSpec, settings: Settings | None = None) -> dict[str, Any]:
    """Sampling params: global defaults, then the model's ``extra_params``.
    A ``null`` in extra_params removes the param (recorded as an exception)."""
    settings = settings or Settings()
    params: dict[str, Any] = {"max_tokens": settings.default_max_tokens}
    if settings.default_temperature is not None:
        params["temperature"] = settings.default_temperature
    for k, v in spec.extra_params.items():
        if v is None:
            params.pop(k, None)
        else:
            params[k] = v
    return params


def per_model_exceptions(spec: ModelSpec, settings: Settings | None = None) -> dict[str, Any]:
    """What differs from the global defaults for this model (goes in config.json)."""
    settings = settings or Settings()
    base = {"max_tokens": settings.default_max_tokens}
    if settings.default_temperature is not None:
        base["temperature"] = settings.default_temperature
    eff = request_params(spec, settings)
    exc: dict[str, Any] = {}
    for k in set(base) | set(eff):
        if base.get(k) != eff.get(k):
            exc[k] = eff.get(k, "<omitted>")
    return exc


# --------------------------------------------------------------------------- #
# Calls
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    name: str
    arguments_raw: str
    arguments: dict[str, Any] | None  # None when the JSON did not parse
    id: str | None = None

    @property
    def valid(self) -> bool:
        return self.arguments is not None


@dataclass
class CallResult:
    ok: bool
    status: str  # ok | error | timeout | rate_limit | auth | malformed_tool_call
    latency_s: float
    ttft_s: float | None = None
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    usage_returned: bool = False
    cost_usd: float | None = None
    error_type: str | None = None
    error_msg: str | None = None
    raw: Any = None

    def record(self, spec: ModelSpec, call_type: str, **tags: Any) -> UsageRecord:
        return UsageRecord(
            ts=time.time(),
            model_id=spec.id,
            litellm_model=litellm_model_name(spec),
            call_type=call_type,
            status=self.status,
            latency_s=self.latency_s,
            ttft_s=self.ttft_s,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=self.cost_usd,
            finish_reason=self.finish_reason,
            n_tool_calls=len(self.tool_calls),
            error_type=self.error_type,
            error_msg=self.error_msg,
            tags=tags,
        )


def classify_exception(e: BaseException) -> tuple[str, str]:
    """Map a LiteLLM/OpenAI exception to (status, error_type)."""
    name = type(e).__name__
    if isinstance(e, (litellm.Timeout, asyncio.TimeoutError, TimeoutError)):
        return "timeout", name
    if isinstance(e, litellm.RateLimitError):
        return "rate_limit", name
    if isinstance(e, (litellm.AuthenticationError, litellm.PermissionDeniedError)):
        return "auth", name
    return "error", name


# server-side hiccups worth another try; a wrong answer or a bad key never is
TRANSIENT_ERRORS = {
    "InternalServerError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "BadGatewayError",
    "APIError",
}


def transient(res: CallResult) -> bool:
    if res.ok:
        return False
    if res.status in ("rate_limit", "timeout"):
        return True
    return res.status == "error" and res.error_type in TRANSIENT_ERRORS


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) or ""
        raw = getattr(fn, "arguments", None) or ""
        try:
            args = json.loads(raw) if raw else {}
            if not isinstance(args, dict):
                args = None
        except json.JSONDecodeError:
            args = None
        out.append(
            ToolCall(name=name, arguments_raw=raw, arguments=args, id=getattr(tc, "id", None))
        )
    return out


def _finish(spec: ModelSpec, resp: Any, latency: float, ttft: float | None) -> CallResult:
    choice = resp.choices[0] if resp.choices else None
    message = choice.message if choice else None
    usage = getattr(resp, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tout = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    usage_returned = bool(usage) and (tin + tout) > 0
    cost = spec.cost_usd(tin, tout) if usage_returned else None
    if cost is None and usage_returned and not spec.has_prices:
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = None
    tool_calls = _parse_tool_calls(message) if message else []
    status = "ok"
    if tool_calls and any(not t.valid for t in tool_calls):
        status = "malformed_tool_call"
    return CallResult(
        ok=status == "ok",
        status=status,
        latency_s=latency,
        ttft_s=ttft,
        content=getattr(message, "content", None) if message else None,
        tool_calls=tool_calls,
        finish_reason=getattr(choice, "finish_reason", None) if choice else None,
        tokens_in=tin,
        tokens_out=tout,
        usage_returned=usage_returned,
        cost_usd=cost,
        raw=resp,
    )


def _fail(e: BaseException, latency: float) -> CallResult:
    status, etype = classify_exception(e)
    return CallResult(
        ok=False,
        status=status,
        latency_s=latency,
        error_type=etype,
        error_msg=redact(str(e))[:2000],
    )


class LLMClient:
    """Thin stateful wrapper: holds settings, optional usage log, per-model semaphores."""

    def __init__(
        self,
        settings: Settings | None = None,
        usage_log: UsageLog | None = None,
        num_retries: int = 0,
        retries: int | None = None,
    ):
        self.settings = settings or Settings()
        self.usage_log = usage_log
        self.num_retries = num_retries  # LiteLLM's own retries; BenchLab's loop does the rest
        self.retries = self.settings.request_retries if retries is None else retries
        self._sems: dict[str, asyncio.Semaphore] = {}

    # -- kwargs -----------------------------------------------------------
    def build_kwargs(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        timeout: float | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        kw = litellm_params(spec, self.settings)
        kw.update(request_params(spec, self.settings))
        kw.update(overrides)
        kw["messages"] = messages
        kw["drop_params"] = True
        kw["num_retries"] = self.num_retries
        if timeout is not None:
            kw["timeout"] = timeout
        if tools:
            kw["tools"] = tools
            if tool_choice is not None:
                kw["tool_choice"] = tool_choice
        if stream:
            kw["stream"] = True
            kw["stream_options"] = {"include_usage": True}
        return kw

    def _log(self, spec: ModelSpec, res: CallResult, call_type: str, tags: dict[str, Any]) -> None:
        if self.usage_log is not None:
            self.usage_log.append(res.record(spec, call_type, **tags))

    # -- sync -------------------------------------------------------------
    def complete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        call_type: str | None = None,
        tags: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> CallResult:
        """One chat completion. ``stream=True`` also measures time-to-first-token."""
        kw = self.build_kwargs(spec, messages, tools, tool_choice, stream, **overrides)
        ctype = call_type or ("tool" if tools else ("stream" if stream else "chat"))
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                if not stream:
                    resp = litellm.completion(**kw)
                    res = _finish(spec, resp, time.perf_counter() - t0, None)
                else:
                    chunks, ttft = [], None
                    for chunk in litellm.completion(**kw):
                        if ttft is None and _chunk_has_token(chunk):
                            ttft = time.perf_counter() - t0
                        chunks.append(chunk)
                    resp = litellm.stream_chunk_builder(chunks, messages=messages)
                    res = _finish(spec, resp, time.perf_counter() - t0, ttft)
            except Exception as e:  # noqa: BLE001 - we classify and record everything
                res = _fail(e, time.perf_counter() - t0)
            if not transient(res) or attempt == self.retries:
                break
            time.sleep(self._backoff(attempt))
        self._log(spec, res, ctype, {**(tags or {}), **({"retries": attempt} if attempt else {})})
        return res

    def _backoff(self, attempt: int) -> float:
        """2, 4, 8, 16 s … (capped at 60) plus jitter, so parallel requests don't retry in step."""
        base = self.settings.retry_backoff_s * (2**attempt)
        return min(60.0, base) * (0.75 + random.random() * 0.5)

    # -- async ------------------------------------------------------------
    def _sem(self, spec: ModelSpec) -> asyncio.Semaphore:
        if spec.id not in self._sems:
            self._sems[spec.id] = asyncio.Semaphore(spec.max_concurrency)
        return self._sems[spec.id]

    async def acomplete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        call_type: str | None = None,
        tags: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> CallResult:
        """Async variant, bounded by the model's ``max_concurrency``."""
        kw = self.build_kwargs(spec, messages, tools, tool_choice, stream, **overrides)
        ctype = call_type or ("tool" if tools else ("stream" if stream else "chat"))
        # the slot is kept while backing off: a rate-limited endpoint gets fewer requests
        async with self._sem(spec):
            for attempt in range(self.retries + 1):
                t0 = time.perf_counter()
                try:
                    if not stream:
                        resp = await litellm.acompletion(**kw)
                        res = _finish(spec, resp, time.perf_counter() - t0, None)
                    else:
                        chunks, ttft = [], None
                        async for chunk in await litellm.acompletion(**kw):
                            if ttft is None and _chunk_has_token(chunk):
                                ttft = time.perf_counter() - t0
                            chunks.append(chunk)
                        resp = litellm.stream_chunk_builder(chunks, messages=messages)
                        res = _finish(spec, resp, time.perf_counter() - t0, ttft)
                except Exception as e:  # noqa: BLE001
                    res = _fail(e, time.perf_counter() - t0)
                if not transient(res) or attempt == self.retries:
                    break
                await asyncio.sleep(self._backoff(attempt))
        self._log(spec, res, ctype, {**(tags or {}), **({"retries": attempt} if attempt else {})})
        return res


def _chunk_has_token(chunk: Any) -> bool:
    try:
        delta = chunk.choices[0].delta
    except (IndexError, AttributeError):
        return False
    if getattr(delta, "content", None):
        return True
    if getattr(delta, "tool_calls", None):
        return True
    return bool(getattr(delta, "reasoning_content", None))
