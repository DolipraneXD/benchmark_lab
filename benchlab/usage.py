"""Per-request usage log (tokens, cost, latency) and the budget guard.

Every request benchlab makes itself (``benchlab check``, TTFT probes, the
retrieval pipeline) is appended to a JSONL log. Requests made by harnesses go
through the LiteLLM proxy and are logged in the same format by
``benchlab.proxy_logger``. Both feed cost tracking and the budget cap.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchlab.redact import redact_obj

Status = str  # "ok" | "error" | "timeout" | "rate_limit" | "auth" | "malformed_tool_call"


@dataclass
class UsageRecord:
    ts: float
    model_id: str
    litellm_model: str
    call_type: str  # chat | tool | stream | embedding | rerank | proxy
    status: Status
    latency_s: float
    ttft_s: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    finish_reason: str | None = None
    n_tool_calls: int = 0
    error_type: str | None = None
    error_msg: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(redact_obj(asdict(self)), ensure_ascii=False)


@dataclass
class UsageSummary:
    n_requests: int = 0
    n_ok: int = 0
    n_errors: int = 0
    n_timeouts: int = 0
    n_rate_limit: int = 0
    n_malformed_tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)

    def add(self, r: UsageRecord | dict[str, Any]) -> None:
        d = r if isinstance(r, dict) else asdict(r)
        self.n_requests += 1
        st = d.get("status")
        if st == "ok":
            self.n_ok += 1
        elif st == "timeout":
            self.n_timeouts += 1
        elif st == "rate_limit":
            self.n_rate_limit += 1
        elif st == "malformed_tool_call":
            self.n_malformed_tool_calls += 1
        else:
            self.n_errors += 1
        self.tokens_in += int(d.get("tokens_in") or 0)
        self.tokens_out += int(d.get("tokens_out") or 0)
        c = d.get("cost_usd")
        if c is None:
            if (d.get("tokens_in") or 0) + (d.get("tokens_out") or 0) > 0:
                self.cost_known = False
        else:
            self.cost_usd += float(c)
        if d.get("latency_s") is not None:
            self.latencies.append(float(d["latency_s"]))
        if d.get("ttft_s") is not None:
            self.ttfts.append(float(d["ttft_s"]))

    @staticmethod
    def _pct(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
        return s[k]

    @property
    def error_rate(self) -> float:
        bad = self.n_requests - self.n_ok
        return bad / self.n_requests if self.n_requests else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_requests": self.n_requests,
            "n_ok": self.n_ok,
            "n_errors": self.n_errors,
            "n_timeouts": self.n_timeouts,
            "n_rate_limit": self.n_rate_limit,
            "n_malformed_tool_calls": self.n_malformed_tool_calls,
            "error_rate": round(self.error_rate, 4),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6) if self.cost_known else None,
            "latency_p50_s": self._pct(self.latencies, 0.5),
            "latency_p95_s": self._pct(self.latencies, 0.95),
            "ttft_p50_s": self._pct(self.ttfts, 0.5),
            "ttft_p95_s": self._pct(self.ttfts, 0.95),
        }


class UsageLog:
    """Append-only JSONL log, safe for concurrent writers within one process."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, rec: UsageRecord) -> None:
        line = rec.to_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        return self._iter()

    def _iter(self) -> Iterator[dict[str, Any]]:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written line from a crashed run

    def summarize(self, model_id: str | None = None) -> UsageSummary:
        s = UsageSummary()
        for r in self.records():
            if model_id and r.get("model_id") != model_id:
                continue
            s.add(r)
        return s


class BudgetExceeded(RuntimeError):
    def __init__(self, spent: float, limit: float):
        super().__init__(f"budget exceeded: spent ${spent:.4f} of ${limit:.2f}")
        self.spent = spent
        self.limit = limit


class BudgetGuard:
    """Tracks spend against ``--budget-usd``; raises BudgetExceeded when passed.

    ``check()`` is cheap enough to call before every request. For harness runs
    (where requests happen in another process via the proxy) ``watch()`` tails
    the shared usage log every ``interval`` seconds and sets ``tripped``.
    """

    def __init__(self, limit_usd: float | None, logs: list[UsageLog] | None = None):
        self.limit = limit_usd
        self.logs = logs or []
        self.spent = 0.0
        self.tripped = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, cost: float | None) -> None:
        if cost:
            self.spent += cost
        self.check()

    def current_spend(self) -> float:
        total = self.spent
        for log in self.logs:
            total += log.summarize().cost_usd
        return total

    def check(self) -> None:
        if self.limit is None:
            return
        spent = self.current_spend()
        if spent > self.limit:
            self.tripped = True
            raise BudgetExceeded(spent, self.limit)

    def watch(self, interval: float = 5.0, on_trip=None) -> None:
        if self.limit is None or self._thread:
            return

        def loop():
            while not self._stop.wait(interval):
                try:
                    self.check()
                except BudgetExceeded as e:
                    if on_trip:
                        on_trip(e)
                    return

        self._thread = threading.Thread(target=loop, daemon=True, name="budget-watch")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def now() -> float:
    return time.time()


def default_proxy_log_path() -> Path:
    return Path(os.environ.get("BENCHLAB_USAGE_LOG", "runs/proxy_usage.jsonl"))
