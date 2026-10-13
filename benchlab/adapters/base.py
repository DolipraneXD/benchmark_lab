"""The adapter interface every benchmark implements.

An adapter wraps one *official* harness. It never scores anything itself: it
prepares the harness, runs it against a model (through the LiteLLM proxy or the
LiteLLM SDK), and parses the harness's own output into a ``NormalizedResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchlab.config import BenchConfig, ModelSpec
from benchlab.runs import NormalizedResult, RunDir
from benchlab.usage import BudgetGuard


@dataclass
class RunPlan:
    """What a run would do - used for ``--dry-run`` and the budget estimate."""

    n_tasks: int
    trials: int
    subset_ids: list[str] | None
    est_tokens_in_per_task: int
    est_tokens_out_per_task: int
    notes: list[str] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return self.n_tasks * self.trials

    def est_cost_usd(self, spec: ModelSpec) -> float | None:
        return spec.cost_usd(
            self.est_tokens_in_per_task * self.total_calls,
            self.est_tokens_out_per_task * self.total_calls,
        )


class HarnessNotReady(RuntimeError):
    """Raised by prepare() when the official harness cannot be installed/verified."""


class BenchAdapter(ABC):
    key: str = ""  # bench key as used in tracker keys / --bench
    display: str = ""
    stage: int = 4
    needs_docker: bool = False
    needs_proxy: bool = True

    def __init__(self, cfg: BenchConfig, harness_root: Path | None = None):
        self.cfg = cfg
        self.harness_root = Path(harness_root or "harnesses") / self.key

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Install / verify the official harness; return pinned version info
        ({"name", "package"|"repo", "version"|"commit"}). Must be idempotent."""

    @abstractmethod
    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        """Describe the run without executing it."""

    @abstractmethod
    def run(
        self,
        model: ModelSpec,
        run_dir: RunDir,
        subset: str | None,
        trials: int,
        smoke: bool,
        budget: BudgetGuard | None = None,
    ) -> Path:
        """Execute the harness, writing raw output under ``run_dir.raw_dir``.
        Must honour ``run_dir.completed_ids()`` to resume. Returns the raw dir."""

    @abstractmethod
    def parse(self, run_dir: RunDir) -> NormalizedResult:
        """Read the harness's own scoring output and normalize it."""

    # optional hook: adapters that know published scores can override
    def published_score(self, model: ModelSpec) -> float | None:
        return model.published.get(self.key)
