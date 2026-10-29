"""AutomationBench (Zapier) - realistic business workflows over 47 simulated SaaS
tools, 600 public scored tasks (6 domains x 100) + a 200-task `simple` warm-up
domain that is not part of the score.

Official harness: https://github.com/zapier/AutomationBench (MIT), pinned by commit
in ``settings.autobench_ref``; installed with ``uv sync`` into
``harnesses/automationbench/repo`` (needs Python 3.13, uv fetches it). Checked
against the README and ``automationbench/scripts/eval.py`` at that commit:
``auto-bench --model M --base-url URL --api-key-var VAR --domains d1,d2
--max-steps 50 --max-concurrent N --toolset api --export-json out.json``.
Scoring is the harness' own assertion rubric; the official pass rate is the mean
of ``task_completed_correctly`` over the 6 public domains (``summary.pass_rate``
in the export, 0-1). Note the public task set is not the held-out set behind
the official leaderboard: expect directional agreement only.

Model names not starting with ``claude-``/``gemini-`` go through Chat Completions
at ``--base-url``, so every model is called through BenchLab's proxy under its
BenchLab id. ``--reasoning-effort`` is left unset (same for every model).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from benchlab.adapters._proc import run_logged
from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import ModelSpec
from benchlab.harness_env import HarnessInstallError, ensure_repo_env, frozen_versions, repo_commit
from benchlab.proxy import ProxyProcess
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

HARNESS_NAME = "automationbench"
REPO = "https://github.com/zapier/AutomationBench.git"
PUBLIC_DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")
KEY_VAR = "BENCHLAB_PROXY_KEY"  # auto-bench reads the bearer token from the env var named by --api-key-var


class AutomationBenchAdapter(BenchAdapter):
    key = "autobench"
    tracker_key = "autobench"
    display = "AutomationBench (public set)"
    stage = 4
    needs_proxy = True
    needs_docker = False
    tokens_in_per_task = 60_000  # ~20 tool-call turns with tool schemas
    tokens_out_per_task = 3_000

    @property
    def ref(self) -> str:
        return self.cfg.settings.autobench_ref

    def repo(self) -> tuple[Path, Path]:
        try:
            return ensure_repo_env(HARNESS_NAME, REPO, self.ref, python="3.13")
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def prepare(self) -> dict[str, Any]:
        repo, bin_dir = self.repo()
        return {
            "name": "automationbench",
            "package": "automation-bench",
            "version": frozen_versions(bin_dir, ["automation-bench", "verifiers"]).get(
                "automation-bench"
            ),
            "verifiers": frozen_versions(bin_dir, ["verifiers"]).get("verifiers"),
            "commit": repo_commit(repo),
            "ref": self.ref,
            "toolset": self.cfg.settings.autobench_toolset,
            "max_steps": self.cfg.settings.autobench_max_steps,
            "domains": list(self.cfg.settings.autobench_domains or PUBLIC_DOMAINS),
        }

    def domains(self, subset: str | None, smoke: bool) -> list[str]:
        if smoke:
            return ["sales"]
        if subset and subset != "all":
            return [d.strip() for d in subset.split(",") if d.strip()]
        return list(self.cfg.settings.autobench_domains or PUBLIC_DOMAINS)

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        doms = self.domains(subset, smoke)
        n = 3 if smoke else 100 * len(doms)
        notes = [
            f"domains {doms}" + (" (first 3 tasks)" if smoke else ""),
            f"max_steps {self.cfg.settings.autobench_max_steps}, toolset {self.cfg.settings.autobench_toolset}",
        ]
        return RunPlan(n, 1, doms, self.tokens_in_per_task, self.tokens_out_per_task, notes)

    def command(
        self,
        model: ModelSpec,
        proxy_url: str,
        doms: list[str],
        out_json: Path,
        smoke: bool,
        bin_dir: Path | None = None,
    ) -> list[str]:
        s = self.cfg.settings
        exe = str(bin_dir / "auto-bench") if bin_dir else "auto-bench"
        cmd = [
            exe,
            "--model",
            model.id,
            "--base-url",
            proxy_url,
            "--api-key-var",
            KEY_VAR,
            "--domains",
            ",".join(doms),
            "--max-steps",
            str(s.autobench_max_steps),
            "--max-concurrent",
            str(model.max_concurrency),
            "--toolset",
            s.autobench_toolset,
            "--export-json",
            str(out_json),
        ]
        if smoke:
            cmd += ["--num-examples", "3"]
        return cmd

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        repo, bin_dir = self.repo()
        raw = run_dir.raw_dir
        out_json = raw / "results.json"
        if out_json.exists() and "done" in run_dir.completed_ids():
            return raw
        doms = self.domains(subset, smoke)
        with ProxyProcess(self.cfg, raw, run_dir.path / RunDir.USAGE) as proxy:
            env = dict(os.environ)
            env.pop("VIRTUAL_ENV", None)
            env[KEY_VAR] = proxy.master_key
            env["OPENAI_API_KEY"] = proxy.master_key  # harness default var, in case
            cmd = self.command(model, proxy.base_url, doms, out_json, smoke, bin_dir)
            (raw / "command.txt").write_text(" ".join(cmd) + "\n")
            run_logged(cmd, env, raw / "auto-bench.log", budget=budget, cwd=repo)
        if out_json.exists():
            run_dir.mark_done("done")
        return raw

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        p = run_dir.raw_dir / "results.json"
        if not p.exists():
            raise FileNotFoundError(f"AutomationBench export missing: {p} (see raw/auto-bench.log)")
        d = json.loads(p.read_text())
        return self.normalize(d, cfg, run_dir)

    def normalize(
        self, d: dict[str, Any], cfg: dict[str, Any], run_dir: RunDir
    ) -> NormalizedResult:
        summ = d.get("summary") or {}
        tasks = d.get("tasks") or []
        n = len(tasks)
        passed = sum(1 for t in tasks if t.get("passed"))
        pass_rate = (passed / n * 100) if n else None
        per_domain: dict[str, list[int]] = {}
        for t in tasks:
            dom = str(t.get("name", "")).split(".", 1)[0]
            per_domain.setdefault(dom, []).append(1 if t.get("passed") else 0)
        metrics: dict[str, float | None] = {
            "pass_rate": pass_rate,
            "partial_credit_mean": (summ.get("avg_score") * 100)
            if summ.get("avg_score") is not None
            else None,
            "tool_calls_per_task": (summ.get("total_tool_calls") / n)
            if n and summ.get("total_tool_calls") is not None
            else None,
            "steps_per_task": (sum(int(t.get("steps") or 0) for t in tasks) / n) if n else None,
        }
        for dom, xs in sorted(per_domain.items()):
            metrics[f"pass_rate_{dom}"] = sum(xs) / len(xs) * 100
        errors = ErrorCounts(
            api_errors=int(summ.get("tasks_with_errors") or 0),
            other=int(summ.get("tasks_with_empty_responses") or 0)
            + len(summ.get("aborted_tasks") or []),
        )
        usage = run_dir.usage_log.summarize(cfg["model_id"])
        tokens_in = usage.tokens_in or int(summ.get("total_input_tokens") or 0)
        tokens_out = usage.tokens_out or int(summ.get("total_output_tokens") or 0)
        warnings = []
        if summ.get("aborted_tasks"):
            warnings.append(
                f"{len(summ['aborted_tasks'])} tasks aborted mid-run (infra/transient); the harness' complete_run.py can heal them"
            )
        if summ.get("tasks_with_errors"):
            warnings.append(f"{summ['tasks_with_errors']} tasks had model-call errors")
        doms = sorted(per_domain) or (cfg.get("subset") or {}).get("ids") or []
        h = cfg.get("harness") or {}
        note = (
            f"AutomationBench public set {h.get('version', '')} (commit {str(h.get('commit') or '')[:8]}), domains {doms}, "
            f"{n} tasks, strict pass rate (all assertions), max_steps {h.get('max_steps')}, toolset {h.get('toolset')}, "
            f"no reasoning_effort; public tasks differ from the held-out leaderboard set"
        )
        if cfg.get("smoke"):
            note += "; SMOKE"
        model_id = cfg["model_id"]
        try:
            tracker_id = self.cfg.get(model_id).tracker
        except KeyError:
            tracker_id = model_id
        tracker = {}
        if pass_rate is not None:
            tracker[f"{tracker_id}__{self.tracker_key}"] = {"v": f"{pass_rate:.1f}", "note": note}
        per_task = [
            {
                "id": t.get("name"),
                "passed": bool(t.get("passed")),
                "score": t.get("score"),
                "steps": t.get("steps"),
                "tool_calls": t.get("num_tool_calls"),
                "errors": bool(t.get("errors")),
            }
            for t in tasks
        ]
        return NormalizedResult(
            bench=self.key,
            model_id=model_id,
            metrics=metrics,
            primary_metric="pass_rate",
            n_tasks=n,
            trials=1,
            errors=errors,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=usage.cost_usd if (usage.n_requests and usage.cost_known) else None,
            wall_time_s=float((d.get("meta") or {}).get("duration_seconds") or 0.0),
            harness=h,
            note=note,
            per_task=per_task,
            tracker=tracker,
            warnings=warnings,
        )
