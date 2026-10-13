"""Toolathlon-Verified (HKUST-NLP) - 108 long-horizon tasks over 32 apps / 604
MCP tools. Official harness: https://github.com/hkust-nlp/Toolathlon, pinned by
commit (``settings.toolathlon_ref``).

What the harness needs on the host, per its README at that commit, and what
BenchLab does NOT set up for you: Docker/Podman + the task image
(``global_preparation/pull_toolathlon_image.sh``), the locally deployed apps
(``global_preparation/deploy_containers.sh``), Node 22, and real accounts /
tokens for the external services in ``configs/token_key_session.py``
(``global_preparation/how2register_accounts.md``). ``prepare()`` refuses to run
until ``bash global_preparation/check_installation_containerized.sh`` has been
run successfully once on the host (it records a marker file).

Run: ``scripts/run_parallel.sh <model> <dump> unified <workers>`` with
``TOOLATHLON_OPENAI_BASE_URL`` / ``_API_KEY`` pointing at BenchLab's proxy (the
"unified" OpenAI-compatible provider). The harness fixes max_tokens 65536, 50
turns / 100 steps, a 5400 s task timeout and provider-default sampling: these
deviations from BenchLab's common parameters are recorded in the note. Scoring
is the harness' own per-task evaluator (``eval_res.json`` -> ``eval_stats.json``).

The authors also run a public evaluation service (``eval_client.py --mode
public``); BenchLab does not drive it because it would hand your model API key
to a third-party server.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from benchlab.adapters._proc import run_logged
from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import ModelSpec
from benchlab.harness_env import HarnessInstallError, ensure_repo, repo_commit
from benchlab.proxy import ProxyProcess
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

HARNESS_NAME = "toolathlon"
REPO = "https://github.com/hkust-nlp/Toolathlon.git"
TASKS_FOLDER = "finalpool"
SETUP_MARKER = ".benchlab_setup_verified"
SMOKE_TASKS = ("find-alita-paper", "add-bibtex", "ab-testing")


class ToolathlonAdapter(BenchAdapter):
    key = "mimo_tool"
    tracker_key = "mimo_tool"
    display = "Toolathlon-Verified"
    stage = 4
    needs_proxy = True
    needs_docker = True
    tokens_in_per_task = 400_000  # ~20-30 tool turns with big tool schemas and 64k outputs allowed
    tokens_out_per_task = 20_000

    @property
    def ref(self) -> str:
        return self.cfg.settings.toolathlon_ref

    def repo(self) -> Path:
        try:
            return ensure_repo(HARNESS_NAME, REPO, self.ref)
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def setup_problems(self, repo: Path) -> list[str]:
        probs = []
        if not (shutil.which("docker") or shutil.which("podman")):
            probs.append("docker or podman not on PATH")
        if not (repo / ".venv" / "bin" / "python").exists():
            probs.append(
                "harness venv missing: run `bash global_preparation/install_env_minimal.sh false` in "
                + str(repo)
            )
        if not (repo / "configs" / "token_key_session.py").exists():
            probs.append(
                "configs/token_key_session.py missing (copy the _example and fill in the app tokens: global_preparation/how2register_accounts.md)"
            )
        if not (repo / SETUP_MARKER).exists():
            probs.append(
                f"run `bash global_preparation/check_installation_containerized.sh` in {repo} and then `touch {SETUP_MARKER}` there once every MCP server checks out"
            )
        return probs

    def prepare(self) -> dict[str, Any]:
        if not (shutil.which("docker") or shutil.which("podman")):  # before cloning anything
            raise HarnessNotReady(f"{self.key}: Toolathlon needs docker or podman on this host")
        repo = self.repo()
        probs = self.setup_problems(repo)
        if probs:
            raise HarnessNotReady(
                f"{self.key}: Toolathlon host setup incomplete: " + "; ".join(probs)
            )
        return {
            "name": "toolathlon",
            "commit": repo_commit(repo),
            "ref": self.ref,
            "tasks_folder": TASKS_FOLDER,
            "runner": "scripts/run_parallel.sh (containerized, unified provider)",
            "fixed_by_harness": {
                "max_tokens": 65536,
                "max_turns": 50,
                "max_steps": 100,
                "timeout_s": 5400,
                "sampling": "provider defaults",
            },
        }

    def all_tasks(self, repo: Path) -> list[str]:
        d = repo / "tasks" / TASKS_FOLDER
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        ids = list(SMOKE_TASKS) if smoke else None
        n = len(SMOKE_TASKS) if smoke else 108
        notes = [
            "official run_parallel.sh; harness-fixed max_tokens 65536, 50 turns, 5400 s/task, provider-default sampling",
            f"workers {model.max_concurrency}",
        ]
        return RunPlan(n, 1, ids, self.tokens_in_per_task, self.tokens_out_per_task, notes)

    def command(self, model: ModelSpec, dump: Path, workers: int) -> list[str]:
        return ["bash", "scripts/run_parallel.sh", model.id, str(dump), "unified", str(workers)]

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        repo = self.repo()
        raw = run_dir.raw_dir
        dump = raw / "dump"
        if (dump / "eval_stats.json").exists():
            return raw
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        if smoke:
            tl = raw / "task_list.txt"
            tl.write_text("\n".join(SMOKE_TASKS) + "\n")
            env["TASK_LIST"] = str(tl)
        with ProxyProcess(self.cfg, raw, run_dir.path / RunDir.USAGE) as proxy:
            env["TOOLATHLON_OPENAI_BASE_URL"] = proxy.base_url
            env["TOOLATHLON_OPENAI_API_KEY"] = proxy.master_key
            cmd = self.command(model, dump, model.max_concurrency)
            (raw / "command.txt").write_text(" ".join(cmd) + "\n")
            run_logged(cmd, env, raw / "toolathlon.log", budget=budget, cwd=repo)
        if (dump / "eval_stats.json").exists():
            run_dir.mark_done("done")
        return raw

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        p = run_dir.raw_dir / "dump" / "eval_stats.json"
        if not p.exists():
            raise FileNotFoundError(
                f"eval_stats.json missing under {p.parent} (see raw/toolathlon.log)"
            )
        return self.normalize(json.loads(p.read_text()), cfg, run_dir)

    def normalize(
        self, st: dict[str, Any], cfg: dict[str, Any], run_dir: RunDir
    ) -> NormalizedResult:
        n = int(st.get("total_tasks") or 0)
        rate = st.get("average_success_rate")
        success = float(rate) * 100 if rate is not None else None
        sb = st.get("status_breakdown") or {}
        running = sb.get("running") or {}
        evaluation = sb.get("evaluation") or {}
        pre = sb.get("preprocess") or {}
        errors = ErrorCounts(
            timeouts=int(running.get("timeout_count") or 0) + int(pre.get("timeout_count") or 0),
            api_errors=int(running.get("fail_count") or 0),
            other=int(pre.get("fail_count") or 0)
            + int(evaluation.get("null_count") or 0)
            + int(running.get("null_count") or 0),
        )
        metrics: dict[str, float | None] = {
            "success_rate": success,
            "avg_turns": st.get("average_turns"),
            "avg_tool_calls": st.get("average_tool_calls"),
            "max_turns_exceeded": float(running.get("max_turns_count") or 0),
        }
        usage = run_dir.usage_log.summarize(cfg["model_id"])
        warnings = []
        if errors.total:
            warnings.append(
                f"{errors.total} tasks did not complete normally (timeouts {errors.timeouts}, run failures {errors.api_errors}, preprocess/eval problems {errors.other}); they count as failed"
            )
        h = cfg.get("harness") or {}
        note = (
            f"Toolathlon-Verified commit {str(h.get('commit') or '')[:8]}, {n} tasks, official containerized runner, "
            f"harness-fixed max_tokens 65536 / 50 turns / 5400 s per task / provider-default sampling (deviates from the common params)"
        )
        if cfg.get("smoke"):
            note += "; SMOKE"
        model_id = cfg["model_id"]
        try:
            tracker_id = self.cfg.get(model_id).tracker
        except KeyError:
            tracker_id = model_id
        tracker = {}
        if success is not None:
            tracker[f"{tracker_id}__{self.tracker_key}"] = {"v": f"{success:.1f}", "note": note}
        per_task = [{"id": t, "passed": True} for t in evaluation.get("pass_tasks") or []] + [
            {"id": t, "passed": False} for t in evaluation.get("fail_tasks") or []
        ]
        return NormalizedResult(
            bench=self.key,
            model_id=model_id,
            metrics=metrics,
            primary_metric="success_rate",
            n_tasks=n,
            trials=1,
            errors=errors,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=usage.cost_usd if (usage.n_requests and usage.cost_known) else None,
            wall_time_s=run_dir.state().get("elapsed_s", 0.0),
            harness=h,
            note=note,
            per_task=per_task,
            tracker=tracker,
            warnings=warnings,
        )
