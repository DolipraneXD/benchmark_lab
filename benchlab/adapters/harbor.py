"""Harbor adapters: Terminal-Bench 2.x and SWE-bench Verified (fixed 50-task subsets).

Harbor (``pip install harbor``, pinned 0.24.0) is the official harness for
Terminal-Bench 2.0 and ships registry datasets for SWE-bench Verified, Aider
Polyglot, LiveCodeBench, … (``harbor dataset list`` → registry.json in the
laude-institute/harbor repo). Verified against harbor 0.24.0:

* ``harbor run -d <name>@<version> -a terminus-2 -m openai/<id> --ak api_base=<proxy>``
  runs the Terminus-2 reference agent (LiteLLM inside) against our per-run proxy;
  ``-i <task>`` selects tasks, ``--jobs-dir``/``--job-name`` place the job folder,
  ``harbor job resume -p <job dir>`` resumes it.
* Every trial writes ``<job>/<trial>/result.json`` (TrialResult: ``verifier_result.rewards``
  from the task's own tests, ``exception_info``, ``agent_result`` token counts) and
  the job writes ``<job>/result.json``. Scoring is the task verifier's; we only
  aggregate rewards.
* Docker is required locally (harbor refuses to start without a daemon, even
  for ``--dry-run``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from benchlab.adapters._proc import run_logged
from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import ModelSpec
from benchlab.harness_env import (
    HarnessInstallError,
    building_image,
    ensure_venv,
    frozen_versions,
    harness_root,
)
from benchlab.llm import request_params
from benchlab.proxy import ProxyProcess
from benchlab.redact import redact
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.subsets import load_subset, pick_subset
from benchlab.usage import BudgetGuard

HARNESS_NAME = "harbor"
HARBOR_VERSION = "0.24.0"
PACKAGES = [f"harbor=={HARBOR_VERSION}"]
REGISTRY_URL = "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
AGENT = "terminus-2"
SUBSET_SEED = 20261013
SMOKE_TASKS = 2


def docker_available() -> bool:
    exe = shutil.which("docker")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def classify_exception(exc_type: str | None) -> str:
    """Map a harbor TrialResult.exception_info.exception_type to an error class."""
    if not exc_type:
        return ""
    t = exc_type.lower()
    if "timeout" in t:
        return "timeout"
    if any(
        k in t
        for k in (
            "apierror",
            "apiconnection",
            "ratelimit",
            "authentication",
            "badrequest",
            "serviceunavailable",
            "internalserver",
            "contextwindow",
        )
    ):
        return "api_error"
    return "other"  # environment build failures, cancelled trials, agent crashes


class HarborAdapter(BenchAdapter):
    dataset_setting: str = ""  # Settings attribute holding "<name>@<version>"
    subset_prefix: str = ""
    stage = 3
    needs_docker = True
    tokens_in_per_task = 400_000
    tokens_out_per_task = 15_000
    subsets_root: Path | None = None  # tests override

    # -- harness -------------------------------------------------------------------
    @property
    def dataset(self) -> str:
        return getattr(self.cfg.settings, self.dataset_setting)

    @property
    def tracker_key(self) -> str:
        return self.key

    def harness_bin(self) -> Path:
        try:
            return ensure_venv(HARNESS_NAME, PACKAGES, python="3.12")
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def registry(self) -> list[dict[str, Any]]:
        """Harbor's dataset registry (downloaded, cached under harnesses/harbor/)."""
        cache = harness_root(HARNESS_NAME) / "registry.json"
        try:
            r = httpx.get(REGISTRY_URL, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(r.text)
            return r.json()
        except Exception as e:  # noqa: BLE001
            if cache.exists():
                return json.loads(cache.read_text())
            raise HarnessNotReady(
                f"{self.key}: cannot download the Harbor registry ({type(e).__name__})"
            ) from e

    def dataset_entry(self) -> dict[str, Any]:
        name, _, version = self.dataset.partition("@")
        reg = self.registry()
        for d in reg:
            if d.get("name") == name and str(d.get("version")) == version:
                return d
        versions = sorted({str(d.get("version")) for d in reg if d.get("name") == name})
        raise HarnessNotReady(
            f"{self.key}: dataset {self.dataset!r} is not in the Harbor registry"
            + (
                f" (available versions of {name}: {versions})"
                if versions
                else f" (no dataset named {name!r})"
            )
        )

    def all_task_ids(self) -> list[str]:
        return [t["name"] for t in self.dataset_entry().get("tasks", [])]

    def prepare(self) -> dict[str, Any]:
        if not docker_available() and not building_image():
            raise HarnessNotReady(
                f"{self.key}: Docker daemon not reachable (harbor runs every task in a container)"
            )
        bin_dir = self.harness_bin()
        if not (bin_dir / "harbor").exists():
            raise HarnessNotReady(f"{self.key}: `harbor` CLI missing in {bin_dir}")
        entry = self.dataset_entry()
        tasks = entry.get("tasks", [])
        first = tasks[0] if tasks else {}
        return {
            "name": "harbor",
            "harbor": frozen_versions(bin_dir, ["harbor"]).get("harbor"),
            "pinned": HARBOR_VERSION,
            "dataset": self.dataset,
            "dataset_tasks_total": len(tasks),
            "dataset_git_url": first.get("git_url"),
            "dataset_git_commit": first.get("git_commit_id"),
            "agent": AGENT,
            "n_attempts": 1,
            "max_tokens": "harness default (not capped for agentic tasks)",
        }

    # -- tasks -----------------------------------------------------------------------
    def subset_spec(self, subset: str | None) -> dict[str, Any] | None:
        if subset == "all":
            return None
        n = int(subset) if (subset and subset.isdigit()) else self.cfg.settings.harbor_subset_size
        name, _, version = self.dataset.partition("@")
        sname = f"{self.subset_prefix}_{name}_{version}_{n}"
        existing = load_subset(sname, self.subsets_root)
        if existing:
            return existing
        ids = self.all_task_ids()
        if n >= len(ids):
            return None
        entry = self.dataset_entry()
        commit = (entry.get("tasks") or [{}])[0].get("git_commit_id", "")
        return pick_subset(
            sname,
            ids,
            n,
            SUBSET_SEED,
            root=self.subsets_root,
            source=f"harbor registry {self.dataset} ({commit[:8]})",
        )

    def task_ids(self, subset: str | None, smoke: bool) -> list[str] | None:
        spec = self.subset_spec(subset)
        ids = spec["ids"] if spec else None
        if smoke:
            return (ids or self.all_task_ids())[:SMOKE_TASKS]
        return ids

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        notes = [f"harbor {HARBOR_VERSION} {self.dataset}, agent {AGENT}"]
        if smoke:
            return RunPlan(
                SMOKE_TASKS,
                1,
                None,
                self.tokens_in_per_task // 4,
                self.tokens_out_per_task // 4,
                notes + [f"smoke: {SMOKE_TASKS} tasks"],
            )
        try:
            ids = self.task_ids(subset, smoke)
            n = len(ids) if ids else len(self.all_task_ids())
            notes.append(
                "full dataset" if ids is None else f"fixed subset of {n} (seed {SUBSET_SEED})"
            )
        except HarnessNotReady as e:
            ids = None
            n = (
                int(subset)
                if (subset and subset.isdigit())
                else self.cfg.settings.harbor_subset_size
            )
            notes.append(f"registry not reachable yet ({e})")
        if trials != 1:
            notes.append(
                "harbor runs 1 attempt per task here (-k 1); use --trials only for pass^k later"
            )
        return RunPlan(n, 1, ids, self.tokens_in_per_task, self.tokens_out_per_task, notes)

    # -- command ---------------------------------------------------------------------
    def agent_kwargs(self, model: ModelSpec, proxy_url: str) -> list[str]:
        params = request_params(model, self.cfg.settings)
        kw = [f"api_base={proxy_url}"]
        if "temperature" in params:
            kw.append(f"temperature={params['temperature']}")
        return kw

    def job_name(self) -> str:
        return self.key

    def build_command(
        self,
        model: ModelSpec,
        jobs_dir: Path,
        subset: str | None,
        smoke: bool,
        proxy_url: str,
        bin_dir: Path | None = None,
    ) -> list[str]:
        exe = str(bin_dir / "harbor") if bin_dir else "harbor"
        cmd = [exe, "run", "-d", self.dataset, "-a", AGENT, "-m", f"openai/{model.id}"]
        for kw in self.agent_kwargs(model, proxy_url):
            cmd += ["--ak", kw]
        cmd += [
            "-n",
            str(model.max_concurrency),
            "-k",
            "1",
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            self.job_name(),
            "-q",
        ]
        for t in self.task_ids(subset, smoke) or []:
            cmd += ["-i", t]
        return cmd

    def resume_command(self, job_dir: Path, bin_dir: Path | None = None) -> list[str]:
        exe = str(bin_dir / "harbor") if bin_dir else "harbor"
        return [exe, "job", "resume", "-p", str(job_dir)]

    def job_dir(self, run_dir: RunDir) -> Path:
        raw = run_dir.raw_dir.resolve()
        job = raw / "jobs" / self.job_name()
        # Older BenchLab runs passed a path relative to ``raw`` and then ran
        # Harbor with ``cwd=raw``. Harbor therefore created this nested path.
        # Keep those jobs resumable while new runs use the canonical path.
        legacy_job = raw / "runs" / run_dir.path.name / "raw" / "jobs" / self.job_name()
        if not (job / "config.json").exists() and (legacy_job / "config.json").exists():
            return legacy_job
        return job

    # -- run / parse -------------------------------------------------------------------
    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        bin_dir = self.harness_bin()
        raw = run_dir.raw_dir.resolve()
        job_dir = self.job_dir(run_dir)
        with ProxyProcess(self.cfg, raw, run_dir.path / RunDir.USAGE) as proxy:
            if (job_dir / "config.json").exists():
                cmd = self.resume_command(job_dir, bin_dir)
            else:
                cmd = self.build_command(
                    model, job_dir.parent, subset, smoke, proxy.base_url, bin_dir
                )
            env = dict(os.environ)
            env.pop("VIRTUAL_ENV", None)
            env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
            env.update(proxy.env_for_harness())
            (raw / "command.txt").write_text(redact(" ".join(cmd)) + "\n")
            run_logged(cmd, env, raw / "harbor.log", budget=budget, cwd=raw)
        try:
            done = sorted(
                {t["task_name"] for t in self.trial_results(job_dir) if t.get("verifier_result")}
            )
            st = run_dir.state()
            st["completed"] = sorted(set(st.get("completed", [])) | set(done))
            run_dir.save_state(st)
        except Exception:  # noqa: BLE001
            pass
        return raw

    @staticmethod
    def trial_results(job_dir: Path) -> list[dict[str, Any]]:
        out = []
        if not job_dir.exists():
            return out
        for p in sorted(job_dir.glob("*/result.json")):
            if p.parent.name.startswith("."):
                continue
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        job_dir = self.job_dir(run_dir)
        trials = self.trial_results(job_dir)
        if not trials:
            raise FileNotFoundError(f"no trial results under {job_dir}")
        job_result = None
        jr = job_dir / "result.json"
        if jr.exists():
            try:
                job_result = json.loads(jr.read_text())
            except (OSError, json.JSONDecodeError):
                job_result = None
        return self.normalize(trials, job_result, cfg, run_dir)

    @staticmethod
    def _reward(t: dict[str, Any]) -> float | None:
        rewards = (t.get("verifier_result") or {}).get("rewards") or None
        if not rewards:
            return None
        if "reward" in rewards:
            return float(rewards["reward"])
        for v in rewards.values():
            if isinstance(v, (int, float)):
                return float(v)
        return None

    @staticmethod
    def _tokens(t: dict[str, Any]) -> tuple[int, int, float | None]:
        ctxs = []
        if t.get("agent_result"):
            ctxs = [t["agent_result"]]
        else:
            ctxs = [
                s["agent_result"] for s in (t.get("step_results") or []) if s.get("agent_result")
            ]
        tin = sum(int(c.get("n_input_tokens") or 0) for c in ctxs)
        tout = sum(int(c.get("n_output_tokens") or 0) for c in ctxs)
        costs = [c["cost_usd"] for c in ctxs if c.get("cost_usd") is not None]
        return tin, tout, (sum(costs) if costs else None)

    @staticmethod
    def _duration(t: dict[str, Any]) -> float | None:
        ae = t.get("agent_execution") or {}
        try:
            a = datetime.fromisoformat(ae["started_at"])
            b = datetime.fromisoformat(ae["finished_at"])
            return (b - a).total_seconds()
        except Exception:  # noqa: BLE001
            return None

    def normalize(
        self,
        trials: list[dict[str, Any]],
        job_result: dict[str, Any] | None,
        cfg: dict[str, Any],
        run_dir: RunDir,
    ) -> NormalizedResult:
        errors = ErrorCounts()
        per_task = []
        rewards: list[float] = []
        n_solved = 0
        tin_sum = tout_sum = 0
        for t in trials:
            reward = self._reward(t)
            exc = (t.get("exception_info") or {}).get("exception_type")
            cls = classify_exception(exc)
            if cls == "timeout":
                errors.timeouts += 1
            elif cls == "api_error":
                errors.api_errors += 1
            elif cls == "other":
                errors.other += 1
            tin, tout, _ = self._tokens(t)
            tin_sum += tin
            tout_sum += tout
            if reward is not None:
                rewards.append(reward)
                if reward >= 1.0 - 1e-6:
                    n_solved += 1
            per_task.append(
                {
                    "id": t.get("task_name"),
                    "trial": t.get("trial_name"),
                    "reward": reward,
                    "solved": (reward is not None and reward >= 1.0 - 1e-6),
                    "error": cls or None,
                    "exception_type": exc,
                    "duration_s": self._duration(t),
                    "tokens_in": tin,
                    "tokens_out": tout,
                }
            )
        n_trials = len(trials)
        task_names = {t.get("task_name") for t in trials}
        metrics: dict[str, float | None] = {
            # "% tasks solved": every trial in the subset counts, errored trials are unsolved
            "resolved": round(100.0 * n_solved / n_trials, 4) if n_trials else None,
            "resolved_excl_errors": round(100.0 * n_solved / len(rewards), 4) if rewards else None,
            "mean_reward": round(100.0 * sum(rewards) / len(rewards), 4) if rewards else None,
            "n_verified": float(len(rewards)),
            "n_errored": float(n_trials - len(rewards)),
        }
        agent_id = cfg["model_id"]
        usage = run_dir.usage_log.summarize(agent_id)
        if usage.n_requests:
            tin, tout = usage.tokens_in, usage.tokens_out
            cost = usage.cost_usd if usage.cost_known else None
        else:
            tin, tout = tin_sum, tout_sum
            try:
                cost = self.cfg.get(agent_id).cost_usd(tin, tout)
            except Exception:  # noqa: BLE001
                cost = None
        warnings = []
        infra = errors.api_errors + errors.timeouts + errors.other
        if n_trials and infra / n_trials > 0.02:
            warnings.append(
                f"{infra}/{n_trials} trials errored (timeouts/API/environment): counted as unsolved; "
                "check the endpoint and Docker resources before trusting the score"
            )
        harness_cfg = cfg.get("harness") or {}
        harness = {
            "name": "harbor",
            "harbor": harness_cfg.get("harbor") or HARBOR_VERSION,
            "dataset": self.dataset,
            "dataset_git_commit": harness_cfg.get("dataset_git_commit"),
            "agent": AGENT,
            "job_dir": str(self.job_dir(run_dir)),
            "job_stats": (job_result or {}).get("stats", {})
            and {
                k: (job_result or {})["stats"].get(k)
                for k in (
                    "n_completed_trials",
                    "n_errored_trials",
                    "n_input_tokens",
                    "n_output_tokens",
                    "cost_usd",
                )
            },
        }
        wall = 0.0
        try:
            a = datetime.fromisoformat(job_result["started_at"])
            b = datetime.fromisoformat(job_result["finished_at"])
            wall = (b - a).total_seconds()
        except Exception:  # noqa: BLE001
            wall = sum(x["duration_s"] or 0 for x in per_task)
        n_tasks = len(task_names)
        note = self.note(cfg, n_tasks, harness)
        try:
            tracker_id = self.cfg.get(agent_id).tracker
        except Exception:  # noqa: BLE001
            tracker_id = agent_id
        tracker = {}
        if metrics["resolved"] is not None:
            tracker[f"{tracker_id}__{self.tracker_key}"] = {
                "v": f"{metrics['resolved']:.1f}",
                "note": note,
            }
        return NormalizedResult(
            bench=self.key,
            model_id=agent_id,
            metrics=metrics,
            primary_metric="resolved",
            n_tasks=n_tasks,
            trials=1,
            errors=errors,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            wall_time_s=wall,
            harness=harness,
            note=note,
            per_task=per_task,
            tracker=tracker,
            warnings=warnings,
        )

    def note(self, cfg: dict[str, Any], n_tasks: int, harness: dict[str, Any]) -> str:
        params = cfg.get("request_params", {})
        t = params.get("temperature", "default")
        total = (cfg.get("harness") or {}).get("dataset_tasks_total")
        subset = (
            "full dataset"
            if (total and n_tasks >= total)
            else f"fixed subset of {n_tasks} (seed {SUBSET_SEED})"
        )
        s = (
            f"harbor {harness.get('harbor')} {self.dataset}, agent {AGENT}, {subset}, 1 attempt, "
            f"temp {t}, max_tokens harness default; errored trials count as unsolved"
        )
        exc = cfg.get("per_model_exceptions") or {}
        if exc:
            s += f"; per-model: {json.dumps(exc, sort_keys=True)}"
        if cfg.get("smoke"):
            s += "; SMOKE"
        return s


class TerminalBenchAdapter(HarborAdapter):
    key = "tb"
    display = "Terminal-Bench 2.x"
    dataset_setting = "tb_dataset"
    subset_prefix = "tb"
    tokens_in_per_task = 400_000
    tokens_out_per_task = 15_000


class SWEBenchAdapter(HarborAdapter):
    key = "swe"
    display = "SWE-bench Verified"
    dataset_setting = "swe_dataset"
    subset_prefix = "swe"
    tokens_in_per_task = 500_000
    tokens_out_per_task = 15_000
