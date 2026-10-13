"""τ²-bench adapters (airline, retail, telecom) wrapping the official
``sierra-research/tau2-bench`` harness.

Verified against tau2-bench **v1.0.1** (the repo is now τ³-bench; its ``base``
task split is the full τ²-style task set, but v1.0.0 shipped 75+ task fixes so
scores are *not* comparable with τ²-era published numbers - see README). The
harness is installed the official way: ``git clone`` at the pinned tag +
``uv sync`` under ``harnesses/tau2/repo`` (the PyPI name ``tau2`` is an
unrelated package).

Per run we start a per-run LiteLLM proxy and call ``tau2 run`` with
``--agent-llm openai/<id>`` and ``--user-llm openai/<user id>``; tau2 itself
uses LiteLLM, so ``api_base``/``api_key`` ride in ``--agent-llm-args`` /
``--user-llm-args``. The user simulator is the FIXED model from
``settings.tau2_user_model`` for every run and is recorded in config/result.
``--save-to`` takes an absolute path and ``--auto-resume`` resumes from
``results.json``; metrics (pass^k, termination counts) come from tau2's own
``compute_metrics``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from benchlab.adapters._proc import run_logged
from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import MissingKeyError, ModelSpec
from benchlab.harness_env import HarnessInstallError, ensure_repo_env, frozen_versions, repo_commit
from benchlab.llm import request_params
from benchlab.proxy import ProxyProcess
from benchlab.redact import redact
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.subsets import load_subset, pick_subset
from benchlab.usage import BudgetGuard

HARNESS_NAME = "tau2"
TAU2_REPO = "https://github.com/sierra-research/tau2-bench"
TAU2_SEED_DEFAULT = 300  # tau2's own default simulation seed
SUBSET_SEED = 20261013  # benchlab's seed for picking the fixed task subset
SMOKE_TASKS = 2
SMOKE_MAX_STEPS = 20
BASE_SPLIT_SIZES = {
    "airline": 50,
    "retail": 114,
    "telecom": 114,
}  # split_tasks.json["base"], v1.0.1

_DUMP = r"""
import json, sys
from pathlib import Path
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics
r = Results.load(Path(sys.argv[1]))
m = compute_metrics(r)
sims = []
for s in r.simulations:
    tin = tout = n_tc = 0
    for msg in (s.messages or []):
        u = getattr(msg, "usage", None) or {}
        tin += int(u.get("prompt_tokens") or 0)
        tout += int(u.get("completion_tokens") or 0)
        if getattr(msg, "role", "") == "assistant" and getattr(msg, "tool_calls", None):
            n_tc += len(msg.tool_calls)
    sims.append({
        "task_id": s.task_id, "trial": s.trial, "seed": s.seed,
        "reward": (s.reward_info.reward if s.reward_info else None),
        "termination_reason": getattr(s.termination_reason, "value", str(s.termination_reason)),
        "agent_cost": s.agent_cost, "user_cost": s.user_cost, "duration": s.duration,
        "tokens_in": tin, "tokens_out": tout, "n_tool_calls": n_tc, "n_messages": len(s.messages or []),
    })
info = r.info.model_dump(mode="json")
for k in ("llm_args_agent", "llm_args_user"):  # never persist credentials
    v = info.get(k) or (info.get("agent_info") or {}).get(k)
for section in (info, info.get("agent_info") or {}, info.get("user_info") or {}):
    for key in list(section.keys()):
        if isinstance(section[key], dict):
            section[key].pop("api_key", None)
out = {
    "metrics": m.model_dump(mode="json"),
    "pass_hat_ks": {str(k): v for k, v in m.pass_hat_ks.items()},
    "simulations": sims,
    "info": info,
    "n_tasks": len(r.tasks),
}
print(json.dumps(out, default=str))
"""


def classify_termination(reason: str) -> str:
    """Map a tau2 TerminationReason to a benchlab error class ('' = normal end)."""
    r = (reason or "").lower()
    if r in ("user_stop", "agent_stop"):
        return ""
    if r == "timeout":
        return "timeout"
    if r in ("infrastructure_error", "unexpected_error", "context_window_exceeded"):
        return "api_error"
    if r in ("too_many_errors", "agent_error", "user_error", "max_steps"):
        return "other"  # the agent misbehaved; counted, not hidden, but not an infra error
    return "other"


class Tau2Adapter(BenchAdapter):
    domain: str = ""
    stage = 2
    needs_docker = False
    tokens_in_per_task = 60000
    tokens_out_per_task = 3000
    subsets_root: Path | None = None  # tests override

    # -- harness -------------------------------------------------------------------
    @property
    def ref(self) -> str:
        return self.cfg.settings.tau2_ref

    def harness(self) -> tuple[Path, Path]:
        try:
            return ensure_repo_env(HARNESS_NAME, TAU2_REPO, self.ref)
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def user_spec(self) -> ModelSpec:
        return self.cfg.get(self.cfg.settings.tau2_user_model)

    def prepare(self) -> dict[str, Any]:
        user = self.user_spec()
        try:
            user.api_key()
        except MissingKeyError as e:
            raise HarnessNotReady(f"{self.key}: user simulator model {user.id}: {e}") from e
        repo, bin_dir = self.harness()
        if not (bin_dir / "tau2").exists():
            raise HarnessNotReady(f"{self.key}: `tau2` CLI missing in {bin_dir}")
        split = repo / "data" / "tau2" / "domains" / self.domain / "split_tasks.json"
        if not split.exists():
            raise HarnessNotReady(f"{self.key}: {split} missing (data dir not in the clone?)")
        v = frozen_versions(bin_dir, ["tau2", "litellm"])
        return {
            "name": "tau2-bench",
            "repo": TAU2_REPO,
            "ref": self.ref,
            "commit": repo_commit(repo),
            "tau2": v.get("tau2"),
            "litellm": v.get("litellm"),
            "domain": self.domain,
            "task_split": "base",
            "user_model": {"id": user.id, "model": user.model, "provider": user.provider},
            "user_llm_args": self.user_llm_args(None),
            "max_steps": self.cfg.settings.tau2_max_steps,
            "seed": TAU2_SEED_DEFAULT,
            "comparability": "v1.x includes the τ³ task fixes; not comparable with τ²-paper numbers",
        }

    # -- tasks -----------------------------------------------------------------------
    def base_task_ids(self) -> list[str]:
        repo, _ = self.harness()
        split = json.loads(
            (repo / "data" / "tau2" / "domains" / self.domain / "split_tasks.json").read_text()
        )
        return [str(t) for t in split["base"]]

    def subset_spec(self, subset: str | None) -> dict[str, Any] | None:
        """Fixed seeded subset of the base split (None = whole base split)."""
        if subset == "all":
            return None
        n = (
            int(subset)
            if (subset and subset.isdigit())
            else self.cfg.settings.tau2_tasks_per_domain
        )
        if n >= BASE_SPLIT_SIZES.get(self.domain, 10**9):
            return None
        name = f"tau2_{self.domain}_{n}"
        existing = load_subset(name, self.subsets_root)
        if existing:
            return existing
        return pick_subset(
            name,
            self.base_task_ids(),
            n,
            SUBSET_SEED,
            root=self.subsets_root,
            source=f"tau2-bench {self.ref} {self.domain} base split",
        )

    def task_ids(self, subset: str | None, smoke: bool) -> list[str] | None:
        spec = self.subset_spec(subset)
        ids = spec["ids"] if spec else None
        if smoke:
            pool = ids or self.base_task_ids()
            return pool[:SMOKE_TASKS]
        return ids

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        notes = [f"user sim: {self.cfg.settings.tau2_user_model}", f"tau2-bench {self.ref}"]
        if smoke:
            return RunPlan(
                SMOKE_TASKS,
                1,
                None,
                self.tokens_in_per_task // 10,
                self.tokens_out_per_task // 10,
                notes + [f"smoke: {SMOKE_TASKS} tasks, 1 trial, max_steps {SMOKE_MAX_STEPS}"],
            )
        try:
            ids = self.task_ids(subset, smoke)
            n = len(ids) if ids else BASE_SPLIT_SIZES.get(self.domain, 0)
            notes.append(
                "full base split" if ids is None else f"fixed subset of {n} (seed {SUBSET_SEED})"
            )
        except (HarnessNotReady, HarnessInstallError, OSError, KeyError) as e:
            ids = None
            n = (
                int(subset)
                if subset and subset.isdigit()
                else self.cfg.settings.tau2_tasks_per_domain
            )
            if subset == "all":
                n = BASE_SPLIT_SIZES.get(self.domain, 0)
            notes.append(
                f"subset picked on first run (seed {SUBSET_SEED}); harness not set up yet ({type(e).__name__})"
            )
        return RunPlan(n, trials, ids, self.tokens_in_per_task, self.tokens_out_per_task, notes)

    # -- command ---------------------------------------------------------------------
    def _llm_args(
        self, spec: ModelSpec, proxy_url: str, master_key: str, overrides: dict[str, Any] | None
    ) -> dict:
        args = request_params(spec, self.cfg.settings)
        for k, v in (overrides or {}).items():
            if v is None:
                args.pop(k, None)
            else:
                args[k] = v
        args["api_base"] = proxy_url
        args["api_key"] = master_key
        return args

    def agent_llm_args(self, model: ModelSpec, proxy_url: str = "", master_key: str = "") -> dict:
        return self._llm_args(model, proxy_url, master_key, None)

    def user_llm_args(self, proxy_url: str | None, master_key: str = "") -> dict:
        args = self._llm_args(
            self.user_spec(), proxy_url or "", master_key, self.cfg.settings.tau2_user_model_params
        )
        if proxy_url is None:  # for config.json: no transport details
            args.pop("api_base", None)
            args.pop("api_key", None)
        return args

    def build_command(
        self,
        model: ModelSpec,
        save_to: Path,
        subset: str | None,
        trials: int,
        smoke: bool,
        proxy_url: str,
        master_key: str,
        bin_dir: Path | None = None,
    ) -> list[str]:
        exe = str(bin_dir / "tau2") if bin_dir else "tau2"
        user = self.user_spec()
        cmd = [
            exe,
            "run",
            "--domain",
            self.domain,
            "--task-split-name",
            "base",
            "--agent-llm",
            f"openai/{model.id}",
            "--agent-llm-args",
            json.dumps(self.agent_llm_args(model, proxy_url, master_key)),
            "--user-llm",
            f"openai/{user.id}",
            "--user-llm-args",
            json.dumps(self.user_llm_args(proxy_url, master_key)),
            "--num-trials",
            str(1 if smoke else trials),
            "--max-concurrency",
            str(model.max_concurrency),
            "--max-steps",
            str(SMOKE_MAX_STEPS if smoke else self.cfg.settings.tau2_max_steps),
            "--seed",
            str(TAU2_SEED_DEFAULT),
            "--save-to",
            str(save_to),
            "--auto-resume",
            "--log-level",
            "ERROR",
        ]
        ids = self.task_ids(subset, smoke)
        if ids:
            cmd += ["--task-ids", *ids]
        return cmd

    # -- run / parse -------------------------------------------------------------------
    def results_dir(self, run_dir: RunDir) -> Path:
        return run_dir.raw_dir / f"tau2_{self.domain}"

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        repo, bin_dir = self.harness()
        raw = run_dir.raw_dir
        with ProxyProcess(self.cfg, raw, run_dir.path / RunDir.USAGE) as proxy:
            cmd = self.build_command(
                model,
                self.results_dir(run_dir).resolve(),
                subset,
                trials,
                smoke,
                proxy.base_url,
                proxy.master_key,
                bin_dir,
            )
            env = dict(os.environ)
            env.pop("VIRTUAL_ENV", None)
            env.pop("TAU2_DATA_DIR", None)
            env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
            env.update(proxy.env_for_harness())
            (raw / "command.txt").write_text(redact(" ".join(cmd)) + "\n")
            run_logged(cmd, env, raw / "tau2.log", budget=budget, cwd=repo)
        try:
            dump = self.read_results(run_dir, bin_dir)
            done = sorted({str(s["task_id"]) for s in dump["simulations"]})
            st = run_dir.state()
            st["completed"] = sorted(set(st.get("completed", [])) | set(done))
            run_dir.save_state(st)
        except Exception:  # noqa: BLE001
            pass
        return raw

    def read_results(self, run_dir: RunDir, bin_dir: Path) -> dict[str, Any]:
        results = self.results_dir(run_dir) / "results.json"
        if not results.exists():
            raise FileNotFoundError(f"no tau2 results at {results}")
        cached = run_dir.raw_dir / f"tau2_{self.domain}_dump.json"
        if cached.exists() and cached.stat().st_mtime >= results.stat().st_mtime:
            return json.loads(cached.read_text())
        import subprocess

        out = subprocess.run(
            [str(bin_dir / "python"), "-I", "-c", _DUMP, str(results)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if out.returncode != 0:
            raise RuntimeError(f"reading {results} failed: {redact(out.stderr[-2000:])}")
        cached.write_text(out.stdout)
        return json.loads(out.stdout)

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        _, bin_dir = self.harness()
        return self.normalize(self.read_results(run_dir, bin_dir), cfg, run_dir)

    def normalize(
        self, dump: dict[str, Any], cfg: dict[str, Any], run_dir: RunDir
    ) -> NormalizedResult:
        m = dump.get("metrics") or {}
        sims = dump.get("simulations") or []
        pass_ks = {int(k): float(v) for k, v in (dump.get("pass_hat_ks") or {}).items()}
        metrics: dict[str, float | None] = {}
        for k in sorted(pass_ks):
            metrics[f"pass^{k}"] = round(pass_ks[k] * 100.0, 4)
        if "avg_reward" in m:
            metrics["avg_reward"] = round(float(m["avg_reward"]) * 100.0, 4)
        for key in (
            "termination_user_stop",
            "termination_agent_stop",
            "termination_max_steps",
            "termination_error",
            "termination_infrastructure_error",
            "infra_error_count",
        ):
            if key in m:
                metrics[f"n_{key}"] = float(m[key])

        errors = ErrorCounts()
        per_task = []
        for s in sims:
            cls = classify_termination(s.get("termination_reason", ""))
            if cls == "timeout":
                errors.timeouts += 1
            elif cls == "api_error":
                errors.api_errors += 1
            elif cls == "other":
                errors.other += 1
            per_task.append(
                {
                    "id": s["task_id"],
                    "trial": s.get("trial"),
                    "reward": s.get("reward"),
                    "termination_reason": s.get("termination_reason"),
                    "duration_s": s.get("duration"),
                    "tokens_in": s.get("tokens_in"),
                    "tokens_out": s.get("tokens_out"),
                    "n_tool_calls": s.get("n_tool_calls"),
                    "error": cls or None,
                }
            )

        info = dump.get("info") or {}
        trials = int(info.get("num_trials") or cfg.get("trials") or 1)
        n_tasks = int(
            m.get("total_tasks") or dump.get("n_tasks") or len({s["task_id"] for s in sims})
        )

        # tokens/cost for the model under test only (the user simulator is logged separately)
        agent_id = cfg["model_id"]
        usage_agent = run_dir.usage_log.summarize(agent_id)
        usage_user = (
            run_dir.usage_log.summarize(self.cfg.settings.tau2_user_model)
            if self.cfg.settings.tau2_user_model != agent_id
            else None
        )
        if usage_agent.n_requests:
            tin, tout = usage_agent.tokens_in, usage_agent.tokens_out
            cost = usage_agent.cost_usd if usage_agent.cost_known else None
        else:  # parsing without the proxy log: tau2's per-message usage covers agent+user together
            tin = sum(int(s.get("tokens_in") or 0) for s in sims)
            tout = sum(int(s.get("tokens_out") or 0) for s in sims)
            try:
                cost = self.cfg.get(agent_id).cost_usd(tin, tout)
            except Exception:  # noqa: BLE001
                cost = None
        if usage_user is not None and usage_user.n_requests and usage_user.cost_known:
            metrics["user_sim_cost_usd"] = round(usage_user.cost_usd, 6)

        warnings = []
        infra = int(m.get("infra_error_count") or 0)
        if sims and infra / len(sims) > 0.02:
            warnings.append(
                f"{infra}/{len(sims)} simulations hit infrastructure errors (excluded from pass^k by tau2): check the endpoint"
            )
        if sims and errors.total / len(sims) > 0.3:
            warnings.append(
                f"{errors.total}/{len(sims)} simulations ended abnormally (timeout/errors/max_steps)"
            )

        wall = sum(float(s.get("duration") or 0) for s in sims)
        harness = {
            "name": "tau2-bench",
            "ref": self.ref,
            "commit": (cfg.get("harness") or {}).get("commit"),
            "tau2": (cfg.get("harness") or {}).get("tau2"),
            "domain": self.domain,
            "task_split": "base",
            "agent_llm": (info.get("agent_info") or {}).get("llm"),
            "user_llm": (info.get("user_info") or {}).get("llm"),
            "max_steps": (cfg.get("harness") or {}).get("max_steps"),
            "results": str(self.results_dir(run_dir) / "results.json"),
        }
        user = self.user_spec()
        note = self.note(cfg, n_tasks, trials, user, pass_ks)
        try:
            tracker_id = self.cfg.get(agent_id).tracker
        except Exception:  # noqa: BLE001
            tracker_id = agent_id
        tracker = {}
        if "pass^1" in metrics:
            tracker[f"{tracker_id}__{self.key}"] = {"v": f"{metrics['pass^1']:.1f}", "note": note}
        return NormalizedResult(
            bench=self.key,
            model_id=agent_id,
            metrics=metrics,
            primary_metric="pass^1",
            n_tasks=n_tasks,
            trials=trials,
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

    def note(self, cfg, n_tasks, trials, user: ModelSpec, pass_ks: dict[int, float]) -> str:
        params = cfg.get("request_params", {})
        t = params.get("temperature", "default")
        ut = self.user_llm_args(None).get("temperature", "default")
        s = (
            f"tau2-bench {self.ref} {self.domain} base split, {n_tasks} tasks"
            f"{'' if n_tasks >= BASE_SPLIT_SIZES.get(self.domain, 0) else f' (fixed subset, seed {SUBSET_SEED})'}, "
            f"{trials} trial{'s' if trials != 1 else ''}, pass^1; user sim {user.id} ({user.model}) temp {ut}; "
            f"agent temp {t}, max_steps {self.cfg.settings.tau2_max_steps}"
        )
        extra = [f"pass^{k}={v * 100:.1f}" for k, v in sorted(pass_ks.items()) if k > 1]
        if extra:
            s += "; " + ", ".join(extra)
        exc = cfg.get("per_model_exceptions") or {}
        if exc:
            s += f"; per-model: {json.dumps(exc, sort_keys=True)}"
        if cfg.get("smoke"):
            s += "; SMOKE"
        return s


class Tau2Airline(Tau2Adapter):
    key = "t2air"
    display = "τ²-bench airline"
    domain = "airline"


class Tau2Retail(Tau2Adapter):
    key = "t2ret"
    display = "τ²-bench retail"
    domain = "retail"


class Tau2Telecom(Tau2Adapter):
    key = "t2tel"
    display = "τ²-bench telecom"
    domain = "telecom"
