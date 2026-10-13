"""OSWorld-Verified (xlang-ai) - 369 real-desktop tasks (Ubuntu VM), execution-based
scoring. Official harness: https://github.com/xlang-ai/OSWorld, pinned by commit
(``settings.osworld_ref``).

Needs VMs: the Docker provider requires KVM on the host (``/dev/kvm``), ~tens of
GB for the VM image, and the agent sends a screenshot per step, so the model
must accept images. ``prepare()`` refuses without Docker + KVM. Checked against
``scripts/python/run_multienv.py`` and ``show_result.py`` at the pinned commit:

``python scripts/python/run_multienv.py --provider_name docker --headless
--observation_type screenshot --model <name> --sleep_after_execution 3
--max_steps 15 --num_envs N --client_password password --result_dir R
--test_all_meta_path META``; results are ``R/pyautogui/screenshot/<model>/
<domain>/<id>/result.txt`` (0/1 per task), which the official ``show_result.py``
averages; re-running skips tasks that already have a result.txt.

Model dispatch: the baseline ``PromptAgent`` picks its client from the model-name
prefix and only the ``gpt*`` branch speaks plain OpenAI chat at ``OPENAI_BASE_URL``.
BenchLab therefore registers the alias ``gpt-benchlab-<id>`` on its per-run proxy
and passes that name; the upstream model is still the BenchLab model (recorded).
Harness defaults temperature 1.0 / top_p 0.9 / max_tokens 1500 / 15 steps apply
to every model and are recorded as deviations.
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
from benchlab.harness_env import (
    HarnessInstallError,
    ensure_pip_venv_in_repo,
    ensure_repo,
    repo_commit,
)
from benchlab.proxy import ProxyProcess
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

HARNESS_NAME = "osworld"
REPO = "https://github.com/xlang-ai/OSWorld.git"
ACTION_SPACE = "pyautogui"
OBS = "screenshot"
SMOKE = {
    "os": ["28cc3b7e-b194-4bc9-8353-d04c0f4d6735"],
    "chrome": ["bb5e4c0d-f964-439c-97b6-bdb9747de3f4"],
}


def alias_for(model: ModelSpec) -> str:
    return f"gpt-benchlab-{model.id}"


class OSWorldAdapter(BenchAdapter):
    key = "mimo_osw"
    tracker_key = "mimo_osw"
    display = "OSWorld-Verified"
    stage = 4
    needs_proxy = True
    needs_docker = True
    tokens_in_per_task = 40_000  # 15 screenshot steps
    tokens_out_per_task = 4_000

    @property
    def ref(self) -> str:
        return self.cfg.settings.osworld_ref

    def repo(self) -> Path:
        try:
            return ensure_repo(HARNESS_NAME, REPO, self.ref)
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def venv(self, repo: Path) -> Path:
        try:
            return ensure_pip_venv_in_repo(
                repo, "3.12", ["-r", str(repo / "requirements.txt")], editable_no_deps=False
            )
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def host_problems(self) -> list[str]:
        probs = []
        if not shutil.which("docker"):
            probs.append("docker not on PATH")
        if not Path("/dev/kvm").exists():
            probs.append(
                "/dev/kvm missing (the docker provider needs KVM; or use --provider aws/vmware per the OSWorld README)"
            )
        return probs

    def prepare(self) -> dict[str, Any]:
        probs = self.host_problems()
        if probs:
            raise HarnessNotReady(f"{self.key}: OSWorld needs VMs: " + "; ".join(probs))
        repo = self.repo()
        self.venv(repo)
        return {
            "name": "osworld",
            "commit": repo_commit(repo),
            "ref": self.ref,
            "provider": self.cfg.settings.osworld_provider,
            "max_steps": self.cfg.settings.osworld_max_steps,
            "observation": OBS,
            "action_space": ACTION_SPACE,
            "fixed_by_harness": {"temperature": 1.0, "top_p": 0.9, "max_tokens": 1500},
            "meta": self.cfg.settings.osworld_meta,
        }

    def meta_path(self, repo: Path, raw: Path, smoke: bool) -> Path:
        if smoke:
            p = raw / "test_smoke.json"
            p.write_text(json.dumps(SMOKE))
            return p
        return repo / "evaluation_examples" / self.cfg.settings.osworld_meta

    def n_tasks(self, repo: Path | None, smoke: bool) -> int:
        if smoke:
            return sum(len(v) for v in SMOKE.values())
        p = (repo / "evaluation_examples" / self.cfg.settings.osworld_meta) if repo else None
        if p and p.exists():
            return sum(len(v) for v in json.loads(p.read_text()).values())
        return 369

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        s = self.cfg.settings
        n = self.n_tasks(None, smoke)
        notes = [
            f"{s.osworld_meta} via {s.osworld_provider} provider, {s.osworld_max_steps} steps, screenshot-only agent (model must accept images)",
            "harness-fixed temperature 1.0 / top_p 0.9 / max_tokens 1500",
            f"num_envs {model.max_concurrency}",
        ]
        return RunPlan(n, 1, None, self.tokens_in_per_task, self.tokens_out_per_task, notes)

    def command(
        self, model: ModelSpec, result_dir: Path, meta: Path, python: str = "python"
    ) -> list[str]:
        s = self.cfg.settings
        return [
            python,
            "scripts/python/run_multienv.py",
            "--provider_name",
            s.osworld_provider,
            "--headless",
            "--observation_type",
            OBS,
            "--action_space",
            ACTION_SPACE,
            "--model",
            alias_for(model),
            "--sleep_after_execution",
            "3",
            "--max_steps",
            str(s.osworld_max_steps),
            "--num_envs",
            str(model.max_concurrency),
            "--client_password",
            s.osworld_client_password,
            "--result_dir",
            str(result_dir),
            "--test_all_meta_path",
            str(meta),
        ]

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        repo = self.repo()
        bin_dir = self.venv(repo)
        raw = run_dir.raw_dir
        results = raw / "results"
        meta = self.meta_path(repo, raw, smoke)
        with ProxyProcess(
            self.cfg, raw, run_dir.path / RunDir.USAGE, aliases={alias_for(model): model.id}
        ) as proxy:
            env = dict(os.environ)
            env.pop("VIRTUAL_ENV", None)
            env["OPENAI_BASE_URL"] = proxy.base_url
            env["OPENAI_API_KEY"] = proxy.master_key
            cmd = self.command(model, results, meta, str(bin_dir / "python"))
            (raw / "command.txt").write_text(" ".join(cmd) + "\n")
            run_logged(cmd, env, raw / "osworld.log", budget=budget, cwd=repo)
        return raw

    @staticmethod
    def read_results(results: Path, model_name: str) -> dict[str, dict[str, float]]:
        """{domain: {example_id: score}} exactly as show_result.py reads result.txt."""
        base = results / ACTION_SPACE / OBS / model_name
        out: dict[str, dict[str, float]] = {}
        if not base.exists():
            return out
        for dom in sorted(p for p in base.iterdir() if p.is_dir()):
            for ex in sorted(p for p in dom.iterdir() if p.is_dir()):
                rt = ex / "result.txt"
                if rt.exists():
                    txt = rt.read_text().strip()
                    try:
                        v = float(txt)
                    except ValueError:
                        v = float(txt.lower() == "true")
                    out.setdefault(dom.name, {})[ex.name] = v
        return out

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        model_id = cfg["model_id"]
        res = self.read_results(run_dir.raw_dir / "results", f"gpt-benchlab-{model_id}")
        expected = None
        meta = run_dir.raw_dir / "test_smoke.json" if cfg.get("smoke") else None
        if meta and meta.exists():
            expected = sum(len(v) for v in json.loads(meta.read_text()).values())
        return self.normalize(res, cfg, run_dir, expected)

    def normalize(
        self,
        res: dict[str, dict[str, float]],
        cfg: dict[str, Any],
        run_dir: RunDir,
        expected: int | None,
    ) -> NormalizedResult:
        scores = [v for d in res.values() for v in d.values()]
        n = len(scores)
        if n == 0:
            raise FileNotFoundError("no result.txt files under raw/results (see raw/osworld.log)")
        metrics: dict[str, float | None] = {"success_rate": sum(scores) / n * 100}
        for dom, d in sorted(res.items()):
            metrics[f"success_rate_{dom}"] = sum(d.values()) / len(d) * 100
        warnings = []
        total = expected or self.n_tasks(None, False)
        if n < total:
            warnings.append(
                f"only {n}/{total} tasks have a result.txt; re-run to resume the rest (score is over finished tasks only)"
            )
        errors = ErrorCounts(other=max(0, total - n))
        usage = run_dir.usage_log.summarize(cfg["model_id"])
        if not usage.n_requests:  # the proxy logs requests under the alias name
            usage = run_dir.usage_log.summarize(f"gpt-benchlab-{cfg['model_id']}")
        h = cfg.get("harness") or {}
        note = (
            f"OSWorld-Verified commit {str(h.get('commit') or '')[:8]}, {h.get('meta')} ({n} scored tasks), official run_multienv.py "
            f"PromptAgent screenshot/pyautogui, {h.get('max_steps')} steps, {h.get('provider')} VMs; harness-fixed temperature 1.0 / top_p 0.9 / "
            f"max_tokens 1500 (deviates from the common params); model called as proxy alias gpt-benchlab-{cfg['model_id']}"
        )
        if cfg.get("smoke"):
            note += "; SMOKE"
        model_id = cfg["model_id"]
        try:
            tracker_id = self.cfg.get(model_id).tracker
        except KeyError:
            tracker_id = model_id
        tracker = {
            f"{tracker_id}__{self.tracker_key}": {
                "v": f"{metrics['success_rate']:.1f}",
                "note": note,
            }
        }
        per_task = [
            {"id": ex, "domain": dom, "score": v} for dom, d in res.items() for ex, v in d.items()
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
