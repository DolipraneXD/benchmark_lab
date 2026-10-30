"""LiveCodeBench (code generation) via the official LiveCodeBench/LiveCodeBench repo.

Verified against commit 28fef95 (README + ``lcb_runner/runner/{parser,main,oai_runner}.py``):

* ``python -m lcb_runner.runner.main --model <name> --scenario codegeneration --evaluate``
  with ``--release_version``, ``--n``, ``--temperature``, ``--max_tokens``,
  ``--multiprocess``, ``--start_date/--end_date`` (problem release window),
  ``--continue_existing_with_eval`` (resume). Official protocol: n=10, temperature 0.2.
* The OpenAI runner reads ``OPENAI_KEY`` and builds ``OpenAI()`` without a base
  URL, so the SDK's ``OPENAI_BASE_URL`` env var routes it through our proxy.
* Models must be registered in ``lcb_runner.lm_styles`` (the README's documented
  extension point). We do that at runtime from a wrapper script - the model under
  test is added as ``LMStyle.OpenAIChat`` with the benchlab id as its API name -
  so the harness code is never edited.
* Scoring: the repo's own evaluator (``--evaluate``) writes
  ``output/<model>/Scenario.codegeneration_<n>_<temp>_eval_all.json`` with per
  problem ``pass@1``/``graded_list``/``contest_date``/``difficulty`` and
  ``..._eval.json`` with the summary; we aggregate pass@1 and record the window.
* Install: the repo declares ``vllm``/``torch`` as dependencies for local models;
  the API path never imports them, so the harness venv installs the other
  declared deps plus the repo with ``--no-deps`` (documented deviation).
"""

from __future__ import annotations

import json
import os
from datetime import date
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
from benchlab.llm import request_params
from benchlab.proxy import ProxyProcess
from benchlab.redact import redact
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

HARNESS_NAME = "livecodebench"
LCB_REPO = "https://github.com/LiveCodeBench/LiveCodeBench.git"
LCB_PYTHON = "3.11"
# declared in the repo's pyproject minus vllm/torch (vllm is unused on the API path; torch is imported by parser.py, CPU build is enough)
LCB_PACKAGES = [
    "annotated-types>=0.7.0",
    "anthropic>=0.42.0,<1",  # 1.x dropped HUMAN_PROMPT/AI_PROMPT which lcb_runner.prompts imports
    "cohere>=5.13.6",
    "datasets>=3.2.0,<4",  # 4.x dropped dataset loading scripts; LCB loads code_generation_lite.py
    "google-genai>=0.6.0",
    "mistralai==0.4.2",
    "openai>=1.59.6",
    "pebble>=5.1.0",
    "together>=0.21.0",
    "numpy",
    "tqdm",
    "torch",
]
RELEASE_END = {  # last contest month per README; a cutoff at/after it leaves no problems
    "release_v1": "2024-03-31",
    "release_v2": "2024-05-31",
    "release_v3": "2024-07-31",
    "release_v4": "2024-09-30",
    "release_v5": "2025-01-31",
    "release_v6": "2025-04-30",
}
RELEASE_SIZE = {
    "release_v1": 400,
    "release_v2": 511,
    "release_v3": 612,
    "release_v4": 713,
    "release_v5": 880,
    "release_v6": 1055,
}
SMOKE_PROBLEMS = 2

WRAPPER = r"""
import sys
from lcb_runner.lm_styles import LMStyle, LanguageModel, LanguageModelList, LanguageModelStore
name = sys.argv[1]
lm = LanguageModel(model_name=name, model_repr=name, model_style=LMStyle.OpenAIChat, release_date=None)
LanguageModelList.append(lm)
LanguageModelStore[name] = lm
sys.argv = ["lcb_runner.runner.main"] + sys.argv[2:]
from lcb_runner.runner.main import main
main()
"""


def link_package(repo: Path, raw: Path) -> None:
    """LCB opens its few-shot prompt files by paths relative to the folder it is started in
    (``lcb_runner/prompts/…``), and writes ``output/`` there. It runs in the run's raw folder
    (the install may be read-only), so that folder gets a ``lcb_runner`` link to the package."""
    link = raw / "lcb_runner"
    if not link.exists():
        link.symlink_to(repo / "lcb_runner", target_is_directory=True)


class LCBAdapter(BenchAdapter):
    key = "lcb"
    display = "LiveCodeBench"
    stage = 4
    needs_docker = False  # the official evaluator runs tests in subprocesses on the host
    tokens_in_per_task = 1500
    tokens_out_per_task = 2500

    # -- harness -------------------------------------------------------------------
    def harness(self) -> tuple[Path, Path]:
        os.environ.setdefault(
            "UV_TORCH_BACKEND", "cpu"
        )  # parser.py imports torch; CPU wheels suffice
        try:
            repo = ensure_repo(HARNESS_NAME, LCB_REPO, self.cfg.settings.lcb_ref)
            bin_dir = ensure_pip_venv_in_repo(repo, LCB_PYTHON, LCB_PACKAGES, editable_no_deps=True)
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e
        return repo, bin_dir

    def prepare(self) -> dict[str, Any]:
        repo, _ = self.harness()
        return {
            "name": "LiveCodeBench",
            "repo": LCB_REPO,
            "ref": self.cfg.settings.lcb_ref,
            "commit": repo_commit(repo),
            "release_version": self.cfg.settings.lcb_release_version,
            "n": self.cfg.settings.lcb_n,
            "temperature": self.cfg.settings.lcb_temperature,
            "dataset": "livecodebench/code_generation_lite (HuggingFace)",
            "install": "declared deps minus vllm/torch; repo installed --no-deps (API models only)",
        }

    # -- window ----------------------------------------------------------------------
    def window(self, model: ModelSpec) -> tuple[str | None, str | None, list[str]]:
        start = model.knowledge_cutoff or self.cfg.settings.lcb_default_start_date
        end = RELEASE_END.get(self.cfg.settings.lcb_release_version)
        notes = []
        if model.knowledge_cutoff:
            notes.append(f"problems after the model's cutoff {model.knowledge_cutoff}")
        elif start:
            notes.append(f"problems after settings.lcb_default_start_date {start}")
        else:
            notes.append(
                "no knowledge_cutoff set on the model: whole release used (contamination risk)"
            )
        if start and end and date.fromisoformat(start) >= date.fromisoformat(end):
            notes.append(
                f"WARNING: cutoff {start} is at/after the last problem ({end}) of "
                f"{self.cfg.settings.lcb_release_version}: no problems would remain"
            )
        return start, end, notes

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        start, end, notes = self.window(model)
        rv = self.cfg.settings.lcb_release_version
        total = RELEASE_SIZE.get(rv, 0)
        n = (
            SMOKE_PROBLEMS if smoke else total
        )  # the window reduces it; unknown until the dataset is loaded
        notes = [
            f"LiveCodeBench {rv} ({total} problems before windowing), n={self.cfg.settings.lcb_n} "
            f"temp={self.cfg.settings.lcb_temperature}"
        ] + notes
        est_mult = self.cfg.settings.lcb_n
        return RunPlan(
            n,
            1,
            None,
            self.tokens_in_per_task * est_mult,
            self.tokens_out_per_task * est_mult,
            notes,
        )

    # -- command -------------------------------------------------------------------
    def harness_args(self, model: ModelSpec, smoke: bool) -> list[str]:
        s = self.cfg.settings
        params = request_params(model, s)
        start, end, _ = self.window(model)
        args = [
            "--model",
            model.id,
            "--scenario",
            "codegeneration",
            "--evaluate",
            "--release_version",
            s.lcb_release_version,
            "--n",
            str(1 if smoke else s.lcb_n),
            "--temperature",
            str(s.lcb_temperature),
            "--max_tokens",
            str(params.get("max_tokens", 2000)),
            "--multiprocess",
            str(model.max_concurrency),
            "--openai_timeout",
            str(int(s.request_timeout_s)),
            "--continue_existing_with_eval",
        ]
        if start:
            args += ["--start_date", start]
        if smoke:
            # keep the smoke tiny: a 2-week window at the start of the release
            rv_start = {"release_v6": "2023-05-01"}.get(s.lcb_release_version, "2023-05-01")
            args += ["--start_date", start or rv_start]
            args += ["--end_date", self._plus_days(start or rv_start, 21)]
        return args

    @staticmethod
    def _plus_days(d: str, n: int) -> str:
        from datetime import timedelta

        return (date.fromisoformat(d) + timedelta(days=n)).isoformat()

    def build_command(
        self, model: ModelSpec, smoke: bool, bin_dir: Path | None = None
    ) -> list[str]:
        py = str(bin_dir / "python") if bin_dir else "python"
        return [py, "-c", WRAPPER, model.id, *self.harness_args(model, smoke)]

    # -- run / parse ---------------------------------------------------------------
    def output_prefix(self, run_dir: RunDir, model_id: str, smoke: bool) -> Path:
        s = self.cfg.settings
        n = 1 if smoke else s.lcb_n
        return (
            run_dir.raw_dir
            / "output"
            / model_id
            / f"Scenario.codegeneration_{n}_{s.lcb_temperature}"
        )

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        repo, bin_dir = self.harness()
        raw = run_dir.raw_dir
        link_package(repo, raw)
        with ProxyProcess(self.cfg, raw, run_dir.path / RunDir.USAGE) as proxy:
            env = dict(os.environ)
            env.pop("VIRTUAL_ENV", None)
            env["OPENAI_KEY"] = proxy.master_key  # LCB's own variable name
            env["OPENAI_API_KEY"] = proxy.master_key
            env["OPENAI_BASE_URL"] = proxy.base_url
            env["PYTHONPATH"] = str(repo)
            cmd = self.build_command(model, smoke, bin_dir)
            (raw / "command.txt").write_text(
                redact(" ".join(cmd[:2] + ["<wrapper>"] + cmd[3:])) + "\n"
            )
            run_logged(cmd, env, raw / "lcb.log", budget=budget, cwd=raw)  # output/ lands in raw/
        try:
            ev = Path(str(self.output_prefix(run_dir, model.id, smoke)) + "_eval_all.json")
            if ev.exists():
                done = sorted(str(x["question_id"]) for x in json.loads(ev.read_text()))
                st = run_dir.state()
                st["completed"] = sorted(set(st.get("completed", [])) | set(done))
                run_dir.save_state(st)
        except Exception:  # noqa: BLE001
            pass
        return raw

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        prefix = self.output_prefix(run_dir, cfg["model_id"], bool(cfg.get("smoke")))
        eval_all = Path(str(prefix) + "_eval_all.json")
        if not eval_all.exists():
            raise FileNotFoundError(f"no LCB evaluation at {eval_all}")
        rows = json.loads(eval_all.read_text())
        summary = None
        ev = Path(str(prefix) + "_eval.json")
        if ev.exists():
            try:
                summary = json.loads(ev.read_text())
            except json.JSONDecodeError:
                summary = None
        return self.normalize(rows, summary, cfg, run_dir)

    def normalize(
        self, rows: list[dict[str, Any]], summary: Any, cfg: dict[str, Any], run_dir: RunDir
    ) -> NormalizedResult:
        n = len(rows)
        p1 = [float(r.get("pass@1") or 0.0) for r in rows]
        by_diff: dict[str, list[float]] = {}
        errors = ErrorCounts()
        per_task = []
        dates = []
        for r in rows:
            d = r.get("difficulty") or "unknown"
            by_diff.setdefault(d, []).append(float(r.get("pass@1") or 0.0))
            outs = r.get("output_list") or []
            empties = sum(1 for o in outs if not o)
            errors.api_errors += empties
            if r.get("contest_date"):
                dates.append(str(r["contest_date"])[:10])
            per_task.append(
                {
                    "id": r.get("question_id"),
                    "difficulty": d,
                    "pass@1": r.get("pass@1"),
                    "graded": r.get("graded_list"),
                    "contest_date": str(r.get("contest_date"))[:10],
                    "empty_outputs": empties,
                }
            )
        metrics: dict[str, float | None] = {"pass@1": round(100.0 * sum(p1) / n, 4) if n else None}
        for d, vals in sorted(by_diff.items()):
            metrics[f"pass@1_{d}"] = round(100.0 * sum(vals) / len(vals), 4)
        if isinstance(summary, list) and summary and isinstance(summary[0], dict):
            for k, v in summary[0].items():  # harness summary (pass@1, pass@5, ... as fractions)
                if isinstance(v, (int, float)) and k.startswith("pass@") and k != "pass@1":
                    metrics[f"harness_{k}"] = round(100.0 * float(v), 4)
        metrics["n_empty_outputs"] = float(errors.api_errors)
        window = (min(dates) if dates else None, max(dates) if dates else None)
        agent_id = cfg["model_id"]
        usage = run_dir.usage_log.summarize(agent_id)
        tin, tout = usage.tokens_in, usage.tokens_out
        cost = usage.cost_usd if (usage.n_requests and usage.cost_known) else None
        warnings = []
        if n == 0:
            warnings.append("no problems in the date window")
        if n and errors.api_errors / max(1, n * int((cfg.get("harness") or {}).get("n", 1))) > 0.02:
            warnings.append(
                f"{errors.api_errors} empty generations: API errors or timeouts, check the endpoint"
            )
        h = cfg.get("harness") or {}
        harness = {
            "name": "LiveCodeBench",
            "commit": h.get("commit"),
            "release_version": h.get("release_version"),
            "n": h.get("n"),
            "temperature": h.get("temperature"),
            "window": window,
            "start_date_arg": cfg.get("model", {}).get("knowledge_cutoff")
            or self.cfg.settings.lcb_default_start_date,
        }
        note = self.note(cfg, n, window, h)
        try:
            tracker_id = self.cfg.get(agent_id).tracker
        except Exception:  # noqa: BLE001
            tracker_id = agent_id
        tracker = {}
        if metrics["pass@1"] is not None:
            tracker[f"{tracker_id}__{self.tracker_key}"] = {
                "v": f"{metrics['pass@1']:.1f}",
                "note": note,
            }
        return NormalizedResult(
            bench=self.key,
            model_id=agent_id,
            metrics=metrics,
            primary_metric="pass@1",
            n_tasks=n,
            trials=int(h.get("n") or 1),
            errors=errors,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            wall_time_s=run_dir.state().get("elapsed_s", 0.0),
            harness=harness,
            note=note,
            per_task=per_task,
            tracker=tracker,
            warnings=warnings,
        )

    @property
    def tracker_key(self) -> str:
        return self.key

    def note(self, cfg: dict[str, Any], n: int, window: tuple, h: dict[str, Any]) -> str:
        s = (
            f"LiveCodeBench {h.get('release_version')} ({str(h.get('commit') or '')[:7]}) codegeneration, "
            f"{n} problems in window {window[0]}..{window[1]}, n={h.get('n')} temp {h.get('temperature')} "
            f"(LCB official sampling), pass@1"
        )
        cutoff = (cfg.get("model") or {}).get("knowledge_cutoff")
        s += f", cutoff {cutoff}" if cutoff else ", no cutoff set"
        if cfg.get("smoke"):
            s += "; SMOKE"
        return s
