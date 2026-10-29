"""Aider Polyglot via the official aider benchmark harness, in Docker.

Verified against Aider-AI/aider **v0.86.2** (``benchmark/README.md``,
``benchmark/benchmark.py``, ``benchmark/docker.sh``). The harness runs inside
the ``aider-benchmark`` image built from the repo's ``benchmark/Dockerfile``;
exercises come from ``Aider-AI/polyglot-benchmark`` (pinned commit). aider uses
LiteLLM, so ``--model openai/<id>`` with ``OPENAI_API_BASE`` pointing at our
per-run proxy routes every request through BenchLab. The leaderboard number is
``pass_rate_2`` (percent of exercises passing all tests within two tries);
``benchmark.py --stats`` produces it and we parse that output, cross-checked
against the per-exercise ``.aider.results.json`` files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from benchlab.adapters._proc import run_logged
from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import ModelSpec
from benchlab.harness_env import HarnessInstallError, building_image, ensure_repo, repo_commit
from benchlab.llm import request_params
from benchlab.proxy import ProxyProcess
from benchlab.redact import redact
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

HARNESS_NAME = "aider"
AIDER_REPO = "https://github.com/Aider-AI/aider.git"
POLYGLOT_REPO = "https://github.com/Aider-AI/polyglot-benchmark.git"
EXERCISES_DIRNAME = "polyglot-benchmark"
RUN_NAME = "benchlab"  # aider prefixes it with a timestamp; --cont finds the single prior
SMOKE_TESTS = 3
POLYGLOT_TOTAL = 225


def docker_available() -> bool:
    exe = shutil.which("docker")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def parse_stats_output(text: str) -> dict[str, Any]:
    """Parse the yaml-ish block ``benchmark.py --stats`` prints (``  key: value`` lines)."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*-?\s*([a-z_0-9]+):\s*(.+?)\s*$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        try:
            out[k] = float(v) if re.match(r"^-?\d+(\.\d+)?$", v) else v
        except ValueError:
            out[k] = v
    return out


class AiderAdapter(BenchAdapter):
    key = "aider"
    display = "Aider Polyglot"
    stage = 4
    needs_docker = True
    tokens_in_per_task = 8000
    tokens_out_per_task = 2000

    # -- harness -------------------------------------------------------------------
    @property
    def image(self) -> str:
        return f"benchlab-aider:{self.cfg.settings.aider_ref}"

    def repos(self) -> tuple[Path, Path]:
        try:
            aider = ensure_repo(
                HARNESS_NAME, AIDER_REPO, self.cfg.settings.aider_ref, subdir="repo"
            )
            polyglot = ensure_repo(
                HARNESS_NAME,
                POLYGLOT_REPO,
                self.cfg.settings.aider_polyglot_ref,
                subdir=EXERCISES_DIRNAME,
            )
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e
        return aider, polyglot

    def image_exists(self) -> bool:
        return (
            subprocess.run(
                ["docker", "image", "inspect", self.image], capture_output=True
            ).returncode
            == 0
        )

    def build_image(self, aider_repo: Path, log: Path | None = None) -> None:
        cmd = ["docker", "build", "--file", "benchmark/Dockerfile", "-t", self.image, "."]
        out = subprocess.run(cmd, cwd=str(aider_repo), capture_output=True, text=True)
        if log:
            log.write_text(out.stdout + out.stderr)
        if out.returncode != 0:
            raise HarnessNotReady(f"{self.key}: docker build failed: {redact(out.stderr[-2000:])}")

    def prepare(self) -> dict[str, Any]:
        if not docker_available() and not building_image():
            raise HarnessNotReady(
                f"{self.key}: Docker daemon not reachable (the aider benchmark executes model-written code in a container)"
            )
        aider, polyglot = self.repos()
        if not building_image() and not self.image_exists():
            self.build_image(aider)  # inside containers this image is built ahead by compose
        return {
            "name": "aider-benchmark",
            "repo": AIDER_REPO,
            "ref": self.cfg.settings.aider_ref,
            "commit": repo_commit(aider),
            "polyglot_repo": POLYGLOT_REPO,
            "polyglot_commit": repo_commit(polyglot),
            "image": self.image,
            "edit_format": self.cfg.settings.aider_edit_format,
            "tries": self.cfg.settings.aider_tries,
            "docker_network": self.cfg.settings.aider_docker_network,
        }

    # -- plan / command ------------------------------------------------------------
    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        n = (
            SMOKE_TESTS
            if smoke
            else (int(subset) if subset and subset.isdigit() else POLYGLOT_TOTAL)
        )
        notes = [
            f"aider {self.cfg.settings.aider_ref} polyglot, edit-format {self.cfg.settings.aider_edit_format}, "
            f"{self.cfg.settings.aider_tries} tries"
        ]
        if smoke:
            notes.append(f"smoke: --num-tests {SMOKE_TESTS}")
        elif subset and subset.isdigit():
            notes.append(f"--num-tests {subset} (first N in aider's order)")
        return RunPlan(n, 1, None, self.tokens_in_per_task, self.tokens_out_per_task, notes)

    def benchmark_args(self, model: ModelSpec, subset: str | None, smoke: bool) -> list[str]:
        s = self.cfg.settings
        args = [
            RUN_NAME,
            "--model",
            f"openai/{model.id}",
            "--edit-format",
            s.aider_edit_format,
            "--threads",
            str(model.max_concurrency),
            "--tries",
            str(s.aider_tries),
            "--exercises-dir",
            EXERCISES_DIRNAME,
            "--cont",
        ]
        if smoke:
            args += ["--num-tests", str(SMOKE_TESTS)]
        elif subset and subset.isdigit():
            args += ["--num-tests", subset]
        elif subset:
            args += ["--keywords", subset]
        return args

    def docker_command(
        self, aider_repo: Path, benchmarks_dir: Path, inner: str, interactive_env: dict[str, str]
    ) -> list[str]:
        # inside a run container: join the run's Docker network (the proxy is reachable by
        # the run container's name) and use the code baked into the image, since the
        # harness folder is not a host path
        in_container = bool(os.environ.get("BENCHLAB_ADVERTISE_HOST"))
        network = (
            os.environ.get("BENCHLAB_DOCKER_NETWORK") or self.cfg.settings.aider_docker_network
        )
        cmd = [
            "docker",
            "run",
            "--rm",
            "--memory=12g",
            "--memory-swap=12g",
            f"--network={network if in_container else self.cfg.settings.aider_docker_network}",
            "--add-host=host.docker.internal:host-gateway",
        ]
        if not in_container:
            cmd += ["-v", f"{aider_repo}:/aider"]
        cmd += [
            "-v",
            f"{benchmarks_dir}:/benchmarks",
            "-e",
            "AIDER_DOCKER=1",
            "-e",
            "AIDER_BENCHMARK_DIR=/benchmarks",
        ]
        for k in sorted(interactive_env):
            cmd += ["-e", k]  # value comes from the environment, never from the command line
        cmd += [self.image, "bash", "-lc", inner]
        return cmd

    def inner_run_script(self, model: ModelSpec, subset: str | None, smoke: bool) -> str:
        args = " ".join(self.benchmark_args(model, subset, smoke))
        return f"pip install -q -e '.[dev]' >/dev/null 2>&1; ./benchmark/benchmark.py {args}"

    # -- run -------------------------------------------------------------------------
    def benchmarks_dir(self, run_dir: RunDir) -> Path:
        return run_dir.raw_dir / "benchmarks"

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        aider, polyglot = self.repos()
        raw = run_dir.raw_dir
        bdir = self.benchmarks_dir(run_dir)
        bdir.mkdir(exist_ok=True)
        if not (bdir / EXERCISES_DIRNAME).exists():
            shutil.copytree(
                polyglot, bdir / EXERCISES_DIRNAME, ignore=shutil.ignore_patterns(".git")
            )
        with ProxyProcess(self.cfg, raw, run_dir.path / RunDir.USAGE) as proxy:
            env = dict(os.environ)
            env["OPENAI_API_KEY"] = proxy.master_key
            env["OPENAI_API_BASE"] = proxy.base_url
            env["OPENAI_BASE_URL"] = proxy.base_url
            params = request_params(model, self.cfg.settings)
            # aider reads sampling params from its model settings; temperature is passed via env
            if "temperature" in params:
                env["AIDER_TEMPERATURE"] = str(params["temperature"])
            cmd = self.docker_command(
                aider,
                bdir.resolve(),
                self.inner_run_script(model, subset, smoke),
                {
                    "OPENAI_API_KEY": "",
                    "OPENAI_API_BASE": "",
                    "OPENAI_BASE_URL": "",
                    **({"AIDER_TEMPERATURE": ""} if "temperature" in params else {}),
                },
            )
            (raw / "command.txt").write_text(redact(" ".join(cmd)) + "\n")
            run_logged(cmd, env, raw / "aider.log", budget=budget, cwd=raw)
            # official stats, computed by the harness itself
            tdir = self.testdir(run_dir)
            if tdir is not None:
                stats_cmd = self.docker_command(
                    aider,
                    bdir.resolve(),
                    f"pip install -q -e '.[dev]' >/dev/null 2>&1; ./benchmark/benchmark.py --stats /benchmarks/{tdir.name}",
                    {},
                )
                out = subprocess.run(stats_cmd, capture_output=True, text=True, cwd=str(raw))
                (raw / "stats.txt").write_text(out.stdout + "\n" + out.stderr)
        try:
            done = sorted(p.parent.name for p in self.result_files(run_dir))
            st = run_dir.state()
            st["completed"] = sorted(set(st.get("completed", [])) | set(done))
            run_dir.save_state(st)
        except Exception:  # noqa: BLE001
            pass
        return raw

    def testdir(self, run_dir: RunDir) -> Path | None:
        cands = sorted(self.benchmarks_dir(run_dir).glob(f"*--{RUN_NAME}"))
        return cands[-1] if cands else None

    def result_files(self, run_dir: RunDir) -> list[Path]:
        tdir = self.testdir(run_dir)
        if tdir is None:
            return []
        return sorted(tdir.glob("*/exercises/practice/*/.aider.results.json"))

    # -- parse -----------------------------------------------------------------------
    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        files = self.result_files(run_dir)
        if not files:
            raise FileNotFoundError(f"no .aider.results.json under {self.benchmarks_dir(run_dir)}")
        rows = []
        for p in files:
            try:
                d = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            d["_lang"] = p.parts[-5] if len(p.parts) >= 5 else ""
            rows.append(d)
        stats_txt = run_dir.raw_dir / "stats.txt"
        stats = parse_stats_output(stats_txt.read_text()) if stats_txt.exists() else {}
        return self.normalize(rows, stats, cfg, run_dir)

    def normalize(
        self,
        rows: list[dict[str, Any]],
        stats: dict[str, Any],
        cfg: dict[str, Any],
        run_dir: RunDir,
    ) -> NormalizedResult:
        tries = int(self.cfg.settings.aider_tries)
        n = len(rows)
        passed = [0] * tries
        errors = ErrorCounts()
        per_task = []
        tin = tout = 0
        cost_h = 0.0
        for r in rows:
            outcomes = list(r.get("tests_outcomes") or [])
            for i in range(tries):
                if any(outcomes[: i + 1]):
                    passed[i] += 1
            errors.api_errors += int(r.get("num_error_outputs") or 0)
            errors.malformed_tool_calls += int(r.get("num_malformed_responses") or 0)
            errors.other += int(r.get("test_timeouts") or 0) + int(
                r.get("num_exhausted_context_windows") or 0
            )
            tin += int(r.get("prompt_tokens") or 0)
            tout += int(r.get("completion_tokens") or 0)
            cost_h += float(r.get("cost") or 0)
            per_task.append(
                {
                    "id": f"{r.get('_lang')}/{r.get('testcase')}",
                    "language": r.get("_lang"),
                    "tests_outcomes": outcomes,
                    "passed": any(outcomes),
                    "duration_s": r.get("duration"),
                    "tokens_in": r.get("prompt_tokens"),
                    "tokens_out": r.get("completion_tokens"),
                    "malformed_responses": r.get("num_malformed_responses"),
                    "error_outputs": r.get("num_error_outputs"),
                }
            )
        metrics: dict[str, float | None] = {}
        for i in range(tries):
            key = f"pass_rate_{i + 1}"
            metrics[key] = (
                float(stats[key])
                if key in stats
                else (round(100.0 * passed[i] / n, 4) if n else None)
            )
        metrics["percent_cases_well_formed"] = (
            float(stats["percent_cases_well_formed"])
            if "percent_cases_well_formed" in stats
            else None
        )
        for k in (
            "error_outputs",
            "num_malformed_responses",
            "syntax_errors",
            "test_timeouts",
            "exhausted_context_windows",
        ):
            if k in stats:
                metrics[f"n_{k}"] = float(stats[k])
        primary = f"pass_rate_{tries}"
        agent_id = cfg["model_id"]
        usage = run_dir.usage_log.summarize(agent_id)
        if usage.n_requests:
            tin, tout = usage.tokens_in, usage.tokens_out
            cost = usage.cost_usd if usage.cost_known else None
        else:
            try:
                cost = self.cfg.get(agent_id).cost_usd(tin, tout)
            except Exception:  # noqa: BLE001
                cost = None
        warnings = []
        if (
            n
            and n < POLYGLOT_TOTAL
            and not cfg.get("smoke")
            and not (cfg.get("subset") or {}).get("name")
        ):
            warnings.append(
                f"only {n}/{POLYGLOT_TOTAL} exercises have results; run again to resume (--cont)"
            )
        if n and errors.api_errors / n > 0.05:
            warnings.append(
                f"{errors.api_errors} LLM error outputs over {n} exercises: check the endpoint"
            )
        harness = {
            "name": "aider-benchmark",
            **{k: v for k, v in (cfg.get("harness") or {}).items() if k != "name"},
            "stats": stats or None,
            "testdir": str(self.testdir(run_dir)),
        }
        note = self.note(cfg, n, tries)
        try:
            tracker_id = self.cfg.get(agent_id).tracker
        except Exception:  # noqa: BLE001
            tracker_id = agent_id
        tracker = {}
        if metrics.get(primary) is not None:
            tracker[f"{tracker_id}__{self.key}"] = {"v": f"{metrics[primary]:.1f}", "note": note}
        wall = sum(float(r.get("duration") or 0) for r in rows)
        return NormalizedResult(
            bench=self.key,
            model_id=agent_id,
            metrics=metrics,
            primary_metric=primary,
            n_tasks=n,
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

    def note(self, cfg: dict[str, Any], n: int, tries: int) -> str:
        s = self.cfg.settings
        t = (cfg.get("request_params") or {}).get("temperature", "default")
        base = (
            f"aider {s.aider_ref} polyglot benchmark (docker), {n} exercises, edit-format {s.aider_edit_format}, "
            f"pass_rate_{tries} ({tries} tries), temp {t}"
        )
        exc = cfg.get("per_model_exceptions") or {}
        if exc:
            base += f"; per-model: {json.dumps(exc, sort_keys=True)}"
        if cfg.get("smoke"):
            base += "; SMOKE"
        return base
