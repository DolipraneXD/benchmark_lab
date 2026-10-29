"""Adapters for Inspect-based benchmarks (``inspect_evals``): GPQA Diamond, BFCL,
GAIA and BrowseComp.

Each run launches a per-run LiteLLM proxy, then ``inspect eval-set`` against it
with ``openai/<benchlab id>``. Inspect does the prompting and the official
scoring; we only build the command, pin versions, and read the ``.eval`` log.

The harness lives in its own venv (``harnesses/inspect/.venv``) because
inspect-ai requires ``openai>=3.4`` while LiteLLM pins ``openai<3``. Logs are
read through that venv (``inspect_ai.log.read_eval_log``) and handed back as JSON.

Verified against inspect-ai 0.3.279 / inspect-evals 0.24.0 (READMEs of each
eval in the inspect_evals repo; notable: gpqa_diamond defaults to 4 epochs and a
seeded choice shuffle, bfcl's default run is V1+V2+V3 = 4,981 samples, browse_comp
grades with the model under test unless ``scorer_model`` is given).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import ModelSpec
from benchlab.harness_env import (
    HarnessInstallError,
    building_image,
    ensure_venv,
    frozen_versions,
    run_python,
)
from benchlab.llm import request_params
from benchlab.proxy import ProxyProcess
from benchlab.redact import redact
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.subsets import load_subset, pick_subset
from benchlab.usage import BudgetExceeded, BudgetGuard

HARNESS_NAME = "inspect"
HARNESS_PYTHON = "3.12"
PINNED = {"inspect-ai": "0.3.279", "inspect-evals": "0.24.0"}
# openai is an optional dependency of inspect-ai (its openai/ provider needs >=3.4)
PACKAGES = ["inspect-ai==0.3.279", "inspect-evals[bfcl,gaia]==0.24.0", "openai==3.29.0"]
SMOKE_LIMIT = 3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def docker_available() -> bool:
    exe = shutil.which("docker")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _yaml_list(items: list[str]) -> str:
    return "[" + ",".join(f"'{i}'" for i in items) + "]"


def classify_error_message(msg: str) -> str:
    """Map an Inspect sample error message to a benchlab error class."""
    m = (msg or "").lower()
    if "timeout" in m or "timed out" in m:
        return "timeout"
    if "rate limit" in m or "ratelimit" in m or "429" in m:
        return "rate_limit"
    if "tool" in m and ("parse" in m or "invalid" in m or "malformed" in m or "json" in m):
        return "malformed_tool_call"
    api_markers = (
        "api",
        "connection",
        "status",
        "http",
        "500",
        "502",
        "503",
        "bad request",
        "401",
        "403",
    )
    if any(k in m for k in api_markers):
        return "api_error"
    return "other"


def _ns(obj: Any) -> Any:
    """Recursive attribute view over the JSON dump of an EvalLog."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


_DUMP_LOG = """
import json, sys
from inspect_ai.log import read_eval_log
log = read_eval_log(sys.argv[1])
d = log.model_dump(mode="json", exclude={"samples": {"__all__": {"events", "store"}}})
LIMIT = 6000

def short(x, att):
    # large message parts are stored once as attachment://<id>; put the text back
    if isinstance(x, str):
        if x.startswith("attachment://"):
            x = att.get(x[len("attachment://"):], x)
        return x if len(x) <= LIMIT else x[:LIMIT] + " [...]"
    if isinstance(x, list):
        return [short(v, att) for v in x]
    if isinstance(x, dict):
        return {k: short(v, att) for k, v in x.items()}
    return x

# keep what the per-test view needs (prompts, answer, tool calls, scores), trimmed
for s in d.get("samples") or []:
    att = s.pop("attachments", None) or {}
    msgs = []
    for m in s.get("messages") or []:
        tcs = [{"function": tc.get("function"), "arguments": short(tc.get("arguments"), att),
                "parse_error": tc.get("parse_error")} for tc in (m.get("tool_calls") or [])]
        msgs.append({"role": m.get("role"), "content": short(m.get("content"), att), "tool_calls": tcs or None})
    s["messages"] = msgs
    s["input"] = short(s.get("input"), att)
    out = s.get("output") or {}
    s["output"] = {"completion": short(out.get("completion"), att), "error": out.get("error"), "usage": out.get("usage")}
    s["scores"] = short(s.get("scores"), att)
print(json.dumps(d))
"""


# --------------------------------------------------------------------------- #
# base adapter
# --------------------------------------------------------------------------- #
class InspectAdapter(BenchAdapter):
    task: str = ""  # e.g. "inspect_evals/gpqa_diamond"
    metric: str = "accuracy"
    env_required: tuple[str, ...] = ()
    task_version_note: str = ""
    default_n_tasks: int = 0
    tokens_in_per_task: int = 2000
    tokens_out_per_task: int = 500

    @property
    def tracker_key(self) -> str:
        return self.key

    # -- harness env -------------------------------------------------------------
    def harness_bin(self) -> Path:
        try:
            return ensure_venv(HARNESS_NAME, PACKAGES, python=HARNESS_PYTHON)
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    def harness_versions(self, bin_dir: Path) -> dict[str, Any]:
        v = frozen_versions(bin_dir, list(PINNED))
        info: dict[str, Any] = {
            "name": "inspect_evals",
            "task": self.task,
            "inspect_evals": v.get("inspect-evals"),
            "inspect_ai": v.get("inspect-ai"),
            "pinned": dict(PINNED),
            "venv": str(bin_dir.parent),
        }
        drift = {p: (v.get(p), want) for p, want in PINNED.items() if v.get(p) != want}
        if drift:
            info["version_drift"] = {p: f"installed {a}, pinned {b}" for p, (a, b) in drift.items()}
        return info

    # -- interface ---------------------------------------------------------------
    def prepare(self) -> dict[str, Any]:
        # cheap checks first so the message names the real blocker, then the venv
        for var in self.env_required:
            if not os.environ.get(var) and not building_image():
                raise HarnessNotReady(f"{self.key}: env var {var} is required (put it in .env)")
        if self.needs_docker and not docker_available() and not building_image():
            raise HarnessNotReady(
                f"{self.key}: Docker daemon not reachable (needed for the sandbox)"
            )
        bin_dir = self.harness_bin()
        if not (bin_dir / "inspect").exists():
            raise HarnessNotReady(f"{self.key}: `inspect` CLI missing in {bin_dir}")
        return self.harness_versions(bin_dir)

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        n, ids, notes = self.resolve_subset(subset, smoke)
        return RunPlan(
            n_tasks=n,
            trials=trials,
            subset_ids=ids,
            est_tokens_in_per_task=self.tokens_in_per_task,
            est_tokens_out_per_task=self.tokens_out_per_task,
            notes=notes,
        )

    def run(
        self,
        model: ModelSpec,
        run_dir: RunDir,
        subset: str | None,
        trials: int,
        smoke: bool,
        budget: BudgetGuard | None = None,
    ) -> Path:
        raw = run_dir.raw_dir
        bin_dir = self.harness_bin()
        with ProxyProcess(self.cfg, raw, run_dir.path / RunDir.USAGE) as proxy:
            cmd = self.build_command(
                model, raw / "logs", subset, trials, smoke, proxy.base_url, bin_dir
            )
            env = dict(os.environ)
            env.pop("VIRTUAL_ENV", None)
            env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
            env.update(proxy.env_for_harness())
            env.update(self.harness_env(proxy))
            (raw / "command.txt").write_text(" ".join(cmd) + "\n")
            self._exec(cmd, env, raw, budget)
        try:  # informational: eval-set itself resumes from the log dir
            dump = self.read_log(run_dir, bin_dir)
            done = sorted({str(s["id"]) for s in dump.get("samples") or []})
            st = run_dir.state()
            st["completed"] = sorted(set(st.get("completed", [])) | set(done))
            run_dir.save_state(st)
        except Exception:  # noqa: BLE001
            pass
        return raw

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        dump = self.read_log(run_dir, self.harness_bin())
        return self.normalize(_ns(dump), cfg, run_dir)

    # -- building the command ----------------------------------------------------
    def task_args(self, subset: str | None, smoke: bool) -> dict[str, Any]:
        return {}

    def harness_env(self, proxy: ProxyProcess) -> dict[str, str]:
        return {}

    def resolve_subset(
        self, subset: str | None, smoke: bool
    ) -> tuple[int, list[str] | None, list[str]]:
        """(n_tasks, explicit ids or None, notes)."""
        n = self.default_n_tasks
        notes: list[str] = []
        if subset and subset.isdigit():
            n = min(int(subset), n or int(subset))
            notes.append(f"first {n} samples (--limit)")
        if smoke:
            n = min(SMOKE_LIMIT, n or SMOKE_LIMIT)
            notes.append("smoke")
        return n, None, notes

    def generate_flags(self, model: ModelSpec) -> list[str]:
        params = request_params(model, self.cfg.settings)
        flags: list[str] = []
        if "temperature" in params:
            flags += ["--temperature", str(params["temperature"])]
        if "max_tokens" in params:
            flags += ["--max-tokens", str(params["max_tokens"])]
        return flags

    def build_command(
        self,
        model: ModelSpec,
        log_dir: Path,
        subset: str | None,
        trials: int,
        smoke: bool,
        base_url: str,
        bin_dir: Path | None = None,
    ) -> list[str]:
        s = self.cfg.settings
        exe = str(bin_dir / "inspect") if bin_dir else "inspect"
        cmd = [
            exe,
            "eval-set",
            self.task,
            "--model",
            f"openai/{model.id}",
            "--model-base-url",
            base_url,
            "-M",
            "responses_api=false",
            "--log-dir",
            str(log_dir),
            "--log-format",
            "eval",
            "--display",
            "plain",
            "--epochs",
            str(trials),
            "--max-connections",
            str(model.max_concurrency),
            "--timeout",
            str(int(s.request_timeout_s)),
            "--max-retries",
            "3",
            "--no-fail-on-error",  # errors are counted, not fatal
            "--retry-on-error",
            "1",
            "--log-dir-allow-dirty",
        ]
        cmd += self.generate_flags(model)
        if smoke:  # a few samples spread across the dataset, not the first three ids
            cmd += ["--limit", str(SMOKE_LIMIT), "--sample-shuffle", "42"]
        elif subset and subset.isdigit() and self.numeric_subset_is_limit:
            cmd += ["--limit", subset]
        for k, v in self.task_args(subset, smoke).items():
            if isinstance(v, list):
                cmd += ["-T", f"{k}={_yaml_list([str(x) for x in v])}"]
            elif isinstance(v, bool):
                cmd += ["-T", f"{k}={'true' if v else 'false'}"]
            else:
                cmd += ["-T", f"{k}={v}"]
        return cmd

    numeric_subset_is_limit = True  # a numeric --subset means `--limit N`

    # -- execution ---------------------------------------------------------------
    def _exec(
        self, cmd: list[str], env: dict[str, str], raw: Path, budget: BudgetGuard | None
    ) -> None:
        out = open(raw / "inspect.log", "ab")
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd())
        )
        tripped: list[BudgetExceeded] = []

        def on_trip(e: BudgetExceeded) -> None:
            tripped.append(e)
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)  # lets Inspect flush its logs

        if budget is not None:
            budget.watch(interval=5.0, on_trip=on_trip)
        try:
            while proc.poll() is None:
                time.sleep(1.0)
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=60)
            raise
        finally:
            if budget is not None:
                budget.stop()
            out.close()
        if tripped:
            raise tripped[0]
        if proc.returncode != 0:
            tail = (raw / "inspect.log").read_text(errors="replace")[-3000:]
            raise RuntimeError(
                f"inspect exited with {proc.returncode}; tail of inspect.log:\n{redact(tail)}"
            )

    # -- reading logs ------------------------------------------------------------
    @staticmethod
    def latest_log(log_dir: Path) -> Path | None:
        if not log_dir.exists():
            return None
        # eval-set also writes logs.json / eval-set.json manifests next to the task logs
        logs = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        if not logs:
            logs = sorted(
                (p for p in log_dir.glob("*.json") if p.name not in ("logs.json", "eval-set.json")),
                key=lambda p: p.stat().st_mtime,
            )
        return logs[-1] if logs else None

    def read_log(self, run_dir: RunDir, bin_dir: Path) -> dict[str, Any]:
        log_path = self.latest_log(run_dir.raw_dir / "logs")
        if log_path is None:
            raise FileNotFoundError(f"no Inspect log under {run_dir.raw_dir / 'logs'}")
        cached = run_dir.raw_dir / "log_dump.json"
        if cached.exists() and cached.stat().st_mtime >= log_path.stat().st_mtime:
            return json.loads(cached.read_text())
        out = subprocess.run(
            [str(bin_dir / "python"), "-I", "-c", _DUMP_LOG, str(log_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if out.returncode != 0:
            raise RuntimeError(f"reading {log_path} failed: {redact(out.stderr[-2000:])}")
        cached.write_text(out.stdout)
        return json.loads(out.stdout)

    # -- normalization -----------------------------------------------------------
    def normalize(self, log: Any, cfg: dict[str, Any], run_dir: RunDir) -> NormalizedResult:
        results = getattr(log, "results", None)
        scores = (results.scores if results else None) or []
        metrics: dict[str, float | None] = {}
        for sc in scores:
            for name, m in vars(sc.metrics).items():
                v = m.value
                if v is None or (isinstance(v, float) and v != v):
                    continue
                if getattr(m, "group", None) == "grouped" and getattr(m, "name", name) == "all":
                    continue  # the grouped() metric's all-samples aggregate duplicates accuracy/stderr
                key = name if len(scores) == 1 else f"{sc.name}.{name}"
                metrics[key] = round(float(v) * 100.0, 4) if self._is_fraction(name) else float(v)

        errors = ErrorCounts()
        per_task: list[dict[str, Any]] = []
        n_error_samples = 0
        sample_ids: set[str] = set()
        for s in getattr(log, "samples", None) or []:
            sample_ids.add(str(s.id))
            row: dict[str, Any] = {"id": s.id, "epoch": s.epoch}
            if getattr(s, "scores", None):
                row["scores"] = {k: self._plain(v.value) for k, v in vars(s.scores).items()}
            if getattr(s, "error", None) is not None:
                n_error_samples += 1
                cls = classify_error_message(getattr(s.error, "message", str(s.error)))
                row["error"] = cls
                self._bump(errors, cls)
            n_bad_tc = self._count_tool_parse_errors(s)
            if n_bad_tc:
                row["malformed_tool_calls"] = n_bad_tc
                errors.malformed_tool_calls += n_bad_tc
            per_task.append(row)

        tin = tout = 0
        stats = getattr(log, "stats", None)
        for mu in vars(stats.model_usage).values() if stats and stats.model_usage else []:
            tin += int(mu.input_tokens or 0)
            tout += int(mu.output_tokens or 0)
        usage = run_dir.usage_log.summarize()
        if usage.n_requests:
            cost = usage.cost_usd if usage.cost_known else None
        else:  # no proxy log (e.g. parsing a copied log): price from the spec
            try:
                cost = self.cfg.get(cfg["model_id"]).cost_usd(tin, tout)
            except Exception:  # noqa: BLE001
                cost = None

        total = int(results.total_samples) if results else len(sample_ids)
        completed = int(results.completed_samples) if results else len(sample_ids)
        epochs = int(getattr(log.eval.config, "epochs", None) or cfg.get("trials") or 1)
        n_tasks = len(sample_ids) if sample_ids else int(total / max(epochs, 1))

        warnings: list[str] = []
        acc = metrics.get(self.metric)
        if acc is not None and n_error_samples and total:
            # harness accuracy is over scored samples; also report errors-as-wrong
            metrics[f"{self.metric}_errors_as_wrong"] = (
                round(acc * completed / total, 4) if completed else 0.0
            )
            rate = n_error_samples / total
            if rate > 0.02:
                warnings.append(
                    f"{n_error_samples}/{total} samples errored ({rate:.1%}): likely a serving/config "
                    "problem, not model quality"
                )
        if getattr(log, "status", "success") != "success":
            warnings.append(f"inspect log status is {log.status!r}")

        wall = 0.0
        try:
            a = datetime.fromisoformat(stats.started_at)
            b = datetime.fromisoformat(stats.completed_at)
            wall = (b - a).total_seconds()
        except Exception:  # noqa: BLE001
            pass

        ev = log.eval
        packages = vars(ev.packages) if getattr(ev, "packages", None) else {}
        harness = {
            "name": "inspect_evals",
            "task": ev.task,
            "task_version": getattr(ev, "task_version", None),
            "task_args": self._plain(getattr(ev, "task_args", None)),
            "inspect_evals": packages.get("inspect_evals"),
            "inspect_ai": packages.get("inspect_ai"),
            "revision": self._plain(getattr(ev, "revision", None)),
            "model": ev.model,
            "log": str(self.latest_log(run_dir.raw_dir / "logs")),
        }
        note = self.note(cfg, n_tasks, epochs, harness, metrics)
        try:
            tracker_id = self.cfg.get(cfg["model_id"]).tracker
        except Exception:  # noqa: BLE001
            tracker_id = cfg["model_id"]
        tracker = {}
        if acc is not None:
            tracker[f"{tracker_id}__{self.tracker_key}"] = {"v": f"{acc:.1f}", "note": note}
        return NormalizedResult(
            bench=self.key,
            model_id=cfg["model_id"],
            metrics=metrics,
            primary_metric=self.metric,
            n_tasks=n_tasks,
            trials=epochs,
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

    @staticmethod
    def _bump(errors: ErrorCounts, cls: str) -> None:
        field = {
            "timeout": "timeouts",
            "rate_limit": "rate_limits",
            "malformed_tool_call": "malformed_tool_calls",
            "api_error": "api_errors",
        }.get(cls, "other")
        setattr(errors, field, getattr(errors, field) + 1)

    @staticmethod
    def _is_fraction(metric_name: str) -> bool:
        n = metric_name.lower()
        return any(k in n for k in ("accuracy", "acc", "stderr", "sterr", "mean", "error"))

    @staticmethod
    def _plain(v: Any) -> Any:
        if isinstance(v, SimpleNamespace):
            return {k: InspectAdapter._plain(x) for k, x in vars(v).items()}
        if isinstance(v, list):
            return [InspectAdapter._plain(x) for x in v]
        return v

    @staticmethod
    def _count_tool_parse_errors(sample: Any) -> int:
        n = 0
        for msg in getattr(sample, "messages", None) or []:
            for tc in getattr(msg, "tool_calls", None) or []:
                if getattr(tc, "parse_error", None):
                    n += 1
        return n

    def note(
        self, cfg: dict[str, Any], n: int, epochs: int, harness: dict[str, Any], metrics: dict
    ) -> str:
        params = cfg.get("request_params", {})
        t = params.get("temperature", "default")
        exc = cfg.get("per_model_exceptions") or {}
        s = (
            f"inspect_evals {harness.get('inspect_evals')} {self.task.split('/')[-1]}, {n} q, temp {t}, "
            f"{epochs} trial{'s' if epochs != 1 else ''}"
        )
        if self.task_version_note:
            s += f", {self.task_version_note}"
        if exc:
            s += f", per-model: {json.dumps(exc, sort_keys=True)}"
        if cfg.get("smoke"):
            s += ", SMOKE"
        return s


# --------------------------------------------------------------------------- #
# GPQA Diamond
# --------------------------------------------------------------------------- #
class GPQAAdapter(InspectAdapter):
    key = "gpqa"
    display = "GPQA Diamond"
    stage = 1
    task = "inspect_evals/gpqa_diamond"
    default_n_tasks = 198
    tokens_in_per_task = 700
    tokens_out_per_task = 1500
    # task defaults (inspect_evals 0.24): cot=True, shuffle_choices=42 (seeded), 4 epochs;
    # we pin --epochs to `trials` (1 by default) so the note's "1 trial" is true
    task_version_note = "cot, choices shuffled seed 42"


# --------------------------------------------------------------------------- #
# BFCL
# --------------------------------------------------------------------------- #
BFCL_COUNTS = {  # inspect_evals/bfcl/utils/CATEGORIES.md (0.24.0)
    "simple_python": 400,
    "simple_java": 100,
    "simple_javascript": 50,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
    "irrelevance": 240,
    "live_simple": 258,
    "live_multiple": 1053,
    "live_parallel": 16,
    "live_parallel_multiple": 24,
    "live_relevance": 16,
    "live_irrelevance": 884,
    "sql": 100,
    "exec_simple": 100,
    "exec_multiple": 50,
    "exec_parallel": 50,
    "exec_parallel_multiple": 40,
    "multi_turn_base": 200,
    "multi_turn_miss_func": 200,
    "multi_turn_miss_param": 200,
    "multi_turn_long_context": 200,
    "multi_turn_composite": 200,
}
BFCL_SINGLE_TURN = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
]
BFCL_MULTI_TURN = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]
BFCL_SUBSETS = {
    # leaderboard-relevant AST + hallucination categories; exec_* (retired upstream),
    # sql (not on the leaderboard) and multi_turn_composite (not in the paper) are excluded
    "single_turn": BFCL_SINGLE_TURN,
    "multi_turn": BFCL_MULTI_TURN,
    "leaderboard": BFCL_SINGLE_TURN + BFCL_MULTI_TURN,
    "smoke": ["simple_python", "multiple", "irrelevance"],
    # `profile: lite`: the cheap single-turn core without the two biggest categories
    # (simple_python 400, live_multiple 1053) and the hallucination sets; 1064 samples
    "lite": [
        "simple_java",
        "simple_javascript",
        "multiple",
        "parallel",
        "parallel_multiple",
        "live_simple",
        "live_parallel",
        "live_parallel_multiple",
        "live_relevance",
    ],
}
BFCL_DEFAULT_SUBSET = "single_turn"


class BFCLAdapter(InspectAdapter):
    key = "bfcl"
    display = "BFCL"
    stage = 1
    task = "inspect_evals/bfcl"
    tokens_in_per_task = 1500
    tokens_out_per_task = 200

    def categories(self, subset: str | None, smoke: bool) -> list[str]:
        if smoke:
            return BFCL_SUBSETS["smoke"]
        if subset and subset in BFCL_SUBSETS:
            return BFCL_SUBSETS[subset]
        if subset and not subset.isdigit():
            cats = [c.strip() for c in subset.split("+") if c.strip()]
            unknown = [c for c in cats if c not in BFCL_COUNTS]
            if unknown:
                raise ValueError(
                    f"unknown BFCL categories {unknown}; known subsets {list(BFCL_SUBSETS)}"
                )
            return cats
        return BFCL_SUBSETS[self.default_subset_name()]

    def default_subset_name(self) -> str:
        name = self.cfg.settings.bfcl_subset or BFCL_DEFAULT_SUBSET
        if name not in BFCL_SUBSETS:
            raise ValueError(f"settings.bfcl_subset {name!r} unknown; known {list(BFCL_SUBSETS)}")
        return name

    def task_args(self, subset: str | None, smoke: bool) -> dict[str, Any]:
        return {"categories": self.categories(subset, smoke)}

    def resolve_subset(self, subset, smoke):
        cats = self.categories(subset, smoke)
        n = sum(BFCL_COUNTS[c] for c in cats)
        name = (
            subset
            if (subset and subset in BFCL_SUBSETS)
            else ("smoke" if smoke else self.default_subset_name())
        )
        notes = [f"categories={name} ({len(cats)} cats, {n} samples)"]
        if smoke:
            n = SMOKE_LIMIT
            notes.append("smoke: --limit 3")
        elif subset and subset.isdigit():
            n = min(int(subset), n)
            notes.append(f"--limit {subset}")
        return n, None, notes

    def note(self, cfg, n, epochs, harness, metrics):
        base = super().note(cfg, n, epochs, harness, metrics)
        cats = (harness.get("task_args") or {}).get("categories") or []
        name = next(
            (k for k, v in BFCL_SUBSETS.items() if list(v) == list(cats)), f"{len(cats)} categories"
        )
        return base + f", categories={name}, sample-weighted overall; per-category in result.json"

    def normalize(self, log, cfg, run_dir):
        res = super().normalize(log, cfg, run_dir)
        # leaderboard-style unweighted mean of the per-category accuracies (an aggregation
        # of harness-produced numbers, not re-scoring)
        cats = {
            k[: -len("_acc")]: v
            for k, v in res.metrics.items()
            if k.endswith("_acc") and v is not None
        }
        if cats:
            res.metrics["accuracy_unweighted_categories"] = round(sum(cats.values()) / len(cats), 4)
        return res


# --------------------------------------------------------------------------- #
# GAIA (validation split, fixed seeded subset)
# --------------------------------------------------------------------------- #
GAIA_VALIDATION_N = 165
GAIA_DEFAULT_N = 50
GAIA_SEED = 20261013


class GAIAAdapter(InspectAdapter):
    key = "gaia"
    display = "GAIA"
    stage = 4
    task = "inspect_evals/gaia"
    needs_docker = True
    env_required = ("HF_TOKEN",)  # gated dataset on HuggingFace

    @property
    def default_n_tasks(self) -> int:  # type: ignore[override]
        return self.cfg.settings.gaia_subset_size

    tokens_in_per_task = 50000
    tokens_out_per_task = 3000
    task_version_note = "validation split, react agent (bash/python/web), message_limit 100"
    numeric_subset_is_limit = False  # numeric subset = fixed seeded subset, not --limit
    subsets_root: Path | None = None  # tests override

    def _all_validation_ids(self) -> list[str]:
        """Instance ids of the validation split (needs HF access; runs in the harness venv)."""
        code = (
            "import json\nfrom inspect_evals.gaia.dataset import gaia_dataset\n"
            "ds = gaia_dataset(input_prompt=None, subset='2023_all', split='validation')\n"
            "print(json.dumps([str(s.id) for s in ds]))"
        )
        return json.loads(run_python(self.harness_bin(), code))

    def subset_spec(self, subset: str | None, smoke: bool) -> dict[str, Any] | None:
        if smoke:
            return None  # --limit 3 on the full validation set
        n = int(subset) if (subset and subset.isdigit()) else self.cfg.settings.gaia_subset_size
        if n >= GAIA_VALIDATION_N:
            return None  # full validation set
        name = f"gaia_validation_{n}"
        existing = load_subset(name, self.subsets_root)
        if existing:
            return existing
        try:
            ids = self._all_validation_ids()
        except Exception as e:  # noqa: BLE001
            raise HarnessNotReady(
                f"gaia: cannot load the validation split from HuggingFace ({type(e).__name__}); "
                "check HF_TOKEN and that you accepted the gaia-benchmark/GAIA terms"
            ) from e
        return pick_subset(
            name,
            ids,
            n,
            GAIA_SEED,
            root=self.subsets_root,
            source="inspect_evals gaia 2023_all validation",
        )

    def resolve_subset(self, subset, smoke):
        if smoke:
            return SMOKE_LIMIT, None, ["smoke: --limit 3"]
        n = int(subset) if (subset and subset.isdigit()) else self.cfg.settings.gaia_subset_size
        if n >= GAIA_VALIDATION_N:
            return GAIA_VALIDATION_N, None, ["full validation split"]
        try:
            spec = self.subset_spec(subset, smoke)
            return spec["n"], spec["ids"], [f"fixed subset {spec['name']} seed {spec['seed']}"]
        except HarnessNotReady as e:  # dataset not reachable yet
            return n, None, [f"subset of {n} picked on first run (seed {GAIA_SEED}); {e}"]

    def task_args(self, subset, smoke):
        args: dict[str, Any] = {"subset": "2023_all", "split": "validation"}
        spec = self.subset_spec(subset, smoke)
        if spec:
            args["instance_ids"] = spec["ids"]
        return args

    def note(self, cfg, n, epochs, harness, metrics):
        base = super().note(cfg, n, epochs, harness, metrics)
        ids = (harness.get("task_args") or {}).get("instance_ids")
        if ids:
            base += f", fixed subset of {len(ids)} (seed {GAIA_SEED})"
        return base


# --------------------------------------------------------------------------- #
# BrowseComp (alternative to GAIA; fixed grader model, browsing on)
# --------------------------------------------------------------------------- #
class BrowseCompAdapter(InspectAdapter):
    key = "bcomp"
    display = "BrowseComp (subset)"
    stage = 4
    task = "inspect_evals/browse_comp"
    metric = "browse_comp_accuracy"
    needs_docker = True
    env_required = ("TAVILY_API_KEY",)  # web_search provider for non-OpenAI models
    default_n_tasks = 100
    tokens_in_per_task = 80000
    tokens_out_per_task = 4000
    numeric_subset_is_limit = False  # num_samples does the limiting

    @property
    def tracker_key(self) -> str:
        return "gaia"  # fills the same slot when chosen instead of GAIA

    def grader_id(self) -> str:
        return self.cfg.settings.grader_model or self.cfg.settings.baseline_model

    def task_args(self, subset, smoke):
        n = (
            SMOKE_LIMIT
            if smoke
            else (int(subset) if subset and subset.isdigit() else self.default_n_tasks)
        )
        # grader is a FIXED model (never the model under test), reached through the same proxy
        return {
            "with_browsing": True,
            "num_samples": n,
            "scorer_model": f"openai/{self.grader_id()}",
        }

    def build_command(self, model, log_dir, subset, trials, smoke, base_url, bin_dir=None):
        cmd = super().build_command(model, log_dir, subset, trials, smoke, base_url, bin_dir)
        if "--limit" in cmd:  # num_samples does the limiting
            i = cmd.index("--limit")
            del cmd[i : i + 2]
        return cmd

    def note(self, cfg, n, epochs, harness, metrics):
        return super().note(cfg, n, epochs, harness, metrics) + (
            f", first {n} samples, browsing on, graded by {self.grader_id()}"
        )
