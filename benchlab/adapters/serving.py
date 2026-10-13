"""Serving benchmark for self-hosted models: TTFT, tokens/s per request, total
throughput at concurrency 1 / 8 / 32, GPU memory and quantization.

Default tool: `guidellm <https://github.com/vllm-project/guidellm>`_ **0.8.1**
(the vLLM project's client-side load generator, pinned in its own venv). Checked
against the installed 0.8.1: ``guidellm run --scenario <file.json>`` takes the
backend / profile / data / constraints spec, so the API key goes into a 0600
temp file that is deleted after the run and never onto the command line; the
JSON output has per-benchmark ``metrics.<name>.successful`` distributions with
``median`` / ``mean`` / ``percentiles.p95``. Alternative: ``vllm bench serve``
(``settings.serving_tool: vllm``) when the vllm package is installed on the
client box (command built from vllm's benchmark CLI, not exercised here).

Metric mapping (one guidellm run per concurrency level, same synthetic
prompt/output lengths for every model):

* ``srv_ttft``  - TTFT median at concurrency 1 (seconds); p95 in result.json
* ``srv_tps``   - tokens/s per request at concurrency 1 = 1000 / median inter-token latency (ms)
* ``srv_c1/8/32`` - total output tokens/s at that concurrency (``output_tokens_per_second.successful.mean``)
* ``srv_mem``, ``srv_gpus``, ``srv_quant`` - from ``nvidia-smi`` (``serving.nvidia_smi: local`` or
  ``ssh:<host>`` on the model) and/or the model's ``serving:`` block in models.yaml
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchlab.adapters._proc import run_logged
from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import ModelSpec
from benchlab.harness_env import HarnessInstallError, ensure_venv, frozen_versions
from benchlab.redact import redact
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.usage import BudgetGuard

HARNESS_NAME = "serving"
GUIDELLM_VERSION = "0.8.1"
PACKAGES = [f"guidellm[recommended]=={GUIDELLM_VERSION}"]
TRACKER_CONCURRENCIES = (1, 8, 32)


def host_root(base_url: str) -> str:
    """guidellm/vllm want the server root; models.yaml carries .../v1."""
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def nvidia_smi(where: str) -> dict[str, Any] | None:
    """GPU name/memory via nvidia-smi, locally or over ssh. None when unavailable."""
    q = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"]
    if where == "local":
        if not shutil.which("nvidia-smi"):
            return None
        cmd = q
    elif where.startswith("ssh:"):
        cmd = ["ssh", "-o", "BatchMode=yes", where[4:], *q]
    else:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    return parse_nvidia_smi(out.stdout)


def parse_nvidia_smi(text: str) -> dict[str, Any] | None:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append(
                    {
                        "name": parts[0],
                        "mem_used_mb": float(parts[1]),
                        "mem_total_mb": float(parts[2]),
                    }
                )
            except ValueError:
                continue
    if not gpus:
        return None
    return {
        "gpus": gpus,
        "n_gpus": len(gpus),
        "mem_used_gb": round(sum(g["mem_used_mb"] for g in gpus) / 1024, 1),
        "mem_total_gb": round(sum(g["mem_total_mb"] for g in gpus) / 1024, 1),
    }


class ServingAdapter(BenchAdapter):
    key = "serving"
    display = "serving benchmark (self-hosted)"
    stage = 4
    needs_proxy = False  # measures the endpoint itself; the proxy would add latency
    needs_docker = False
    tokens_in_per_task = 0
    tokens_out_per_task = 0

    # -- harness -------------------------------------------------------------------
    def harness_bin(self) -> Path:
        try:
            return ensure_venv(HARNESS_NAME, PACKAGES, python="3.12")
        except HarnessInstallError as e:
            raise HarnessNotReady(f"{self.key}: {e}") from e

    @property
    def tool(self) -> str:
        return self.cfg.settings.serving_tool

    def prepare(self) -> dict[str, Any]:
        s = self.cfg.settings
        info: dict[str, Any] = {
            "name": self.tool,
            "concurrencies": list(s.serving_concurrencies),
            "requests_per_level": s.serving_requests_per_level,
            "prompt_tokens": s.serving_prompt_tokens,
            "output_tokens": s.serving_output_tokens,
            "data": s.serving_prompts_file or "synthetic_text",
        }
        if self.tool == "vllm":
            if not shutil.which("vllm"):
                raise HarnessNotReady(
                    f"{self.key}: serving_tool=vllm but `vllm` is not on PATH (pip install vllm on this box, or use guidellm)"
                )
            info["vllm"] = subprocess.run(
                ["vllm", "--version"], capture_output=True, text=True
            ).stdout.strip()
        elif self.tool == "guidellm":
            bin_dir = self.harness_bin()
            info["guidellm"] = frozen_versions(bin_dir, ["guidellm"]).get("guidellm")
            info["pinned"] = GUIDELLM_VERSION
        else:
            raise HarnessNotReady(f"{self.key}: unknown serving_tool {self.tool!r} (guidellm|vllm)")
        return info

    def check_model(self, model: ModelSpec) -> None:
        if model.kind != "chat" or model.provider != "openai_compatible":
            raise HarnessNotReady(
                f"{self.key}: {model.id} is not an OpenAI-compatible chat endpoint"
            )
        if not model.self_hosted:
            raise HarnessNotReady(
                f"{self.key}: {model.id} is not self_hosted; the serving benchmark is for your own servers "
                "(the TTFT probe `ttft` covers API models)"
            )

    def levels(self, smoke: bool) -> list[int]:
        return [1] if smoke else list(self.cfg.settings.serving_concurrencies)

    def n_requests(self, smoke: bool) -> int:
        return 4 if smoke else self.cfg.settings.serving_requests_per_level

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        s = self.cfg.settings
        levels, n = self.levels(smoke), self.n_requests(smoke)
        notes = [
            f"{self.tool}: concurrency {levels}, {n} requests/level, "
            + (
                f"prompts from {s.serving_prompts_file}"
                if s.serving_prompts_file
                else f"synthetic {s.serving_prompt_tokens}->{s.serving_output_tokens} tokens"
            )
            + "; no API cost (self-hosted)"
        ]
        return RunPlan(len(levels), 1, [f"c{c}" for c in levels], 0, 0, notes)

    # -- commands ---------------------------------------------------------------
    def tokenizer_for(self, model: ModelSpec) -> str:
        return str(model.serving.get("tokenizer") or model.model)

    def guidellm_scenario(
        self, model: ModelSpec, streams: int, n_requests: int, api_key: str | None
    ) -> dict[str, Any]:
        """guidellm 0.8.1 scenario file: CLI options override it, so everything goes here."""
        s = self.cfg.settings
        backend: dict[str, Any] = {
            "kind": "openai_http",
            "target": host_root(model.base_url or ""),
            "model": model.model,
            "request_format": "/v1/chat/completions",
            "validate_backend": False,
        }
        if api_key:
            backend["api_key"] = api_key
        if model.extra_headers:
            backend["headers"] = dict(model.extra_headers)
        if s.serving_prompts_file:
            data: dict[str, Any] = {
                "kind": "json_file",
                "path": str(Path(s.serving_prompts_file).resolve()),
            }
        else:
            data = {
                "kind": "synthetic_text",
                "prompt_tokens": s.serving_prompt_tokens,
                "output_tokens": s.serving_output_tokens,
            }
        return {
            "metadata": {
                "name": f"benchlab-{model.id}-c{streams}",
                "description": "BenchLab serving benchmark",
            },
            "spec": {
                "backend": backend,
                "profile": {"kind": "concurrent", "streams": streams},
                "tokenizer": {"kind": "huggingface_auto", "model": self.tokenizer_for(model)},
                "data": [data],
                "constraints": [{"kind": "max_requests", "count": n_requests}],
            },
        }

    def guidellm_command(
        self, scenario_path: Path, out_json: Path, bin_dir: Path | None = None
    ) -> list[str]:
        exe = str(bin_dir / "guidellm") if bin_dir else "guidellm"
        return [
            exe,
            "run",
            "--scenario",
            str(scenario_path),
            "--output",
            f"kind=json,path={out_json}",
            "--disable-console-interactive",
        ]

    def vllm_command(
        self, model: ModelSpec, concurrency: int, n_requests: int, result_dir: Path
    ) -> list[str]:
        s = self.cfg.settings
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai-chat",
            "--base-url",
            host_root(model.base_url or ""),
            "--endpoint",
            "/v1/chat/completions",
            "--model",
            model.model,
            "--tokenizer",
            self.tokenizer_for(model),
            "--num-prompts",
            str(n_requests),
            "--max-concurrency",
            str(concurrency),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95",
            "--save-result",
            "--result-dir",
            str(result_dir),
            "--result-filename",
            f"c{concurrency}.json",
            "--disable-tqdm",
        ]
        if s.serving_prompts_file:
            cmd += [
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(Path(s.serving_prompts_file).resolve()),
                "--custom-output-len",
                str(s.serving_output_tokens),
            ]
        else:
            cmd += [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(s.serving_prompt_tokens),
                "--random-output-len",
                str(s.serving_output_tokens),
            ]
        return cmd

    # -- run -------------------------------------------------------------------------
    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        self.check_model(model)
        raw = run_dir.raw_dir
        levels, n = self.levels(smoke), self.n_requests(smoke)
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        api_key = model.api_key(required=False)
        if self.tool == "vllm" and api_key:
            env["OPENAI_API_KEY"] = api_key  # vllm bench serve reads the bearer token from the env
        bin_dir = self.harness_bin() if self.tool == "guidellm" else None
        done = run_dir.completed_ids()
        for c in levels:
            tid = f"c{c}"
            out_json = raw / f"{tid}.json"
            if tid in done and out_json.exists():
                continue
            if self.tool == "vllm":
                cmd = self.vllm_command(model, c, n, raw)
                scenario_tmp = None
            else:
                scenario = self.guidellm_scenario(model, c, n, api_key)
                public = json.loads(json.dumps(scenario))
                if "api_key" in public["spec"]["backend"]:
                    public["spec"]["backend"]["api_key"] = f"${{{model.api_key_env}}}"
                (raw / f"scenario_{tid}.json").write_text(json.dumps(public, indent=2))
                # the real key lives only in a 0600 temp file for the duration of the run
                fd, scenario_tmp = tempfile.mkstemp(prefix="benchlab-guidellm-", suffix=".json")
                with os.fdopen(fd, "w") as f:
                    json.dump(scenario, f)
                cmd = self.guidellm_command(Path(scenario_tmp), out_json, bin_dir)
            with open(raw / "command.txt", "a") as f:
                f.write(redact(" ".join(cmd)) + "\n")
            try:
                run_logged(cmd, env, raw / f"serving_{tid}.log", budget=budget, cwd=raw)
            finally:
                if scenario_tmp:
                    Path(scenario_tmp).unlink(missing_ok=True)
            run_dir.mark_done(tid, ok=out_json.exists())
        where = str(model.serving.get("nvidia_smi") or "")
        gpu = nvidia_smi(where) if where else None
        (raw / "gpu.json").write_text(
            json.dumps(
                {
                    "nvidia_smi": where or None,
                    "result": gpu,
                    "declared": {k: v for k, v in model.serving.items() if k != "nvidia_smi"},
                },
                indent=2,
            )
        )
        return raw

    # -- parse -----------------------------------------------------------------------
    @staticmethod
    def _dist(metrics: dict[str, Any], name: str) -> dict[str, Any]:
        return ((metrics.get(name) or {}).get("successful")) or {}

    def parse_guidellm(self, path: Path) -> dict[str, Any] | None:
        try:
            d = json.loads(path.read_text())
            b = (d.get("benchmarks") or [])[0]
        except Exception:  # noqa: BLE001
            return None
        m = b.get("metrics") or {}
        ttft = self._dist(m, "time_to_first_token_ms")
        itl = self._dist(m, "inter_token_latency_ms")
        tps_agg = self._dist(m, "output_tokens_per_second")
        totals = m.get("request_totals") or {}
        lat = self._dist(m, "request_latency")
        return {
            "ttft_p50_s": ttft["median"] / 1000 if ttft.get("median") is not None else None,
            "ttft_p95_s": (ttft.get("percentiles") or {}).get("p95", 0) / 1000 if ttft else None,
            "itl_p50_ms": itl.get("median"),
            "tokens_per_s_per_request": (1000.0 / itl["median"]) if itl.get("median") else None,
            "output_tokens_per_s": tps_agg.get("mean"),
            "requests_per_s": self._dist(m, "requests_per_second").get("mean"),
            "latency_p50_s": lat.get("median"),
            "n_successful": totals.get("successful") if isinstance(totals, dict) else None,
            "n_errored": totals.get("errored") if isinstance(totals, dict) else None,
            "n_requests": totals.get("total") if isinstance(totals, dict) else None,
            "duration_s": b.get("duration"),
            "streams": ((b.get("config") or {}).get("strategy") or {}).get("streams"),
        }

    def parse_vllm(self, path: Path) -> dict[str, Any] | None:
        try:
            d = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return None
        itl = d.get("median_itl_ms")
        return {
            "ttft_p50_s": d["median_ttft_ms"] / 1000
            if d.get("median_ttft_ms") is not None
            else None,
            "ttft_p95_s": d["p95_ttft_ms"] / 1000 if d.get("p95_ttft_ms") is not None else None,
            "itl_p50_ms": itl,
            "tokens_per_s_per_request": (1000.0 / itl) if itl else None,
            "output_tokens_per_s": d.get("output_throughput"),
            "requests_per_s": d.get("request_throughput"),
            "latency_p50_s": d["median_e2el_ms"] / 1000
            if d.get("median_e2el_ms") is not None
            else None,
            "n_successful": d.get("completed"),
            "n_errored": d.get("failed"),
            "n_requests": d.get("num_prompts"),
            "duration_s": d.get("duration"),
            "streams": d.get("max_concurrency"),
        }

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        raw = run_dir.raw_dir
        files = sorted(
            raw.glob("c*.json"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0
        )
        files = [p for p in files if p.stem[1:].isdigit()]
        if not files:
            raise FileNotFoundError(f"no serving results under {raw}")
        tool = ((cfg.get("harness") or {}).get("name")) or self.tool
        parser = self.parse_vllm if tool == "vllm" else self.parse_guidellm
        levels: dict[int, dict[str, Any]] = {}
        for p in files:
            r = parser(p)
            if r:
                levels[int(p.stem[1:])] = r
        gpu = json.loads((raw / "gpu.json").read_text()) if (raw / "gpu.json").exists() else {}
        return self.normalize(levels, gpu, cfg, run_dir)

    def normalize(
        self,
        levels: dict[int, dict[str, Any]],
        gpu: dict[str, Any],
        cfg: dict[str, Any],
        run_dir: RunDir,
    ) -> NormalizedResult:
        metrics: dict[str, float | None] = {}
        errors = ErrorCounts()
        per_task = []
        for c, r in sorted(levels.items()):
            metrics[f"throughput_tok_s_c{c}"] = r.get("output_tokens_per_s")
            metrics[f"ttft_p50_s_c{c}"] = r.get("ttft_p50_s")
            metrics[f"ttft_p95_s_c{c}"] = r.get("ttft_p95_s")
            metrics[f"tokens_per_s_per_request_c{c}"] = r.get("tokens_per_s_per_request")
            errors.api_errors += int(r.get("n_errored") or 0)
            per_task.append({"id": f"c{c}", **r})
        c_min = min(levels) if levels else None
        c1 = levels.get(c_min, {}) if c_min is not None else {}
        metrics["ttft_p50_s"] = c1.get("ttft_p50_s")
        metrics["ttft_p95_s"] = c1.get("ttft_p95_s")
        metrics["tokens_per_s_per_request"] = c1.get("tokens_per_s_per_request")
        declared = gpu.get("declared") or {}
        smi = gpu.get("result") or {}
        n_gpus = smi.get("n_gpus") or declared.get("gpus")
        mem_gb = smi.get("mem_used_gb") or declared.get("gpu_mem_gb")
        quant = declared.get("quantization")
        metrics["gpu_mem_used_gb"] = float(mem_gb) if mem_gb is not None else None
        metrics["n_gpus"] = float(n_gpus) if n_gpus is not None else None
        warnings = []
        if not levels:
            warnings.append("no concurrency level produced a result file")
        if quant is None:
            warnings.append(
                "quantization not declared: set serving.quantization on the model in models.yaml"
            )
        if mem_gb is None:
            warnings.append(
                "GPU memory unknown: set serving.nvidia_smi (local|ssh:host) or serving.gpu_mem_gb"
            )
        if any((r.get("n_errored") or 0) for r in levels.values()):
            warnings.append(
                "some serving requests errored; check the server logs before trusting throughput"
            )
        missing = [c for c in TRACKER_CONCURRENCIES if c not in levels]
        if levels and missing:
            warnings.append(
                f"no result for concurrency {missing} (smoke run or custom serving_concurrencies)"
            )
        h = cfg.get("harness") or {}
        tool = h.get("name") or self.tool
        ver = h.get("guidellm") or h.get("vllm") or ""
        data = h.get("data") or "synthetic_text"
        shape = (
            f"synthetic {h.get('prompt_tokens')}->{h.get('output_tokens')} tokens"
            if data == "synthetic_text"
            else f"prompts {data}, {h.get('output_tokens')} output tokens"
        )
        note = (
            f"{tool} {ver}: concurrency {sorted(levels)}, {h.get('requests_per_level')} req/level, {shape}, "
            f"chat completions, endpoint measured directly (no proxy); ttft = median at c={c_min if c_min is not None else '?'}; "
            f"tokens/s per request = 1000/median ITL; c1/c8/c32 = total output tok/s"
            + (f"; quant {quant}" if quant else "")
            + (f"; {n_gpus} GPU(s)" if n_gpus else "")
            + (f"; {smi['gpus'][0]['name']}" if smi.get("gpus") else "")
        )
        if cfg.get("smoke"):
            note += "; SMOKE"
        model_id = cfg["model_id"]
        try:
            tracker_id = self.cfg.get(model_id).tracker
        except Exception:  # noqa: BLE001
            tracker_id = model_id
        tracker: dict[str, dict[str, str]] = {}

        def put(k: str, v: Any, fmt: str = "{:.1f}") -> None:
            if v is not None:
                tracker[f"{tracker_id}__{k}"] = {
                    "v": v if isinstance(v, str) else fmt.format(v),
                    "note": note,
                }

        put("srv_ttft", metrics["ttft_p50_s"], "{:.2f}")
        put("srv_tps", metrics["tokens_per_s_per_request"])
        for c in TRACKER_CONCURRENCIES:
            put(f"srv_c{c}", metrics.get(f"throughput_tok_s_c{c}"))
        put("srv_mem", metrics["gpu_mem_used_gb"])
        put("srv_gpus", metrics["n_gpus"], "{:.0f}")
        if quant:
            put("srv_quant", str(quant))
        harness = {**h, "gpu": smi or None, "declared": declared}
        return NormalizedResult(
            bench=self.key,
            model_id=model_id,
            metrics=metrics,
            primary_metric="ttft_p50_s",
            n_tasks=len(levels),
            trials=1,
            errors=errors,
            tokens_in=0,
            tokens_out=0,
            cost_usd=None,
            wall_time_s=sum(float(r.get("duration_s") or 0) for r in levels.values()),
            harness=harness,
            note=note,
            per_task=per_task,
            tracker=tracker,
            warnings=warnings,
        )
