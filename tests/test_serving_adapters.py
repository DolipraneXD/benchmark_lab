"""Stage 7: TTFT probe (every model) and the serving benchmark (self-hosted only)."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlab.adapters.base import HarnessNotReady
from benchlab.adapters.serving import ServingAdapter, host_root, parse_nvidia_smi
from benchlab.adapters.ttft import PROMPTS, TTFTAdapter
from benchlab.cli import app
from benchlab.config import load_config
from benchlab.registry import BENCHES, applies_to, select_benches
from benchlab.runs import RunConfig, RunDir

runner = CliRunner()
FIXTURE = Path("tests/fixtures/guidellm_concurrent2_mock.json")
VENV = Path("workspace/harnesses/serving/.venv/bin/guidellm")


def test_ttft_end_to_end_via_cli(models_yaml, tmp_path):
    r = runner.invoke(
        app, ["run", "-m", "mock", "--bench", "ttft", "--smoke", "-c", str(models_yaml)]
    )
    assert r.exit_code == 0, r.output
    run_dir = next((tmp_path / "runs").glob("*_mock_ttft"))
    res = json.loads((run_dir / "result.json").read_text())
    assert res["n_tasks"] == 3 and res["trials"] == 1 and res["errors"]["api_errors"] == 0
    assert 0 < res["metrics"]["ttft_p50_s"] < 5 and res["metrics"]["n_ok"] == 3
    assert res["tracker"]["mock__ttft"]["v"] == f"{res['metrics']['ttft_p50_s']:.2f}"
    assert "SMOKE" in res["tracker"]["mock__ttft"]["note"]
    assert res["cost_usd"] is not None and res["tokens_out"] > 0  # streamed usage still priced
    samples = (run_dir / "raw" / "samples.jsonl").read_text()
    assert samples.count("\n") == 3 and "sk-mock" not in samples
    # resume: nothing left to do, result identical
    r2 = runner.invoke(
        app, ["run", "-m", "mock", "--bench", "ttft", "--smoke", "-c", str(models_yaml)]
    )
    assert r2.exit_code == 0, r2.output
    assert (run_dir / "raw" / "samples.jsonl").read_text().count("\n") == 3


def test_ttft_counts_errors_and_plan(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    plan = TTFTAdapter(cfg).plan(cfg.get("mock"), None, 1, smoke=False)
    assert plan.n_tasks == len(PROMPTS) == 10 and plan.trials == 3
    r = runner.invoke(
        app, ["run", "-m", "broken", "--bench", "ttft", "--smoke", "-c", str(models_yaml)]
    )
    assert r.exit_code != 0 or "errors=" in r.output
    run_dir = next((tmp_path / "runs").glob("*_broken_ttft"), None)
    if run_dir and (run_dir / "result.json").exists():
        res = json.loads((run_dir / "result.json").read_text())
        assert res["errors"]["api_errors"] == 3 and res["tracker"] == {} and res["warnings"]


def test_serving_registry_only_self_hosted(models_yaml):
    cfg = load_config(models_yaml)
    b = BENCHES["serving"]
    assert applies_to(b, cfg.get("notools")) and not applies_to(b, cfg.get("mock"))
    assert select_benches(bench="srv_c8")[0].key == "serving"
    for k in ("srv_quant", "srv_gpus", "srv_mem", "srv_ttft", "srv_tps", "srv_c1", "srv_c32"):
        assert select_benches(bench=k)[0].key == "serving"
    with pytest.raises(HarnessNotReady):
        ServingAdapter(cfg).check_model(cfg.get("mock"))
    ServingAdapter(cfg).check_model(cfg.get("notools"))


def test_serving_scenario_and_commands(models_yaml, monkeypatch):
    cfg = load_config(models_yaml)
    m = cfg.get("notools")
    m.serving = {"quantization": "fp8", "tokenizer": "Qwen/Qwen3-8B"}
    a = ServingAdapter(cfg)
    sc = a.guidellm_scenario(m, 8, 64, "sk-secret")
    be = sc["spec"]["backend"]
    assert be["target"] == host_root(m.base_url) and not be["target"].endswith("/v1")
    assert be["api_key"] == "sk-secret" and be["request_format"] == "/v1/chat/completions"
    assert sc["spec"]["profile"] == {"kind": "concurrent", "streams": 8}
    assert sc["spec"]["constraints"] == [{"kind": "max_requests", "count": 64}]
    assert sc["spec"]["data"] == [
        {"kind": "synthetic_text", "prompt_tokens": 512, "output_tokens": 256}
    ]
    assert sc["spec"]["tokenizer"]["model"] == "Qwen/Qwen3-8B"
    cmd = a.guidellm_command(Path("/x/s.json"), Path("/x/c8.json"), Path("/venv/bin"))
    assert cmd[:3] == ["/venv/bin/guidellm", "run", "--scenario"] and "sk-secret" not in " ".join(
        cmd
    )
    v = a.vllm_command(m, 32, 64, Path("/out"))
    assert (
        v[:3] == ["vllm", "bench", "serve"]
        and "--max-concurrency" in v
        and v[v.index("--max-concurrency") + 1] == "32"
    )
    assert v[v.index("--result-filename") + 1] == "c32.json" and "--random-input-len" in v
    cfg.settings.serving_prompts_file = "prompts.jsonl"
    assert a.guidellm_scenario(m, 1, 4, None)["spec"]["data"][0]["kind"] == "json_file"
    assert "--dataset-name" in a.vllm_command(m, 1, 4, Path("/out"))
    assert (
        host_root("http://h:8000/v1/") == "http://h:8000"
        and host_root("http://h:8000") == "http://h:8000"
    )


def test_parse_nvidia_smi():
    out = "NVIDIA H100 80GB HBM3, 71234, 81559\nNVIDIA H100 80GB HBM3, 70999, 81559\n"
    g = parse_nvidia_smi(out)
    assert (
        g["n_gpus"] == 2
        and g["mem_used_gb"] == 138.9
        and g["gpus"][0]["name"].startswith("NVIDIA H100")
    )
    assert parse_nvidia_smi("") is None


def _serving_run_dir(cfg, tmp_path, smoke=False, harness=None):
    rd = RunDir.create(tmp_path / "runs", "notools", "serving")
    rd.write_config(
        RunConfig(
            model_id="notools",
            bench="serving",
            model={},
            request_params={},
            per_model_exceptions={},
            harness=harness
            or {
                "name": "guidellm",
                "guidellm": "0.8.1",
                "requests_per_level": 6,
                "prompt_tokens": 512,
                "output_tokens": 256,
                "data": "synthetic_text",
            },
            subset={},
            trials=1,
            smoke=smoke,
        )
    )
    return rd


def test_serving_parse_guidellm_fixture(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    cfg.get("notools").serving = {"quantization": "fp8", "gpus": 2, "gpu_mem_gb": 140}
    rd = _serving_run_dir(cfg, tmp_path)
    raw = rd.raw_dir
    fx = FIXTURE.read_text()
    for c in (1, 8, 32):  # same fixture at every level; the parser must key by file name
        (raw / f"c{c}.json").write_text(fx)
    (raw / "gpu.json").write_text(
        json.dumps(
            {
                "nvidia_smi": None,
                "result": None,
                "declared": {"quantization": "fp8", "gpus": 2, "gpu_mem_gb": 140},
            }
        )
    )
    res = ServingAdapter(cfg).parse(rd)
    m = res.metrics
    fixture = json.loads(fx)["benchmarks"][0]["metrics"]
    assert m["ttft_p50_s"] == pytest.approx(
        fixture["time_to_first_token_ms"]["successful"]["median"] / 1000
    )
    assert m["ttft_p95_s"] == pytest.approx(
        fixture["time_to_first_token_ms"]["successful"]["percentiles"]["p95"] / 1000
    )
    assert m["throughput_tok_s_c8"] == pytest.approx(
        fixture["output_tokens_per_second"]["successful"]["mean"]
    )
    assert m["tokens_per_s_per_request"] == pytest.approx(
        1000 / fixture["inter_token_latency_ms"]["successful"]["median"]
    )
    assert m["gpu_mem_used_gb"] == 140 and m["n_gpus"] == 2
    assert res.n_tasks == 3 and res.errors.api_errors == 0 and res.cost_usd is None
    t = res.tracker
    assert set(t) == {
        f"notools__{k}"
        for k in (
            "srv_ttft",
            "srv_tps",
            "srv_c1",
            "srv_c8",
            "srv_c32",
            "srv_mem",
            "srv_gpus",
            "srv_quant",
        )
    }
    assert (
        t["notools__srv_quant"]["v"] == "fp8"
        and t["notools__srv_gpus"]["v"] == "2"
        and t["notools__srv_mem"]["v"] == "140.0"
    )
    assert t["notools__srv_ttft"]["v"] == f"{m['ttft_p50_s']:.2f}"
    assert (
        "guidellm 0.8.1" in t["notools__srv_c1"]["note"]
        and "no proxy" in t["notools__srv_c1"]["note"]
    )
    assert not any("quantization" in w for w in res.warnings)


def test_serving_parse_vllm_and_warnings(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    rd = _serving_run_dir(
        cfg,
        tmp_path,
        harness={
            "name": "vllm",
            "vllm": "0.31.0",
            "requests_per_level": 64,
            "prompt_tokens": 512,
            "output_tokens": 256,
            "data": "synthetic_text",
        },
    )
    (rd.raw_dir / "c1.json").write_text(
        json.dumps(
            {
                "median_ttft_ms": 120.0,
                "p95_ttft_ms": 300.0,
                "median_itl_ms": 12.5,
                "median_e2el_ms": 3500.0,
                "output_throughput": 80.0,
                "request_throughput": 0.3,
                "completed": 62,
                "failed": 2,
                "num_prompts": 64,
                "duration": 210.0,
                "max_concurrency": 1,
            }
        )
    )
    res = ServingAdapter(cfg).parse(rd)
    assert res.metrics["ttft_p50_s"] == 0.12 and res.metrics["tokens_per_s_per_request"] == 80.0
    assert res.errors.api_errors == 2 and res.metrics["throughput_tok_s_c1"] == 80.0
    assert res.tracker["notools__srv_c1"]["v"] == "80.0" and "notools__srv_c8" not in res.tracker
    assert any("quantization" in w for w in res.warnings) and any(
        "GPU memory" in w for w in res.warnings
    )
    assert any("errored" in w for w in res.warnings) and any("[8, 32]" in w for w in res.warnings)
    assert "vllm 0.31.0" in res.note


@pytest.mark.skipif(not VENV.exists(), reason="serving harness venv not installed")
def test_serving_end_to_end_guidellm(models_yaml, tmp_path, mock_server, monkeypatch):
    """Real guidellm 0.8.1 run against the mock (one level, 3 requests, fixed prompts)."""
    monkeypatch.setenv("BENCHLAB_HARNESS_DIR", str(Path("workspace/harnesses").resolve()))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in PROMPTS[:3]))
    text = models_yaml.read_text().replace(
        "  request_timeout_s: 20\n",
        "  request_timeout_s: 20\n  serving_concurrencies: [2]\n"
        f"  serving_requests_per_level: 3\n  serving_prompts_file: {prompts}\n",
    )
    cfgp = tmp_path / "models_srv.yaml"
    cfgp.write_text(text)
    r = runner.invoke(app, ["run", "-m", "notools", "--bench", "serving", "-c", str(cfgp)])
    assert r.exit_code == 0, r.output
    run_dir = next((tmp_path / "runs").glob("*_notools_serving"))
    res = json.loads((run_dir / "result.json").read_text())
    assert (
        res["n_tasks"] == 1
        and res["metrics"]["throughput_tok_s_c2"] > 0
        and res["metrics"]["ttft_p50_s"] > 0
    )
    assert res["tracker"]["notools__srv_ttft"] and "notools__srv_c1" not in res["tracker"]
    assert (run_dir / "raw" / "c2.json").exists() and (
        run_dir / "raw" / "scenario_c2.json"
    ).exists()
    blob = "".join(p.read_text(errors="ignore") for p in (run_dir / "raw").iterdir() if p.is_file())
    assert os.environ["MOCK_API_KEY"] not in blob and "${MOCK_API_KEY}" in blob
