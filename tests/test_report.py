"""Stage 8: report.html + tracker.json from synthetic finished runs."""

import json
from pathlib import Path

from typer.testing import CliRunner

from benchlab.cli import app
from benchlab.config import load_config
from benchlab.report import build_report, collect
from benchlab.runs import ErrorCounts, NormalizedResult, RunConfig, RunDir

runner = CliRunner()

MODELS_EXTRA = """
  - id: step5
    name: Step 5 API
    provider: openai_compatible
    base_url: http://127.0.0.1:1/v1
    model: step5
    api_key_env: MOCK_API_KEY
    price_in_per_mtok: 1.0
    price_out_per_mtok: 2.0
  - id: step5self
    name: Step 5 self-hosted
    provider: openai_compatible
    base_url: http://127.0.0.1:1/v1
    model: step5
    api_key_env: MOCK_API_KEY
    self_hosted: true
    tracker_id: step5
    compare_to: step5
  - id: gptoss
    name: gpt-oss
    provider: openai_compatible
    base_url: http://127.0.0.1:1/v1
    model: gpt-oss
    api_key_env: MOCK_API_KEY
    self_hosted: true
    published: {gpqa: 80.0, bfcl: 60.0}
  - id: dsflash
    name: DS flash
    provider: openai_compatible
    base_url: http://127.0.0.1:1/v1
    model: ds
    api_key_env: MOCK_API_KEY
    price_in_per_mtok: 0.1
    price_out_per_mtok: 0.3
"""


def _cfg(models_yaml, tmp_path):
    p = tmp_path / "models_report.yaml"
    p.write_text(models_yaml.read_text() + MODELS_EXTRA)
    return p, load_config(p)


def _run(
    cfg,
    model_id,
    bench,
    metrics,
    primary,
    n=10,
    trials=1,
    cost=1.0,
    smoke=False,
    errors=None,
    tokens_out=1000,
    tracker=None,
    warnings=(),
    wall=100.0,
    harness=None,
):
    rd = RunDir.create(cfg.settings.runs_dir, model_id, bench)
    rd.write_config(
        RunConfig(
            model_id=model_id,
            bench=bench,
            model={},
            request_params={},
            per_model_exceptions={},
            harness=harness or {"name": bench, "version": "1.0"},
            subset={},
            trials=trials,
            smoke=smoke,
        )
    )
    tid = cfg.get(model_id).tracker
    res = NormalizedResult(
        bench=bench,
        model_id=model_id,
        metrics=metrics,
        primary_metric=primary,
        n_tasks=n,
        trials=trials,
        errors=errors or ErrorCounts(),
        tokens_in=2000,
        tokens_out=tokens_out,
        cost_usd=cost,
        wall_time_s=wall,
        harness=harness or {"name": bench, "version": "1.0"},
        note=f"{bench} note",
        tracker=tracker
        if tracker is not None
        else {f"{tid}__{bench}": {"v": f"{metrics[primary]:.1f}", "note": f"{bench} note"}},
        warnings=list(warnings),
    )
    rd.write_result(res)
    return rd


def _populate(cfg):
    # baseline claude
    _run(cfg, "claude", "gpqa", {"accuracy": 80.0}, "accuracy", n=198, cost=3.5)
    _run(cfg, "claude", "bfcl", {"accuracy": 70.0}, "accuracy", n=100, cost=2.0)
    _run(cfg, "claude", "lcb", {"pass@1": 55.0}, "pass@1", n=50, cost=1.0)
    _run(
        cfg,
        "claude",
        "ttft",
        {"ttft_p50_s": 0.5, "ttft_p95_s": 0.9},
        "ttft_p50_s",
        n=10,
        trials=3,
        cost=0.01,
        tracker={"claude__ttft": {"v": "0.50", "note": "ttft"}},
    )
    # mock: older full run, newer smoke run (full must win); 10% errors on bfcl
    _run(cfg, "mock", "gpqa", {"accuracy": 70.0}, "accuracy", n=198, cost=1.0)
    _run(cfg, "mock", "gpqa", {"accuracy": 66.7}, "accuracy", n=3, cost=0.01, smoke=True)
    _run(
        cfg,
        "mock",
        "bfcl",
        {"accuracy": 60.0},
        "accuracy",
        n=100,
        cost=0.5,
        errors=ErrorCounts(timeouts=10),
    )
    _run(
        cfg, "mock", "t2air", {"pass^1": 100.0}, "pass^1", n=20, trials=3, cost=0.8
    )  # all-or-nothing -> flagged
    _run(
        cfg,
        "mock",
        "ttft",
        {"ttft_p50_s": 0.2, "ttft_p95_s": 0.3},
        "ttft_p50_s",
        n=10,
        trials=3,
        cost=0.01,
        tracker={"mock__ttft": {"v": "0.20", "note": "ttft"}},
    )
    # step5 API vs self-hosted
    _run(cfg, "step5", "gpqa", {"accuracy": 75.0}, "accuracy", n=198, cost=2.0, tokens_out=20000)
    _run(cfg, "step5", "bfcl", {"accuracy": 65.0}, "accuracy", n=100, cost=1.0, tokens_out=5000)
    _run(
        cfg, "step5self", "gpqa", {"accuracy": 73.0}, "accuracy", n=198, cost=None, tokens_out=40000
    )
    _run(
        cfg,
        "step5self",
        "bfcl",
        {"accuracy": 50.0},
        "accuracy",
        n=100,
        cost=None,
        tokens_out=10000,
        tracker={"step5__bfcl": {"v": "50.0", "note": "self"}},
    )  # clashes with API key on purpose -> newest wins
    _run(
        cfg,
        "step5self",
        "serving",
        {"ttft_p50_s": 0.1, "throughput_tok_s_c1": 50.0, "tokens_per_s_per_request": 40.0},
        "ttft_p50_s",
        n=3,
        cost=None,
        tracker={
            "step5__srv_ttft": {"v": "0.10", "note": "srv"},
            "step5__srv_c1": {"v": "50.0", "note": "srv"},
            "step5__srv_quant": {"v": "fp8", "note": "srv"},
        },
        harness={"name": "guidellm", "guidellm": "0.8.1"},
    )
    # gptoss vs published: gpqa ok, bfcl 8 points off
    _run(cfg, "gptoss", "gpqa", {"accuracy": 79.0}, "accuracy", n=198, cost=None)
    _run(cfg, "gptoss", "bfcl", {"accuracy": 52.0}, "accuracy", n=100, cost=None)
    # dsflash: cost / latency per solved
    _run(cfg, "dsflash", "gpqa", {"accuracy": 50.0}, "accuracy", n=100, cost=0.5, wall=1000.0)
    _run(
        cfg,
        "dsflash",
        "tb",
        {"resolved": 20.0},
        "resolved",
        n=50,
        cost=2.0,
        wall=5000.0,
        warnings=["3 tasks hit the harness timeout"],
    )


def test_collect_prefers_full_over_smoke(models_yaml, tmp_path):
    _, cfg = _cfg(models_yaml, tmp_path)
    _populate(cfg)
    col = collect(cfg)
    e = col.get("mock", "gpqa")
    assert e.primary == 70.0 and not e.smoke and ("mock", "gpqa") not in col.smoke_only
    assert collect(cfg, include_smoke=False).get("mock", "gpqa").primary == 70.0
    assert len(col.all_runs) == 18 and len(col.entries) == 17


def test_tracker_and_computed_keys(models_yaml, tmp_path):
    _, cfg = _cfg(models_yaml, tmp_path)
    _populate(cfg)
    rep = build_report(cfg, tmp_path / "report.html")
    t = rep.tracker
    assert json.loads(rep.tracker_path.read_text()) == t
    assert t["claude__gpqa"] == {"v": "80.0", "note": "gpqa note"}
    assert t["mock__gpqa"]["v"] == "70.0" and "SMOKE" not in t["mock__gpqa"]["note"]
    # cost = sum over core benches with a cost
    assert t["claude__cost"]["v"] == "5.50" and "gpqa $3.50" in t["claude__cost"]["note"]
    assert (
        t["step5__cost"]["v"] == "3.00"
    )  # API cost, not overwritten by the unpriced self-hosted twin
    assert t["step5__srv_match"]["v"] == "-8.5" and "chat template" in t["step5__srv_match"]["note"]
    assert t["gptoss__srv_match"]["v"] == "-8.0" and t["gptoss__srv_match"]["note"].startswith(
        "vs published"
    )
    assert not any(k.split("__", 1)[1] in ("s5_api", "gpt_sanity", "ds_cps") for k in t)
    assert t["step5__srv_quant"]["v"] == "fp8" and t["claude__ttft"]["v"] == "0.50"
    assert list(t) == sorted(t)


def test_suspicious_and_html(models_yaml, tmp_path):
    p, cfg = _cfg(models_yaml, tmp_path)
    _populate(cfg)
    rep = build_report(cfg, tmp_path / "out" / "report.html")
    why = [(s["model"], s["bench"], s["why"]) for s in rep.suspicious]
    assert any(m == "mock" and b == "bfcl" and "error rate 10%" in w for m, b, w in why)
    assert any(m == "mock" and b == "t2air" and "score 100.0" in w for m, b, w in why)
    assert any(m == "gptoss" and b == "bfcl" and "-8.0 pts vs published" in w for m, b, w in why)
    assert any(
        m == "step5self" and b == "bfcl" and "below the API counterpart" in w for m, b, w in why
    )
    assert any(m == "dsflash" and b == "tb" and "adapter warning" in w for m, b, w in why)
    assert (
        any(m == "step5self" and b == "gpqa" and "no cost recorded" in w for m, b, w in why)
        is False
    )  # unpriced model
    assert not any(m == "mock" and b == "gpqa" and "smoke" in w for m, b, w in why)
    html = rep.html_path.read_text()
    assert "<title>BenchLab report</title>" in html and "Worth a second look" in html
    assert "<style>" in html and "/static/" not in html  # self-contained file
    assert "-10.0 vs claude" in html  # mock gpqa 70 vs claude 80
    assert 'class="delta up">-0.3 vs claude' in html  # lower TTFT is better
    assert "sk-mock" not in html
    assert (Path(cfg.settings.results_dir) / "suspicious.json").exists()
    # CLI
    r = runner.invoke(app, ["report", "-c", str(p), "--out", str(tmp_path / "cli.html")])
    assert r.exit_code == 0, r.output
    assert "tracker:" in r.output and (tmp_path / "cli.html").exists()


def test_report_without_runs(models_yaml, tmp_path):
    r = runner.invoke(
        app, ["report", "-c", str(models_yaml), "--out", str(tmp_path / "empty.html")]
    )
    assert r.exit_code == 0, r.output
    t = json.loads((tmp_path / "results" / "tracker.json").read_text())
    assert all(v["v"] == "n/a" for v in t.values())  # only the not-integrated placeholders
    assert "No finished runs yet." in (tmp_path / "empty.html").read_text()


def test_reference_model_row_and_tracker(models_yaml, tmp_path):
    from benchlab.config import ModelSpec

    _, cfg = _cfg(models_yaml, tmp_path)
    cfg.models.append(
        ModelSpec(
            id="refm",
            name="Ref Model",
            provider="anthropic",
            model="r",
            reference=True,
            published={"gpqa": 91.0},
        )
    )
    _populate(cfg)
    rep = build_report(cfg, tmp_path / "report.html")
    assert rep.tracker["refm__gpqa"]["v"] == "91.0"
    assert "published" in rep.tracker["refm__gpqa"]["note"]
    html = rep.html_path.read_text()
    assert 'class="ref">refm<span class="sub">published' in html and 'class="ref">91.0</td>' in html
    assert not any(s["model"] == "refm" for s in rep.suspicious)
