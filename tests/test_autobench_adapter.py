"""AutomationBench (opt-in extra `autobench`), verified against the mock endpoint."""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlab.adapters.automationbench import KEY_VAR, PUBLIC_DOMAINS, AutomationBenchAdapter
from benchlab.cli import app
from benchlab.config import load_config
from benchlab.runs import RunConfig, RunDir

runner = CliRunner()
AB_FIXTURE = Path("tests/fixtures/automationbench_sales2_mock.json")
AB_BIN = Path("workspace/harnesses/automationbench/repo/.venv/bin/auto-bench")


def _rd(cfg, tmp_path, model_id, bench, harness, smoke=False):
    rd = RunDir.create(tmp_path / "runs", model_id, bench)
    rd.write_config(
        RunConfig(
            model_id=model_id,
            bench=bench,
            model={},
            request_params={},
            per_model_exceptions={},
            harness=harness,
            subset={},
            trials=1,
            smoke=smoke,
        )
    )
    return rd


# ---- AutomationBench ---------------------------------------------------------------
def test_autobench_command_and_plan(models_yaml):
    cfg = load_config(models_yaml)
    a = AutomationBenchAdapter(cfg)
    m = cfg.get("mock")
    cmd = a.command(
        m,
        "http://127.0.0.1:4321/v1",
        list(PUBLIC_DOMAINS),
        Path("/r/results.json"),
        False,
        Path("/v/bin"),
    )
    assert cmd[0] == "/v/bin/auto-bench" and cmd[cmd.index("--model") + 1] == "mock"
    assert cmd[cmd.index("--api-key-var") + 1] == KEY_VAR and "--num-examples" not in cmd
    assert cmd[cmd.index("--domains") + 1] == "sales,marketing,operations,support,finance,hr"
    assert (
        cmd[cmd.index("--max-concurrent") + 1] == "4" and cmd[cmd.index("--toolset") + 1] == "api"
    )
    assert "--num-examples" in a.command(m, "u", ["sales"], Path("/r.json"), True)
    assert (
        a.plan(m, None, 1, False).n_tasks == 600 and a.plan(m, "sales,hr", 1, False).n_tasks == 200
    )
    assert a.plan(m, None, 1, True).n_tasks == 3 and a.domains(None, True) == ["sales"]


def test_autobench_parse_fixture(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    rd = _rd(
        cfg,
        tmp_path,
        "mock",
        "autobench",
        {
            "name": "automationbench",
            "version": "1.0.6",
            "commit": "4a8e1061",
            "max_steps": 3,
            "toolset": "api",
        },
    )
    shutil.copy(AB_FIXTURE, rd.raw_dir / "results.json")
    res = AutomationBenchAdapter(cfg).parse(rd)
    assert (
        res.n_tasks == 2
        and res.metrics["pass_rate"] == 0.0
        and res.metrics["pass_rate_sales"] == 0.0
    )
    assert res.metrics["tool_calls_per_task"] == 2.0 and res.metrics["steps_per_task"] == 3.0
    assert res.tokens_in == 68 and res.tokens_out == 20  # harness totals when no usage log
    assert (
        res.tracker == {"mock__autobench": {"v": "0.0", "note": res.note}}
        and "public tasks differ" in res.note
    )
    assert res.errors.total == 0 and res.per_task[0]["id"] == "sales.multi_hop_lookup"
    # a passed task and an aborted one
    d = json.loads(AB_FIXTURE.read_text())
    d["tasks"][0]["passed"] = True
    d["summary"]["aborted_tasks"] = ["sales.negative_selection"]
    d["summary"]["tasks_with_errors"] = 1
    (rd.raw_dir / "results.json").write_text(json.dumps(d))
    res = AutomationBenchAdapter(cfg).parse(rd)
    assert (
        res.metrics["pass_rate"] == 50.0
        and res.errors.api_errors == 1
        and res.errors.other == 1
        and len(res.warnings) == 2
    )


@pytest.mark.skipif(not AB_BIN.exists(), reason="AutomationBench harness not installed")
def test_autobench_end_to_end_via_proxy(models_yaml, tmp_path, monkeypatch):
    """Real auto-bench (pinned commit) -> BenchLab proxy -> mock, 3 sales tasks."""
    monkeypatch.setenv("BENCHLAB_HARNESS_DIR", str(Path("workspace/harnesses").resolve()))
    r = runner.invoke(
        app, ["run", "-m", "mock", "--bench", "autobench", "--smoke", "-c", str(models_yaml)]
    )
    assert r.exit_code == 0, r.output
    run_dir = next((tmp_path / "runs").glob("*_mock_autobench"))
    res = json.loads((run_dir / "result.json").read_text())
    assert res["n_tasks"] == 3 and res["metrics"]["pass_rate"] is not None and res["tokens_in"] > 0
    assert res["cost_usd"] is not None and res["cost_usd"] > 0  # priced through the proxy usage log
    assert (run_dir / "raw" / "results.json").exists() and "SMOKE" in res["tracker"][
        "mock__autobench"
    ]["note"]
    assert "sk-mock" not in (run_dir / "raw" / "command.txt").read_text()


# ---- Toolathlon -------------------------------------------------------------------------
