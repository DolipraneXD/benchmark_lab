"""Stage 9: AutomationBench (verified against the mock), Toolathlon and OSWorld (structural)."""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlab.adapters.automationbench import KEY_VAR, PUBLIC_DOMAINS, AutomationBenchAdapter
from benchlab.adapters.base import HarnessNotReady
from benchlab.adapters.osworld import OSWorldAdapter, alias_for
from benchlab.adapters.toolathlon import SMOKE_TASKS, ToolathlonAdapter
from benchlab.cli import app
from benchlab.config import load_config
from benchlab.proxy import build_proxy_config
from benchlab.runs import RunConfig, RunDir

runner = CliRunner()
AB_FIXTURE = Path("tests/fixtures/automationbench_sales2_mock.json")
AB_BIN = Path("harnesses/automationbench/repo/.venv/bin/auto-bench")


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
        "ds_auto",
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
        res.tracker == {"mock__ds_auto": {"v": "0.0", "note": res.note}}
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
    monkeypatch.setenv("BENCHLAB_HARNESS_DIR", str(Path("harnesses").resolve()))
    r = runner.invoke(
        app, ["run", "-m", "mock", "--bench", "ds_auto", "--smoke", "-c", str(models_yaml)]
    )
    assert r.exit_code == 0, r.output
    run_dir = next((tmp_path / "runs").glob("*_mock_ds_auto"))
    res = json.loads((run_dir / "result.json").read_text())
    assert res["n_tasks"] == 3 and res["metrics"]["pass_rate"] is not None and res["tokens_in"] > 0
    assert res["cost_usd"] is not None and res["cost_usd"] > 0  # priced through the proxy usage log
    assert (run_dir / "raw" / "results.json").exists() and "SMOKE" in res["tracker"][
        "mock__ds_auto"
    ]["note"]
    assert "sk-mock" not in (run_dir / "raw" / "command.txt").read_text()


# ---- Toolathlon -------------------------------------------------------------------------
def test_toolathlon_refuses_without_setup_and_parses(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    a = ToolathlonAdapter(cfg)
    monkeypatch.setattr("benchlab.adapters.toolathlon.shutil.which", lambda _x: None)
    with pytest.raises(HarnessNotReady, match="docker or podman"):
        a.prepare()
    fake_repo = tmp_path / "repo"
    (fake_repo / "configs").mkdir(parents=True)
    probs = a.setup_problems(fake_repo)
    assert any("token_key_session" in p for p in probs) and any(
        "check_installation" in p for p in probs
    )
    assert a.command(cfg.get("mock"), Path("/d"), 4) == [
        "bash",
        "scripts/run_parallel.sh",
        "mock",
        "/d",
        "unified",
        "4",
    ]
    assert a.plan(cfg.get("mock"), None, 1, True).subset_ids == list(SMOKE_TASKS)
    rd = _rd(cfg, tmp_path, "mock", "mimo_tool", {"name": "toolathlon", "commit": "9be8d8fe"})
    stats = {
        "total_tasks": 108,
        "average_success_rate": 0.2963,
        "average_turns": 21.5,
        "average_tool_calls": 28.0,
        "status_breakdown": {
            "preprocess": {"fail_count": 1, "timeout_count": 0},
            "running": {"timeout_count": 3, "fail_count": 2, "max_turns_count": 4, "null_count": 0},
            "evaluation": {
                "pass_count": 32,
                "fail_count": 75,
                "null_count": 1,
                "pass_tasks": ["a"],
                "fail_tasks": ["b"],
            },
        },
    }
    (rd.raw_dir / "dump").mkdir()
    (rd.raw_dir / "dump" / "eval_stats.json").write_text(json.dumps(stats))
    res = a.parse(rd)
    assert res.metrics["success_rate"] == pytest.approx(29.63) and res.n_tasks == 108
    assert res.errors.timeouts == 3 and res.errors.api_errors == 2 and res.errors.other == 2
    assert (
        res.tracker["mock__mimo_tool"]["v"] == "29.6"
        and "max_tokens 65536" in res.note
        and res.warnings
    )


# ---- OSWorld -----------------------------------------------------------------------------
def test_osworld_alias_refusal_and_parse(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    a = OSWorldAdapter(cfg)
    m = cfg.get("mock")
    assert alias_for(m) == "gpt-benchlab-mock"
    pc = build_proxy_config(cfg, aliases={alias_for(m): "mock"})
    names = [e["model_name"] for e in pc["model_list"]]
    assert "gpt-benchlab-mock" in names and "mock" in names
    entry = next(e for e in pc["model_list"] if e["model_name"] == "gpt-benchlab-mock")
    assert (
        entry["model_info"]["alias_of"] == "mock"
        and entry["litellm_params"]["model"]
        == next(e for e in pc["model_list"] if e["model_name"] == "mock")["litellm_params"]["model"]
    )
    monkeypatch.setattr("benchlab.adapters.osworld.shutil.which", lambda _x: None)
    with pytest.raises(HarnessNotReady, match="needs VMs"):
        a.prepare()
    cmd = a.command(m, Path("/res"), Path("/meta.json"), "/v/python")
    assert (
        cmd[cmd.index("--model") + 1] == "gpt-benchlab-mock"
        and cmd[cmd.index("--provider_name") + 1] == "docker"
    )
    assert cmd[cmd.index("--max_steps") + 1] == "15" and cmd[cmd.index("--num_envs") + 1] == "4"
    rd = _rd(
        cfg,
        tmp_path,
        "mock",
        "mimo_osw",
        {
            "name": "osworld",
            "commit": "b138d348",
            "meta": "test_all.json",
            "max_steps": 15,
            "provider": "docker",
        },
    )
    base = rd.raw_dir / "results" / "pyautogui" / "screenshot" / "gpt-benchlab-mock"
    for dom, ex, val in (
        ("os", "e1", "1.0"),
        ("os", "e2", "0.0"),
        ("chrome", "e3", "1"),
        ("chrome", "e4", "0.0"),
    ):
        (base / dom / ex).mkdir(parents=True)
        (base / dom / ex / "result.txt").write_text(val)
    (base / "chrome" / "unfinished").mkdir()
    res = a.parse(rd)
    assert (
        res.n_tasks == 4
        and res.metrics["success_rate"] == 50.0
        and res.metrics["success_rate_os"] == 50.0
    )
    assert res.tracker["mock__mimo_osw"]["v"] == "50.0" and any("4/369" in w for w in res.warnings)
    assert res.errors.other == 365 and "gpt-benchlab-mock" in res.note
