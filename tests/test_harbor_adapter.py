"""Unit tests for the Harbor adapters (Terminal-Bench / SWE-bench Verified)."""

import json
from pathlib import Path

import pytest

from benchlab.adapters.base import HarnessNotReady
from benchlab.adapters.harbor import (
    AGENT,
    SUBSET_SEED,
    SWEBenchAdapter,
    TerminalBench21Adapter,
    TerminalBenchAdapter,
    classify_exception,
)
from benchlab.config import load_config
from benchlab.runs import RunConfig, RunDir

FAKE_REGISTRY = [
    {
        "name": "terminal-bench",
        "version": "2.0",
        "tasks": [
            {"name": f"task-{i:02d}", "git_url": "https://x/tb2.git", "git_commit_id": "69671fba00"}
            for i in range(89)
        ],
    },
    {
        "name": "swebench-verified",
        "version": "1.0",
        "tasks": [
            {
                "name": f"django__django-{10000 + i}",
                "git_url": "https://x/hd.git",
                "git_commit_id": "86723674aa",
            }
            for i in range(500)
        ],
    },
]


def _adapter(cls, cfg, tmp_path, monkeypatch, docker=True):
    a = cls(cfg)
    a.subsets_root = tmp_path / "subsets"
    bin_dir = tmp_path / "hbin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "harbor").write_text("")
    monkeypatch.setattr(a, "harness_bin", lambda: bin_dir)
    monkeypatch.setattr(a, "registry", lambda: FAKE_REGISTRY)
    monkeypatch.setattr("benchlab.adapters.harbor.docker_available", lambda: docker)
    monkeypatch.setattr(
        "benchlab.adapters.harbor.frozen_versions", lambda b, names=None: {"harbor": "0.24.0"}
    )
    return a


def test_subsets_fixed_and_reused(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    tb = _adapter(TerminalBenchAdapter, cfg, tmp_path, monkeypatch)
    ids = tb.task_ids(None, False)
    assert (
        len(ids) == 50 and ids == sorted(ids) and set(ids) <= {f"task-{i:02d}" for i in range(89)}
    )
    saved = json.loads((tmp_path / "subsets" / "tb_terminal-bench_2.0_50.json").read_text())
    assert saved["ids"] == ids and saved["seed"] == SUBSET_SEED and saved["pool_size"] == 89
    assert tb.task_ids(None, False) == ids and tb.task_ids("all", False) is None
    assert tb.task_ids(None, True) == ids[:2]
    swe = _adapter(SWEBenchAdapter, cfg, tmp_path, monkeypatch)
    assert (
        len(swe.task_ids(None, False)) == 50
        and (tmp_path / "subsets" / "swe_swebench-verified_1.0_50.json").exists()
    )
    plan = tb.plan(cfg.get("mock"), None, 1, False)
    assert plan.n_tasks == 50 and plan.subset_ids == ids and "terminal-bench@2.0" in plan.notes[0]
    assert tb.plan(cfg.get("mock"), "all", 1, False).n_tasks == 89
    assert tb.plan(cfg.get("mock"), None, 1, True).n_tasks == 2


def test_repo_subsets_match_registry_naming():
    """The committed subset files were drawn from the real registry with the same seed."""
    for name, pool in (("tb_terminal-bench_2.0_50", 89), ("swe_swebench-verified_1.0_50", 500)):
        p = Path("subsets") / f"{name}.json"
        assert p.exists(), p
        d = json.loads(p.read_text())
        assert (
            d["n"] == 50
            and d["pool_size"] == pool
            and d["seed"] == SUBSET_SEED
            and len(set(d["ids"])) == 50
        )


def test_command_and_resume(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    tb = _adapter(TerminalBenchAdapter, cfg, tmp_path, monkeypatch)
    cmd = tb.build_command(cfg.get("mock"), Path("/jobs"), None, False, "http://p/v1")
    s = " ".join(cmd)
    assert s.startswith(
        "harbor run -d terminal-bench@2.0 -a terminus-2 -m openai/mock --ak api_base=http://p/v1 --ak temperature=0.0"
    )
    assert "-n 4 -k 1 --jobs-dir /jobs --job-name tb -q" in s and cmd.count("-i") == 50
    # claude: no temperature kwarg (per-model exception)
    s2 = " ".join(tb.build_command(cfg.get("claude"), Path("/jobs"), None, False, "http://p/v1"))
    assert "temperature" not in s2 and "--ak api_base=http://p/v1" in s2
    assert " ".join(tb.build_command(cfg.get("mock"), Path("/j"), None, True, "u")).count("-i") == 2
    assert tb.resume_command(Path("/jobs/tb")) == ["harbor", "job", "resume", "-p", "/jobs/tb"]
    assert AGENT == "terminus-2"


def test_prepare_checks(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    tb = _adapter(TerminalBenchAdapter, cfg, tmp_path, monkeypatch, docker=False)
    with pytest.raises(HarnessNotReady, match="Docker"):
        tb.prepare()
    tb = _adapter(TerminalBenchAdapter, cfg, tmp_path, monkeypatch)
    info = tb.prepare()
    assert (
        info["dataset"] == "terminal-bench@2.0"
        and info["dataset_tasks_total"] == 89
        and info["agent"] == AGENT
    )
    assert info["dataset_git_commit"] == "69671fba00" and info["harbor"] == "0.24.0"
    tb21 = _adapter(TerminalBench21Adapter, cfg, tmp_path, monkeypatch)
    with pytest.raises(HarnessNotReady, match=r"terminal-bench@2.1.*available versions.*2\.0"):
        tb21.prepare()


def test_classify_exception():
    assert classify_exception(None) == "" and classify_exception("AgentTimeoutError") == "timeout"
    assert classify_exception("VerifierTimeoutError") == "timeout"
    assert (
        classify_exception("APIConnectionError") == "api_error"
        and classify_exception("RateLimitError") == "api_error"
    )
    assert (
        classify_exception("EnvironmentBuildError") == "other"
        and classify_exception("CancelledError") == "other"
    )


def _trial(task, reward=None, exc=None, tin=1000, tout=100):
    t = {
        "task_name": task,
        "trial_name": f"{task}__abc",
        "agent_info": {"name": "terminus-2", "version": "2"},
        "agent_result": {"n_input_tokens": tin, "n_output_tokens": tout, "cost_usd": None},
        "agent_execution": {
            "started_at": "2026-10-13T10:00:00+00:00",
            "finished_at": "2026-10-13T10:05:00+00:00",
        },
        "verifier_result": None,
        "exception_info": None,
    }
    if reward is not None:
        t["verifier_result"] = {"rewards": {"reward": reward}}
    if exc:
        t["exception_info"] = {"exception_type": exc, "exception_message": "boom"}
    return t


def test_parse_job_dir(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    tb = _adapter(TerminalBenchAdapter, cfg, tmp_path, monkeypatch)
    rd = RunDir.create(tmp_path / "runs", "mock", "tb")
    job = rd.raw_dir / "jobs" / "tb"
    for t in (
        _trial("a", 1.0),
        _trial("b", 0.0),
        _trial("c", exc="AgentTimeoutError", tin=5000, tout=500),
        _trial("d", 1.0, tin=2000, tout=200),
    ):
        d = job / t["trial_name"]
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(t))
    (job / ".sources").mkdir()
    (job / "result.json").write_text(
        json.dumps(
            {
                "started_at": "2026-10-13T10:00:00+00:00",
                "finished_at": "2026-10-13T11:00:00+00:00",
                "stats": {
                    "n_completed_trials": 4,
                    "n_errored_trials": 1,
                    "n_input_tokens": 9000,
                    "n_output_tokens": 900,
                    "cost_usd": None,
                },
            }
        )
    )
    rd.write_config(
        RunConfig(
            model_id="mock",
            bench="tb",
            model={},
            request_params={"temperature": 0.0, "max_tokens": 4096},
            per_model_exceptions={},
            harness={
                "harbor": "0.24.0",
                "dataset_tasks_total": 89,
                "dataset_git_commit": "69671fba00",
            },
            subset={},
            trials=1,
        )
    )
    res = tb.parse(rd)
    assert (
        res.primary_metric == "resolved" and res.metrics["resolved"] == 50.0
    )  # 2 of 4 trials, error counts as unsolved
    assert (
        res.metrics["resolved_excl_errors"] == pytest.approx(66.6667, abs=1e-3)
        and res.metrics["n_errored"] == 1
    )
    assert (
        res.errors.timeouts == 1 and res.errors.total == 1 and res.n_tasks == 4 and res.trials == 1
    )
    assert res.tokens_in == 9000 and res.cost_usd == pytest.approx(9000 * 1e-6 + 900 * 2e-6)
    assert res.wall_time_s == 3600.0 and res.harness["job_stats"]["n_errored_trials"] == 1
    assert res.tracker == {"mock__tb": {"v": "50.0", "note": res.note}}
    assert res.note.startswith(
        "harbor 0.24.0 terminal-bench@2.0, agent terminus-2, fixed subset of 4 (seed"
    )
    assert any("errored" in w for w in res.warnings)
    assert [p["solved"] for p in res.per_task] == [True, False, False, True]
