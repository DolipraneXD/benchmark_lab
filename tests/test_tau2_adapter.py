"""Unit tests for the τ²-bench adapters (command building, subsets, parsing)."""

import json
from pathlib import Path

import pytest

from benchlab.adapters.base import HarnessNotReady
from benchlab.adapters.tau2 import (
    BASE_SPLIT_SIZES,
    SUBSET_SEED,
    Tau2Airline,
    Tau2Retail,
    Tau2Telecom,
    classify_termination,
)
from benchlab.config import load_config
from benchlab.runs import RunConfig, RunDir

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_harness(tmp_path, domain="airline", n=50):
    """A fake tau2 clone with a split file and a tau2 binary placeholder."""
    repo = tmp_path / "repo"
    d = repo / "data" / "tau2" / "domains" / domain
    d.mkdir(parents=True)
    (d / "split_tasks.json").write_text(
        json.dumps({"base": [str(i) for i in range(n)], "train": [], "test": []})
    )
    bin_dir = repo / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "tau2").write_text("")
    return repo, bin_dir


def _adapter(cls, cfg, tmp_path, domain):
    a = cls(cfg)
    a.subsets_root = tmp_path / "subsets"
    repo, bin_dir = _fake_harness(tmp_path, domain, BASE_SPLIT_SIZES[domain])
    a.harness = lambda: (repo, bin_dir)
    return a


def test_subset_is_fixed_and_reused(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    a = _adapter(Tau2Airline, cfg, tmp_path, "airline")
    ids = a.task_ids(None, False)
    assert len(ids) == 20 and ids == sorted(ids) and set(ids) <= {str(i) for i in range(50)}
    saved = json.loads((tmp_path / "subsets" / "tau2_airline_20.json").read_text())
    assert saved["ids"] == ids and saved["seed"] == SUBSET_SEED and saved["pool_size"] == 50
    assert a.task_ids(None, False) == ids  # same file, same ids
    assert a.task_ids("all", False) is None  # whole base split
    assert a.task_ids("50", False) is None
    assert len(a.task_ids("5", False)) == 5
    assert a.task_ids(None, True) == ids[:2]  # smoke: first 2 of the fixed subset
    plan = a.plan(cfg.get("mock"), None, 3, False)
    assert plan.n_tasks == 20 and plan.trials == 3 and plan.subset_ids == ids
    assert "user sim: claude" in plan.notes[0]


def test_command_routes_agent_and_fixed_user_sim_through_proxy(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    a = _adapter(Tau2Retail, cfg, tmp_path, "retail")
    cmd = a.build_command(
        cfg.get("mock"), Path("/abs/save"), None, 3, False, "http://p/v1", "sk-master"
    )
    s = " ".join(cmd)
    assert cmd[0] == "tau2" and "--domain retail" in s and "--task-split-name base" in s
    assert "--agent-llm openai/mock" in s and "--user-llm openai/claude" in s
    agent_args = json.loads(cmd[cmd.index("--agent-llm-args") + 1])
    user_args = json.loads(cmd[cmd.index("--user-llm-args") + 1])
    assert agent_args == {
        "temperature": 0.0,
        "max_tokens": 4096,
        "api_base": "http://p/v1",
        "api_key": "sk-master",
    }
    # the user simulator is claude: temperature removed (per-model exception), same proxy
    assert "temperature" not in user_args and user_args["api_base"] == "http://p/v1"
    assert "--num-trials 3" in s and "--max-steps 200" in s and "--seed 300" in s
    assert "--save-to /abs/save --auto-resume" in s and "--max-concurrency 4" in s
    assert "--task-ids" in cmd and len(cmd) - cmd.index("--task-ids") - 1 == 20
    # smoke: 2 tasks, 1 trial, short conversations
    s2 = " ".join(a.build_command(cfg.get("mock"), Path("/x"), None, 3, True, "u", "k"))
    assert "--num-trials 1" in s2 and "--max-steps 20" in s2 and s2.count(" --task-ids ") == 1
    assert len(s2.split("--task-ids ")[1].split()) == 2
    # user-sim params from settings override the model's own (tau2_user_model_params)
    cfg.settings.tau2_user_model_params = {"temperature": 0.7}
    user_args = json.loads(
        a.build_command(cfg.get("mock"), Path("/x"), None, 1, False, "u", "k")[
            cmd.index("--user-llm-args") + 1
        ]
    )
    assert user_args["temperature"] == 0.7


def test_prepare_requires_user_sim_key(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    a = _adapter(Tau2Telecom, cfg, tmp_path, "telecom")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(HarnessNotReady, match="user simulator model claude"):
        a.prepare()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-000000000")
    monkeypatch.setattr(
        "benchlab.adapters.tau2.frozen_versions",
        lambda b, names=None: {"tau2": "1.0.1", "litellm": "1.81.11"},
    )
    monkeypatch.setattr("benchlab.adapters.tau2.repo_commit", lambda r: "fc0055d")
    info = a.prepare()
    assert (
        info["tau2"] == "1.0.1" and info["ref"] == "v1.0.1" and info["user_model"]["id"] == "claude"
    )
    assert "temperature" not in info["user_llm_args"] and "api_key" not in info["user_llm_args"]


def test_classify_termination():
    assert classify_termination("user_stop") == "" and classify_termination("agent_stop") == ""
    assert classify_termination("timeout") == "timeout"
    assert classify_termination("infrastructure_error") == "api_error"
    assert (
        classify_termination("too_many_errors") == "other"
        and classify_termination("max_steps") == "other"
    )


def _run_dir(tmp_path, dump, model_id="mock", bench="t2air", trials=3, smoke=False):
    rd = RunDir.create(tmp_path / "runs", model_id, bench)
    (rd.raw_dir / "tau2_airline").mkdir()
    (rd.raw_dir / "tau2_airline" / "results.json").write_text("{}")
    (rd.raw_dir / "tau2_airline_dump.json").write_text(json.dumps(dump))
    rd.write_config(
        RunConfig(
            model_id=model_id,
            bench=bench,
            model={},
            request_params={"temperature": 0.0, "max_tokens": 4096},
            per_model_exceptions={},
            harness={"commit": "abc1234", "tau2": "1.0.1", "max_steps": 200},
            subset={},
            trials=trials,
            smoke=smoke,
        )
    )
    return rd


def test_normalize_synthetic_dump(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    sims = []
    # 2 tasks x 3 trials: task A passes 3/3, task B passes 1/3 with one infra error
    for trial in range(3):
        sims.append(
            {
                "task_id": "A",
                "trial": trial,
                "reward": 1.0,
                "termination_reason": "user_stop",
                "duration": 10.0,
                "tokens_in": 1000,
                "tokens_out": 100,
                "n_tool_calls": 3,
            }
        )
    sims.append(
        {
            "task_id": "B",
            "trial": 0,
            "reward": 1.0,
            "termination_reason": "agent_stop",
            "duration": 5,
            "tokens_in": 500,
            "tokens_out": 50,
            "n_tool_calls": 2,
        }
    )
    sims.append(
        {
            "task_id": "B",
            "trial": 1,
            "reward": 0.0,
            "termination_reason": "max_steps",
            "duration": 50,
            "tokens_in": 5000,
            "tokens_out": 500,
            "n_tool_calls": 20,
        }
    )
    sims.append(
        {
            "task_id": "B",
            "trial": 2,
            "reward": 0.0,
            "termination_reason": "infrastructure_error",
            "duration": 1,
            "tokens_in": 0,
            "tokens_out": 0,
            "n_tool_calls": 0,
        }
    )
    dump = {
        "metrics": {
            "avg_reward": 0.8,
            "total_tasks": 2,
            "total_simulations": 5,
            "infra_error_count": 1,
            "termination_user_stop": 3,
            "termination_agent_stop": 1,
            "termination_max_steps": 1,
        },
        "pass_hat_ks": {"1": 0.8, "2": 0.6},
        "simulations": sims,
        "info": {
            "num_trials": 3,
            "agent_info": {"llm": "openai/mock"},
            "user_info": {"llm": "openai/claude"},
        },
        "n_tasks": 2,
    }
    rd = _run_dir(tmp_path, dump)
    a = Tau2Airline(cfg)
    a.harness = lambda: (tmp_path, tmp_path)
    res = a.parse(rd)
    assert (
        res.primary_metric == "pass^1"
        and res.metrics["pass^1"] == 80.0
        and res.metrics["pass^2"] == 60.0
    )
    assert res.metrics["avg_reward"] == 80.0 and res.n_tasks == 2 and res.trials == 3
    assert res.errors.api_errors == 1 and res.errors.other == 1 and res.errors.total == 2
    assert res.tokens_in == 8500 and res.cost_usd == pytest.approx(8500 * 1e-6 + 850 * 2e-6)
    assert res.tracker == {"mock__t2air": {"v": "80.0", "note": res.note}}
    assert res.note.startswith("tau2-bench v1.0.1 airline base split, 2 tasks (fixed subset, seed")
    assert (
        "user sim claude (claude-sonnet-5-5) temp default" in res.note and "pass^2=60.0" in res.note
    )
    assert res.harness["agent_llm"] == "openai/mock" and res.harness["user_llm"] == "openai/claude"
    assert any("infrastructure" in w for w in res.warnings)


@pytest.mark.skipif(
    not (FIXTURES / "tau2_airline_smoke_mock.dump.json").exists(), reason="fixture not present"
)
def test_parse_real_tau2_dump(models_yaml, tmp_path):
    """Dump produced by tau2-bench v1.0.1 compute_metrics on a smoke run against the mock server."""
    cfg = load_config(models_yaml)
    dump = json.loads((FIXTURES / "tau2_airline_smoke_mock.dump.json").read_text())
    rd = _run_dir(tmp_path, dump, trials=1, smoke=True)
    a = Tau2Airline(cfg)
    a.harness = lambda: (tmp_path, tmp_path)
    res = a.parse(rd)
    assert "pass^1" in res.metrics and 0.0 <= res.metrics["pass^1"] <= 100.0
    assert res.n_tasks == 2 and res.trials == 1 and len(res.per_task) == 2
    assert all(t["termination_reason"] for t in res.per_task)
    assert "SMOKE" in res.note and list(res.tracker) == ["mock__t2air"]
