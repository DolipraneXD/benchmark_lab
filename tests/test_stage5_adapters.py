"""Unit tests for the Aider Polyglot, LiveCodeBench and RULER adapters."""

import json
from pathlib import Path

import pytest

from benchlab.adapters.aider import AiderAdapter, parse_stats_output
from benchlab.adapters.livecodebench import WRAPPER, LCBAdapter
from benchlab.config import load_config
from benchlab.runs import RunConfig, RunDir

STATS = """
- dirname: 2026-10-13-10-00-00--benchlab
  test_cases: 3
  model: openai/mock
  edit_format: diff
  commit_hash: abc1234
  pass_rate_1: 33.3
  pass_rate_2: 66.7
  percent_cases_well_formed: 100.0
  error_outputs: 1
  num_malformed_responses: 0
  syntax_errors: 0
  test_timeouts: 0
  exhausted_context_windows: 0
  seconds_per_case: 12.0
  total_cost: 0.0123
"""


def test_parse_stats_output():
    d = parse_stats_output(STATS)
    assert (
        d["pass_rate_2"] == 66.7
        and d["test_cases"] == 3
        and d["edit_format"] == "diff"
        and d["error_outputs"] == 1
    )


def test_aider_command_and_docker(models_yaml):
    cfg = load_config(models_yaml)
    a = AiderAdapter(cfg)
    args = a.benchmark_args(cfg.get("mock"), None, False)
    assert (
        args[:3] == ["benchlab", "--model", "openai/mock"]
        and "--edit-format" in args
        and "diff" in args
    )
    assert "--threads" in args and args[args.index("--threads") + 1] == "4" and "--cont" in args
    assert "--num-tests" not in args
    assert "--num-tests" in a.benchmark_args(cfg.get("mock"), None, True)
    assert a.benchmark_args(cfg.get("mock"), "20", False)[-2:] == ["--num-tests", "20"]
    cmd = a.docker_command(
        Path("/h/aider"),
        Path("/r/benchmarks"),
        "echo hi",
        {"OPENAI_API_KEY": "", "OPENAI_API_BASE": ""},
    )
    s = " ".join(cmd)
    assert s.startswith("docker run --rm --memory=12g") and "--network=host" in s
    assert (
        "-v /h/aider:/aider -v /r/benchmarks:/benchmarks" in s
        and "-e AIDER_BENCHMARK_DIR=/benchmarks" in s
    )
    assert (
        "-e OPENAI_API_BASE -e OPENAI_API_KEY" in s and "sk-" not in s
    )  # env names only, no values
    assert (
        cmd[-3:] == [f"benchlab-aider:{cfg.settings.aider_ref}", "bash", "-lc"][:2] + ["-lc"]
        or cmd[-1] == "echo hi"
    )


def test_aider_parse(models_yaml, tmp_path, monkeypatch):
    cfg = load_config(models_yaml)
    a = AiderAdapter(cfg)
    rd = RunDir.create(tmp_path / "runs", "mock", "aider")
    tdir = rd.raw_dir / "benchmarks" / "2026-10-13-10-00-00--benchlab"
    cases = {
        ("python", "zipper"): {
            "tests_outcomes": [True],
            "cost": 0.01,
            "duration": 5,
            "prompt_tokens": 1000,
            "completion_tokens": 100,
        },
        ("rust", "wordy"): {
            "tests_outcomes": [False, True],
            "cost": 0.02,
            "duration": 9,
            "prompt_tokens": 2000,
            "completion_tokens": 300,
            "num_malformed_responses": 1,
        },
        ("go", "zebra"): {
            "tests_outcomes": [False, False],
            "cost": 0.0,
            "duration": 4,
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "num_error_outputs": 1,
        },
    }
    for (lang, name), d in cases.items():
        p = tdir / lang / "exercises" / "practice" / name
        p.mkdir(parents=True)
        (p / ".aider.results.json").write_text(
            json.dumps({**d, "testcase": name, "model": "openai/mock", "edit_format": "diff"})
        )
    (rd.raw_dir / "stats.txt").write_text(STATS)
    rd.write_config(
        RunConfig(
            model_id="mock",
            bench="aider",
            model={},
            request_params={"temperature": 0.0, "max_tokens": 4096},
            per_model_exceptions={},
            harness={"commit": "abc1234", "polyglot_commit": "7e0611e"},
            subset={},
            trials=1,
            smoke=True,
        )
    )
    res = a.parse(rd)
    assert (
        res.primary_metric == "pass_rate_2"
        and res.metrics["pass_rate_2"] == 66.7
        and res.metrics["pass_rate_1"] == 33.3
    )
    assert res.n_tasks == 3 and res.errors.api_errors == 1 and res.errors.malformed_tool_calls == 1
    assert res.tokens_in == 3500 and res.cost_usd == pytest.approx(3500 * 1e-6 + 450 * 2e-6)
    assert res.tracker == {"mock__aider": {"v": "66.7", "note": res.note}} and "SMOKE" in res.note
    assert [t["passed"] for t in res.per_task] == [
        False,
        True,
        True,
    ]  # sorted by path: go, python, rust
    # without the harness stats, pass rates are aggregated from the per-exercise outcomes (same numbers)
    (rd.raw_dir / "stats.txt").unlink()
    res2 = a.parse(rd)
    assert res2.metrics["pass_rate_2"] == pytest.approx(66.6667, abs=1e-3) and res2.metrics[
        "pass_rate_1"
    ] == pytest.approx(33.3333, abs=1e-3)


def test_lcb_command_window_and_wrapper(models_yaml, monkeypatch):
    cfg = load_config(models_yaml)
    a = LCBAdapter(cfg)
    m = cfg.get("mock")
    args = a.harness_args(m, False)
    assert args[:5] == ["--model", "mock", "--scenario", "codegeneration", "--evaluate"]
    assert (
        "--release_version" in args
        and args[args.index("--n") + 1] == "10"
        and args[args.index("--temperature") + 1] == "0.2"
    )
    assert (
        args[args.index("--max_tokens") + 1] == "4096"
        and args[args.index("--multiprocess") + 1] == "4"
    )
    assert "--start_date" not in args and "--continue_existing_with_eval" in args
    plan = a.plan(m, None, 1, False)
    assert any("no knowledge_cutoff" in n for n in plan.notes)
    m.knowledge_cutoff = "2024-10-01"
    args = a.harness_args(m, False)
    assert args[args.index("--start_date") + 1] == "2024-10-01"
    m.knowledge_cutoff = "2025-06-01"  # after the last problem of release_v6
    assert any("no problems would remain" in n for n in a.window(m)[2])
    smoke = a.harness_args(m, True)
    assert smoke[smoke.index("--n") + 1] == "1" and "--end_date" in smoke
    cmd = a.build_command(m, False, Path("/v/bin"))
    assert cmd[:2] == ["/v/bin/python", "-c"] and cmd[2] == WRAPPER and cmd[3] == "mock"
    assert "LMStyle.OpenAIChat" in WRAPPER and "LanguageModelStore[name] = lm" in WRAPPER


def test_lcb_parse(models_yaml, tmp_path):
    cfg = load_config(models_yaml)
    a = LCBAdapter(cfg)
    rd = RunDir.create(tmp_path / "runs", "mock", "lcb")
    out = rd.raw_dir / "output" / "mock"
    out.mkdir(parents=True)
    rows = [
        {
            "question_id": "q1",
            "difficulty": "easy",
            "contest_date": "2024-11-02T00:00:00",
            "pass@1": 1.0,
            "graded_list": [True] * 10,
            "output_list": ["x"] * 10,
        },
        {
            "question_id": "q2",
            "difficulty": "hard",
            "contest_date": "2025-01-15T00:00:00",
            "pass@1": 0.2,
            "graded_list": [True, True] + [False] * 8,
            "output_list": ["x"] * 9 + [""],
        },
        {
            "question_id": "q3",
            "difficulty": "medium",
            "contest_date": "2024-12-01T00:00:00",
            "pass@1": 0.0,
            "graded_list": [False] * 10,
            "output_list": ["x"] * 10,
        },
    ]
    (out / "Scenario.codegeneration_10_0.2_eval_all.json").write_text(json.dumps(rows))
    (out / "Scenario.codegeneration_10_0.2_eval.json").write_text(
        json.dumps([{"pass@1": 0.4, "pass@5": 0.6, "detail": {}}])
    )
    rd.write_config(
        RunConfig(
            model_id="mock",
            bench="lcb",
            model={"knowledge_cutoff": "2024-10-01"},
            request_params={},
            per_model_exceptions={},
            harness={
                "commit": "28fef95ea8",
                "release_version": "release_v6",
                "n": 10,
                "temperature": 0.2,
            },
            subset={},
            trials=1,
        )
    )
    res = a.parse(rd)
    assert res.primary_metric == "pass@1" and res.metrics["pass@1"] == pytest.approx(40.0)
    assert res.metrics["pass@1_easy"] == 100.0 and res.metrics["pass@1_hard"] == pytest.approx(20.0)
    assert (
        res.metrics["harness_pass@5"] == pytest.approx(60.0) and res.metrics["n_empty_outputs"] == 1
    )
    assert res.n_tasks == 3 and res.trials == 10 and res.errors.api_errors == 1
    assert res.harness["window"] == ("2024-11-02", "2025-01-15")
    assert "window 2024-11-02..2025-01-15" in res.note and "cutoff 2024-10-01" in res.note
    assert res.tracker == {"mock__lcb": {"v": "40.0", "note": res.note}}
