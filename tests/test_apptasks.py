"""App tasks: checks on answers, the shipped starter cases, the judge, and a full run
through `benchlab run` against the mock endpoint."""

import asyncio
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlab.adapters.apptasks import (
    CHECKS,
    AppTasksAdapter,
    check,
    load_cases,
    messages_for,
)
from benchlab.cli import app
from benchlab.config import load_config
from benchlab.llm import CallResult, ToolCall
from benchlab.samples import load as load_samples

runner = CliRunner()


def tc(name, args):
    return ToolCall(name=name, arguments_raw=json.dumps(args), arguments=args)


def test_text_checks():
    c = {"checks": {"contains": ["Paris"], "not_contains": "London", "max_words": 5}}
    assert check(c, "It is paris.", []) == []
    assert check(c, "London, not Paris", []) == ["answer must not contain 'London'"]
    assert check({"checks": {"regex": r"^\W*bug\W*$"}}, "Bug.", []) == []
    assert check({"checks": {"equals": "billing"}}, " Billing ", []) == []
    assert check({"checks": {"contains_any": ["a1", "b2"]}}, "zzz", [])[0].startswith(
        "answer should contain one of"
    )
    assert "words, limit 5" in check(c, "Paris " * 6, [])[0]


def test_json_checks():
    c = {"checks": {"json_includes": {"total": 1250, "currency": "eur", "city": None}}}
    fenced = '```json\n{"total": 1250.0, "currency": "EUR", "city": null}\n```'
    assert check(c, fenced, []) == []
    assert check(c, '{"total": 1200, "currency": "EUR", "city": null}', []) == [
        'JSON should include {"total": 1250, "currency": "eur", "city": null}'
    ]
    assert check(c, "not json", []) == ["answer should be valid JSON"]
    assert check({"checks": {"json_keys": ["a", "b"]}}, '{"a": 1}', []) == [
        "JSON is missing keys ['b']"
    ]


def test_tool_checks():
    c = {"checks": {"tool_call": "create_page", "tool_args": {"title": "Pricing"}}}
    assert check(c, "", [tc("create_page", {"title": "pricing", "project_id": "p_91"})]) == []
    assert "should call create_page" in check(c, "", [tc("delete_page", {})])[0]
    assert "arguments should include" in check(c, "", [tc("create_page", {"title": "Home"})])[0]
    assert check({"checks": {"no_tool_call": True}}, "bye", [tc("x", {})]) == [
        "should answer without tools, called ['x']"
    ]


def test_documents_go_into_the_system_prompt():
    msgs = messages_for(
        {
            "system": "Answer from the docs.",
            "documents": ["A", {"title": "T", "text": "B"}],
            "prompt": "Q",
        }
    )
    assert (
        msgs[0]["role"] == "system"
        and "[1] A" in msgs[0]["content"]
        and "[2] T\nB" in msgs[0]["content"]
    )
    assert msgs[1] == {"role": "user", "content": "Q"}


def test_shipped_starter_cases_are_valid():
    cases = load_cases(Path("data/evals"))
    assert len(cases) >= 20 and len({c["category"] for c in cases}) >= 8
    for c in cases:
        assert set(c["checks"]) <= CHECKS
        if "regex" in c["checks"]:
            re.compile(c["checks"]["regex"])


def test_bad_case_files_are_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("- {id: x, prompt: hi, checks: {sounds_good: true}}\n")
    with pytest.raises(ValueError, match="unknown checks"):
        load_cases(tmp_path)
    (tmp_path / "a.yaml").write_text("- {id: x, prompt: hi}\n")
    with pytest.raises(ValueError, match="has no checks"):
        load_cases(tmp_path)


def test_judge_verdicts(models_yaml):
    cfg = load_config(models_yaml)
    a = AppTasksAdapter(cfg)

    class Fake:
        def __init__(self, text):
            self.text = text

        async def acomplete(self, *args, **kw):
            return CallResult(ok=True, status="ok", latency_s=0.1, content=self.text)

    msgs = [{"role": "user", "content": "hi"}]
    grader = cfg.get("claude")
    assert asyncio.run(a.judge(Fake("PASS\nfine"), grader, "be polite", msgs, "hello", [])) is None
    got = asyncio.run(a.judge(Fake("FAIL\nIt was rude."), grader, "be polite", msgs, "go away", []))
    assert got == "judge: It was rude."
    assert a.grader(cfg.get("claude")) is None  # never graded by itself
    assert a.grader(cfg.get("mock")).id == "claude"


def test_full_run_through_the_cli(models_yaml, tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "t.yaml").write_text(
        """
- id: counts
  category: instructions
  system: Count to ten.
  prompt: Go
  checks: {contains: "1 2 3"}
- id: wants-hello
  category: instructions
  prompt: Say hello
  checks: {equals: hello}
- id: weather-tool
  category: tools
  prompt: Weather in Paris?
  tools:
    - type: function
      function:
        name: get_weather
        description: Weather for a city.
        parameters: {type: object, properties: {city: {type: string}, unit: {type: string}}}
  checks: {tool_call: get_weather, tool_args: {city: Paris}}
"""
    )
    yml = tmp_path / "models_apptasks.yaml"
    yml.write_text(
        models_yaml.read_text().replace("settings:\n", f"settings:\n  apptasks_dir: {cases}\n", 1)
    )
    r = runner.invoke(app, ["run", "-m", "mock", "-b", "apptasks", "-c", str(yml)])
    assert r.exit_code == 0, r.output
    cfg = load_config(yml)
    run = next(Path(cfg.settings.runs_dir).glob("*_mock_apptasks"))
    res = json.loads((run / "result.json").read_text())
    assert res["metrics"]["pass_rate"] == pytest.approx(66.67, abs=0.01)
    assert (
        res["metrics"]["pass_rate_tools"] == 100.0
        and res["metrics"]["pass_rate_instructions"] == 50.0
    )
    from benchlab.runs import RunDir

    detail = load_samples(RunDir(run))
    by_id = {s["id"]: s for s in detail["samples"]}
    assert detail["counts"] == {"pass": 2, "fail": 1, "error": 0}
    assert by_id["wants-hello"]["reason"] == "answer should be exactly 'hello'"
    assert by_id["counts"]["system"] == "Count to ten." and by_id["counts"]["response"].startswith(
        "1 2 3"
    )
    assert by_id["weather-tool"]["response"].startswith("get_weather(")
