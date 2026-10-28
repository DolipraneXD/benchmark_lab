"""External red-team import (promptfoo CSV) -> redteam run folder, results.csv, report."""

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from benchlab.adapters.redteam import read_rows, verdict
from benchlab.cli import app
from benchlab.config import load_config
from benchlab.report import build_report

runner = CliRunner()
COLS = [
    "Test ID",
    "Plugin",
    "Plugin ID",
    "Strategy",
    "Target",
    "Prompt",
    "Response",
    "Pass",
    "Score",
    "Reason",
    "Timestamp",
]
ROWS = [
    (
        "1",
        "Gender Bias",
        "bias:gender",
        "Thinking: refuse. No.",
        "Pass (1)",
        "1",
        "All assertions passed",
    ),
    ("2", "Gender Bias", "bias:gender", "", "Fail", "", ""),  # never graded
    ("3", "Profanity", "harmful:profanity", "rude words", "Fail (0)", "", "contains profanity"),
    ("4", "Profanity", "harmful:profanity", "ok", "Pass (0.97)", "0.97", "mostly fine"),
    ("5", "PII via Direct Exposure", "pii:direct", "no", "Pass (1)", "1", "All assertions passed"),
]


def _csv(tmp_path: Path) -> Path:
    p = tmp_path / "report_lab.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(COLS)
        for tid, plugin, pid, resp, passed, score, reason in ROWS:
            w.writerow(
                [
                    tid,
                    plugin,
                    pid,
                    "basic",
                    "mock target",
                    "prompt",
                    resp,
                    passed,
                    score,
                    reason,
                    "2026-10-27T15:43:38.506Z",
                ]
            )
    return p


def test_verdicts(tmp_path):
    rows = read_rows(_csv(tmp_path))
    assert [verdict(r) for r in rows] == ["pass", "error", "fail", "pass", "pass"]


def test_import_cli_and_report(models_yaml, tmp_path):
    r = runner.invoke(
        app, ["import-redteam", str(_csv(tmp_path)), "-m", "mock", "-c", str(models_yaml)]
    )
    assert r.exit_code == 0, r.output
    run = next((tmp_path / "runs").glob("*_mock_redteam"))
    assert (run / "raw" / "report_lab.csv").exists()
    res = json.loads((run / "result.json").read_text())
    m = res["metrics"]
    assert m["pass_rate"] == 75.0  # 3 pass / 4 graded
    assert m["pass_rate_errors_as_fail"] == 60.0
    assert m["pass_rate_bias"] == 100.0 and m["plugin:harmful:profanity"] == 50.0
    assert res["errors"]["api_errors"] == 1 and res["tracker"]["mock__redteam"]["v"] == "75.0"
    assert any("Thinking:" in w for w in res["warnings"])

    rep = build_report(load_config(models_yaml), str(tmp_path / "report.html"))
    assert rep.tracker["mock__redteam"]["v"] == "75.0"
    whys = [s["why"] for s in rep.suspicious if s["bench"] == "redteam"]
    assert not any("no cost" in w for w in whys)  # imported: no usage log expected
    assert "Red team (external lab)" in Path(rep.html_path).read_text()


def test_import_rejects_other_csv(models_yaml, tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    r = runner.invoke(app, ["import-redteam", str(p), "-m", "mock", "-c", str(models_yaml)])
    assert r.exit_code == 2 and "not a promptfoo red-team CSV" in r.output
