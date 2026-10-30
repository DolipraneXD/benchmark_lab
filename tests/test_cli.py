from typer.testing import CliRunner

from benchlab.cli import app

runner = CliRunner()


def test_models_and_check_cli(models_yaml):
    r = runner.invoke(app, ["models", "-c", str(models_yaml), "--keys"])
    assert r.exit_code == 0, r.output
    assert "mock" in r.output and "openai/mock-model" in r.output

    r = runner.invoke(app, ["check", "-m", "mock", "-c", str(models_yaml)])
    assert r.exit_code == 0, r.output
    assert "OK" in r.output and "get_weather" in r.output

    r = runner.invoke(app, ["check", "-m", "notools,broken", "-c", str(models_yaml)])
    assert r.exit_code == 1
    assert "PROBLEM" in r.output and "hint" in r.output  # self-hosted hint printed for notools


def test_run_dry_run_and_not_integrated(models_yaml):
    r = runner.invoke(
        app, ["run", "-m", "mock", "--stage", "1", "--dry-run", "-c", str(models_yaml)]
    )
    assert r.exit_code == 0, r.output
    assert "gpqa" in r.output and "bfcl" in r.output and "estimated total" in r.output
    r = runner.invoke(app, ["run", "-m", "mock", "--bench", "redteam", "-c", str(models_yaml)])
    assert r.exit_code == 1 and "nothing to run" in r.output  # imported only, never run
    r = runner.invoke(app, ["run", "-m", "mock", "--bench", "nope", "-c", str(models_yaml)])
    assert r.exit_code == 2 and "unknown bench" in r.output
    r = runner.invoke(app, ["run", "-m", "mock", "-c", str(models_yaml)])
    assert r.exit_code == 2  # no selector


def test_proxy_config_cli(models_yaml, tmp_path):
    out = tmp_path / "cfg.yaml"
    r = runner.invoke(app, ["proxy-config", "-c", str(models_yaml), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and "os.environ/MOCK_API_KEY" in out.read_text()


def test_one_crashing_benchmark_does_not_stop_the_others(models_yaml, monkeypatch):
    from benchlab.adapters.ttft import TTFTAdapter

    calls = []

    def boom(self):
        calls.append("gpqa")
        raise PermissionError("Permission denied: '/opt/x/python'")

    monkeypatch.setattr("benchlab.adapters.inspect_bench.GPQAAdapter.prepare", boom)
    orig = TTFTAdapter.prepare
    monkeypatch.setattr(TTFTAdapter, "prepare", lambda self: (calls.append("ttft"), orig(self))[1])
    r = runner.invoke(
        app, ["run", "-m", "mock", "-b", "gpqa,ttft", "--smoke", "-c", str(models_yaml)]
    )
    assert r.exit_code == 1  # the failure is reported in the exit code
    assert calls == ["gpqa", "ttft"]  # but TTFT still ran after GPQA crashed
    assert "gpqa mock failed: PermissionError" in r.output and "continuing" in r.output
