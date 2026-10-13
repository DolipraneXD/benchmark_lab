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
    r = runner.invoke(app, ["run", "-m", "all", "--bench", "aider", "-c", str(models_yaml)])
    assert r.exit_code == 0, r.output
    assert "not integrated yet" in r.output
    r = runner.invoke(app, ["run", "-m", "mock", "-c", str(models_yaml)])
    assert r.exit_code == 2  # no selector


def test_proxy_config_cli(models_yaml, tmp_path):
    out = tmp_path / "cfg.yaml"
    r = runner.invoke(app, ["proxy-config", "-c", str(models_yaml), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and "os.environ/MOCK_API_KEY" in out.read_text()
