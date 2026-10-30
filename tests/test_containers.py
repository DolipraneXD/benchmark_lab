"""Runs as containers: the `docker run` a run becomes, key handling, splitting a job by
image family, and the launcher running container steps one after another."""

import json
import sys
from pathlib import Path

from benchlab import containers
from benchlab.config import load_config
from benchlab.ui import launcher


def _argv(cfg, root, benches=("gpqa",), models=("mock",), **kw):
    return containers.run_argv(
        cfg,
        name="benchlab-job-x-0",
        family=kw.pop("family", "core"),
        cli_args=[
            "run",
            "-c",
            str(root / "models.yaml"),
            "-m",
            ",".join(models),
            "-b",
            ",".join(benches),
        ],
        root=root,
        env_names=containers.key_vars(cfg, list(models), list(benches)),
        needs_docker=containers.needs_docker(list(benches)),
        user="1000:1000",
        **kw,
    )


def _flag(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


def test_run_argv_limits_paths_and_image(models_yaml, monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_API_KEY", "sk-mock-secret-key-1234567890")
    monkeypatch.delenv("BENCHLAB_WORKSPACE", raising=False)
    cfg = load_config(models_yaml)
    root = models_yaml.parent
    argv = _argv(cfg, root)
    assert argv[:4] == ["docker", "run", "--rm", "--init"]
    assert "--cpus" in argv and _flag(argv, "--memory") == ["4g"]  # core default
    ws = str((root / "workspace").resolve())
    assert f"{ws}:{ws}" in _flag(argv, "-v")  # path parity
    assert f"{root / 'models.yaml'}:{root / 'models.yaml'}:ro" in _flag(argv, "-v")
    assert _flag(argv, "--workdir") == [str(root.resolve())]
    assert _flag(argv, "--network") == ["benchlab"] and _flag(argv, "--user") == ["1000:1000"]
    i = argv.index("benchlab-core:latest")
    assert argv[i + 1 :][:2] == ["benchlab", "run"]
    env = _flag(argv, "-e")
    assert "MOCK_API_KEY" in env and "BENCHLAB_ADVERTISE_HOST=benchlab-job-x-0" in env
    assert "sk-mock-secret" not in " ".join(argv)  # keys by name only
    assert not any(e.startswith("DOCKER_HOST=") for e in env)  # GPQA needs no Docker


def test_docker_access_only_for_benchmarks_that_need_it(models_yaml):
    cfg = load_config(models_yaml)
    argv = _argv(cfg, models_yaml.parent, benches=("tb",))
    assert f"DOCKER_HOST={containers.RUNS_DOCKER_HOST}" in _flag(argv, "-e")
    argv = _argv(cfg, models_yaml.parent, benches=("lcb",), family="code")
    assert "benchlab-code:latest" in argv and _flag(argv, "--memory") == ["8g"]


def test_key_vars_cover_helpers_and_tokens(models_yaml):
    cfg = load_config(models_yaml)  # baseline and τ² customer: claude (ANTHROPIC_API_KEY)
    assert containers.key_vars(cfg, ["mock"], ["gpqa"]) == ["LITELLM_MASTER_KEY", "MOCK_API_KEY"]
    assert "ANTHROPIC_API_KEY" in containers.key_vars(cfg, ["mock"], ["t2air"])
    assert "ANTHROPIC_API_KEY" in containers.key_vars(cfg, ["mock"], ["apptasks"])  # grader
    assert {"HF_TOKEN", "TAVILY_API_KEY"} <= set(containers.key_vars(cfg, ["mock"], ["bcomp"]))


def test_split_by_family_keeps_registry_order():
    pairs = [["m", "lcb"], ["m", "gpqa"], ["m", "beir"], ["m", "t2air"], ["n", "gpqa"]]
    assert containers.split_by_family(pairs) == [
        ("core", ["m", "n"], ["gpqa", "t2air"]),
        ("code", ["m"], ["lcb"]),
        ("retrieval", ["m"], ["beir"]),
    ]


def test_executor_setting_and_env(models_yaml, monkeypatch):
    cfg = load_config(models_yaml)
    assert containers.executor(cfg) == "local"
    monkeypatch.setenv("BENCHLAB_EXECUTOR", "docker")
    assert containers.executor(cfg) == "docker"


def test_launcher_runs_steps_and_stops_at_first_stop(tmp_path):
    py = sys.executable
    job = tmp_path / "j.json"
    job.write_text(
        json.dumps(
            {
                "argv": ["unused"],
                "steps": [
                    [py, "-c", "print('one')"],
                    [py, "-c", "import sys; print('two'); sys.exit(1)"],
                    [py, "-c", "print('three')"],
                ],
            }
        )
    )
    launcher.main(str(job))
    assert job.with_suffix(".exit").read_text() == "1"  # the first failure decides
    assert job.with_suffix(".log").read_text().split() == ["one", "two", "three"]

    job.write_text(
        json.dumps(
            {
                "argv": [],
                "steps": [[py, "-c", "import sys; sys.exit(130)"], [py, "-c", "print('after')"]],
            }
        )
    )
    job.with_suffix(".log").unlink()
    launcher.main(str(job))
    assert job.with_suffix(".exit").read_text() == "130"  # stopped: later families don't start
    assert "after" not in Path(job.with_suffix(".log")).read_text()


def test_stop_reaches_a_later_step(tmp_path):
    """The launcher ignores SIGINT; the commands it starts must not inherit that, or a
    Stop would never reach a job's second container."""
    import os
    import signal
    import subprocess
    import time

    py = sys.executable
    waiter = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGINT, lambda *a: (print('got it', flush=True), sys.exit(130)))\n"
        "print('waiting', flush=True)\n"
        "time.sleep(60)\n"
    )
    job = tmp_path / "j.json"
    job.write_text(
        json.dumps({"argv": [], "steps": [[py, "-c", "print('first')"], [py, "-c", waiter]]})
    )
    p = subprocess.Popen([py, "-m", "benchlab.ui.launcher", str(job)], start_new_session=True)
    log = job.with_suffix(".log")
    for _ in range(100):  # until the second step is running
        if log.exists() and "waiting" in log.read_text():
            break
        time.sleep(0.1)
    os.killpg(p.pid, signal.SIGINT)  # what the panel's Stop does
    p.wait(timeout=20)
    assert "got it" in log.read_text()
    assert job.with_suffix(".exit").read_text() == "130"
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN  # pytest itself untouched


def test_docker_env_picks_benchlabs_engine_only_when_asked():
    assert "DOCKER_CONTEXT" not in containers.docker_env({"PATH": "/bin"})
    env = containers.docker_env({"BENCHLAB_DOCKER_CONTEXT": "default"})
    assert env["DOCKER_CONTEXT"] == "default"
    env = containers.docker_env(
        {"BENCHLAB_DOCKER_CONTEXT": "default", "DOCKER_HOST": "tcp://p:2375"}
    )
    assert "DOCKER_CONTEXT" not in env  # an explicit DOCKER_HOST (the socket proxy) wins


def test_a_benchmark_can_raise_its_containers_limits(models_yaml):
    cfg = load_config(models_yaml)
    assert containers.limits_for(cfg, "code", ["aider"]) == {"cpus": 4, "memory": "8g"}
    assert containers.limits_for(cfg, "code", ["aider", "lcb"])["memory"] == "14g"
    argv = _argv(cfg, models_yaml.parent, benches=("lcb",), family="code")
    assert _flag(argv, "--memory") == ["8g"]  # _argv doesn't pass benches: family default
    argv = containers.run_argv(
        cfg,
        name="n",
        family="code",
        cli_args=["run"],
        root=models_yaml.parent,
        env_names=[],
        needs_docker=False,
        benches=["lcb"],
    )
    assert _flag(argv, "--memory") == ["14g"]
