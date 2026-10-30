"""Runs as containers: the ``docker run`` command that executes one ``benchlab run`` in a
container of its own, from the image of the benchmark family.

The command inside is the same ``benchlab run …`` a terminal would use. What the
container adds:
* the image for the family (core, code, retrieval, serving) and its CPU / memory limits
* path parity: the project folder's ``data/``, configs and workspace are mounted at the
  same absolute paths as on the host, so paths a harness hands to Docker stay valid
* only the key variables the run needs, passed by name (``-e NAME``): the value comes
  from the launcher's environment and never appears on a command line
* Docker access (through the runs socket proxy) only for benchmarks that start their own
  containers (Terminal-Bench, SWE-bench, GAIA, BrowseComp, Aider)
"""

from __future__ import annotations

import os
from pathlib import Path

from benchlab import paths, registry
from benchlab.config import BenchConfig

# where run containers reach Docker (docker/compose.yaml: the `docker-runs` proxy)
RUNS_DOCKER_HOST = "tcp://benchlab-docker-runs:2375"
# variables a benchmark needs besides the models' keys
BENCH_ENV = {"gaia": ("HF_TOKEN",), "bcomp": ("TAVILY_API_KEY", "HF_TOKEN")}
ALWAYS_ENV = ("LITELLM_MASTER_KEY",)
AWS_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION_NAME")


def docker_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for Docker commands BenchLab runs on the host. BENCHLAB_DOCKER_CONTEXT
    picks the engine for BenchLab only (e.g. `default`, the native engine, when the
    shell's current context is Docker Desktop); an explicit DOCKER_HOST wins."""
    env = dict(os.environ if env is None else env)
    ctx = env.get("BENCHLAB_DOCKER_CONTEXT")
    if ctx and not env.get("DOCKER_HOST"):
        env["DOCKER_CONTEXT"] = ctx
    return env


def executor(cfg: BenchConfig) -> str:
    return os.environ.get("BENCHLAB_EXECUTOR") or cfg.settings.executor


def image(cfg: BenchConfig, family: str) -> str:
    s = cfg.settings
    return f"{s.docker_image_prefix}-{family}:{s.docker_image_tag}"


def container_name(job_id: str, step: int) -> str:
    return f"benchlab-job-{job_id}-{step}"


def key_vars(cfg: BenchConfig, model_ids: list[str], benches: list[str]) -> list[str]:
    """Names of the environment variables one run needs: its models' keys, the keys of
    the helper models its benchmarks use (τ² customer, grader), and benchmark tokens."""
    ids = set(model_ids)
    if any(b.startswith("t2") for b in benches):
        ids.add(cfg.settings.tau2_user_model)
    if any(b in ("bcomp", "apptasks") for b in benches):
        ids.add(cfg.settings.grader_model or cfg.settings.baseline_model)
    names = set(ALWAYS_ENV)
    for mid in ids:
        try:
            m = cfg.get(mid)
        except KeyError:
            continue
        if m.api_key_env:
            names.add(m.api_key_env)
        if m.provider == "bedrock":
            names.update(AWS_ENV)
    for b in benches:
        names.update(BENCH_ENV.get(b, ()))
    return sorted(names)


def split_by_family(pairs: list[list[str]]) -> list[tuple[str, list[str], list[str]]]:
    """[(family, model ids, bench keys)] in registry order, one container each."""
    groups: dict[str, tuple[list[str], list[str]]] = {}
    order = list(registry.BENCHES)
    for m, b in sorted(pairs, key=lambda p: order.index(p[1]) if p[1] in order else 999):
        fam = registry.get(b).family
        ms, bs = groups.setdefault(fam, ([], []))
        if m not in ms:
            ms.append(m)
        if b not in bs:
            bs.append(b)
    return [(f, ms, bs) for f, (ms, bs) in groups.items()]


def run_argv(
    cfg: BenchConfig,
    *,
    name: str,
    family: str,
    cli_args: list[str],
    root: Path,
    env_names: list[str],
    needs_docker: bool,
    user: str | None = None,
    benches: list[str] | None = None,
) -> list[str]:
    """``docker run`` for one ``benchlab <cli_args>`` in a container of ``family``."""
    root = root.resolve()
    ws = paths.workspace()
    ws = (ws if ws.is_absolute() else root / ws).resolve()
    data = (root / paths.DATA).resolve()
    home = ws / ".home"
    home.mkdir(parents=True, exist_ok=True)
    lim = limits_for(cfg, family, benches or [])
    argv = [
        "docker",
        "run",
        "--rm",
        "--init",  # a real PID 1 forwards Stop's SIGINT to benchlab, which saves progress
        "--name",
        name,
        "--network",
        cfg.settings.docker_network,
        "--workdir",
        str(root),
        "--tmpfs",
        "/tmp:exec",
        "-e",
        f"HOME={home}",  # dataset and tool caches survive between runs
    ]
    if user:
        argv += ["--user", user]  # files in the workspace stay owned by the host user
    if lim.get("cpus"):
        argv += ["--cpus", str(lim["cpus"])]
    if lim.get("memory"):
        argv += ["--memory", str(lim["memory"])]
    argv += ["-v", f"{ws}:{ws}", "-v", f"{data}:{data}"]
    for c in sorted(root.glob("models*.yaml")):
        argv += ["-v", f"{c}:{c}:ro"]
    argv += [
        "-e",
        f"BENCHLAB_WORKSPACE={ws}",
        "-e",
        f"BENCHLAB_ADVERTISE_HOST={name}",  # harness containers reach the proxy by this name
        "-e",
        f"BENCHLAB_DOCKER_NETWORK={cfg.settings.docker_network}",
        "-e",
        "BENCHLAB_EXECUTOR=local",
    ]
    if needs_docker:
        argv += [
            "-e",
            f"DOCKER_HOST={os.environ.get('BENCHLAB_RUNS_DOCKER_HOST', RUNS_DOCKER_HOST)}",
        ]
    for n in env_names:
        argv += ["-e", n]  # value from the launcher's environment, never on the command line
    argv += [image(cfg, family), "benchlab", *cli_args]
    return argv


def _gib(mem: str | int | float) -> float:
    s = str(mem).strip().lower().rstrip("b")
    unit = {"k": 1 / 1024**2, "m": 1 / 1024, "g": 1, "t": 1024}.get(s[-1:], None)
    return float(s[:-1]) * unit if unit else float(s) / 1024**3


def limits_for(cfg: BenchConfig, family: str, benches: list[str]) -> dict:
    """The family's limits, raised by any benchmark of the container that has its own
    entry (e.g. LiveCodeBench unpacks every problem's tests in memory)."""
    lim = dict(cfg.settings.container_limits.get(family, {}))
    for b in benches:
        own = cfg.settings.container_limits.get(b) or {}
        if own.get("cpus") and float(own["cpus"]) > float(lim.get("cpus") or 0):
            lim["cpus"] = own["cpus"]
        if own.get("memory") and (
            not lim.get("memory") or _gib(own["memory"]) > _gib(lim["memory"])
        ):
            lim["memory"] = own["memory"]
    return lim


def host_user() -> str | None:
    """uid:gid that run containers use, so the workspace stays owned by the host user."""
    if os.environ.get("BENCHLAB_RUN_USER"):
        return os.environ["BENCHLAB_RUN_USER"]
    if hasattr(os, "getuid"):
        return f"{os.getuid()}:{os.getgid()}"
    return None


def needs_docker(benches: list[str]) -> bool:
    return any(registry.get(b).needs_docker for b in benches)
