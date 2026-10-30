"""Remove the containers a benchmark leaves behind.

Terminal-Bench and SWE-bench (via Harbor), GAIA / BrowseComp (Inspect sandboxes) and Aider
start containers of their own. They normally remove them, but not always: a task that
errors or is interrupted can leave its environment behind, with its network.

``benchlab run`` takes a snapshot of the containers before each such benchmark and, when the
benchmark ends (finished, failed or stopped), removes the containers that appeared since
*and* are recognisably a benchmark's own, plus their compose networks and volumes. Nothing
else is touched: not your own containers, not ones started meanwhile, not BenchLab's panel
or socket proxies. While another BenchLab run is going, its task containers can't be told
apart from ours, so cleanup waits; ``benchlab cleanup`` finishes the job when it's idle.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

from benchlab.containers import docker_env

# compose projects / images that only benchmarks create
HARNESS_PROJECT = re.compile(r"(__env$)|(^inspect-)")  # Harbor trials, Inspect sandboxes
HARNESS_IMAGE = re.compile(r"^benchlab-aider:")  # Aider's exercise containers
RUN_CONTAINER = re.compile(r"^benchlab-job-")  # BenchLab's own run containers


def _docker(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, env=docker_env()
    )


def available() -> bool:
    try:
        return _docker("info", "--format", "{{.ID}}", timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass
class Container:
    id: str
    name: str
    image: str
    project: str
    running: bool

    @property
    def from_benchmark(self) -> bool:
        return bool(HARNESS_PROJECT.search(self.project) or HARNESS_IMAGE.search(self.image))


def containers() -> list[Container]:
    fmt = '{"id":"{{.ID}}","name":"{{.Names}}","image":"{{.Image}}","state":"{{.State}}",' + (
        '"project":"{{.Label "com.docker.compose.project"}}"}'
    )
    out = _docker("ps", "-a", "--no-trunc", "--format", fmt)
    if out.returncode != 0:
        return []
    found = []
    for line in out.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        found.append(
            Container(d["id"], d["name"], d["image"], d["project"], d["state"] == "running")
        )
    return found


def snapshot() -> set[str] | None:
    """Ids of every container now; None when Docker isn't reachable (nothing to clean)."""
    if not available():
        return None
    return {c.id for c in containers()}


def other_runs_active(own_name: str | None = None) -> bool:
    """Another BenchLab run is going: a run container other than ours, or, on the host,
    another `benchlab run` process."""
    own_name = own_name or os.environ.get("BENCHLAB_ADVERTISE_HOST")
    for c in containers():
        if c.running and RUN_CONTAINER.match(c.name) and c.name != own_name:
            return True
    if own_name:  # inside a run container: only containers count
        return False
    try:
        out = subprocess.run(
            ["pgrep", "-f", r"benchlab(\.cli)? run "], capture_output=True, text=True
        )
    except OSError:
        return False
    pids = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    return bool(pids - {os.getpid(), os.getppid()})


@dataclass
class Removed:
    containers: list[str] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    skipped: str = ""

    def __bool__(self) -> bool:
        return bool(self.containers or self.networks or self.volumes)

    def summary(self) -> str:
        parts = []
        if self.containers:
            parts.append(f"{len(self.containers)} container(s): {', '.join(self.containers)}")
        if self.networks:
            parts.append(f"{len(self.networks)} network(s)")
        if self.volumes:
            parts.append(f"{len(self.volumes)} volume(s)")
        return "; ".join(parts)


def _remove(targets: list[Container]) -> Removed:
    done = Removed()
    projects = sorted({c.project for c in targets if c.project})
    for c in targets:
        if _docker("rm", "-f", "-v", c.id).returncode == 0:
            done.containers.append(c.name)
    for p in projects:  # the compose networks and volumes those containers came with
        label = f"label=com.docker.compose.project={p}"
        for kind, bucket in (("network", done.networks), ("volume", done.volumes)):
            ls = _docker(kind, "ls", "-q", "--filter", label)
            for name in ls.stdout.split():
                if _docker(kind, "rm", name).returncode == 0:
                    bucket.append(name)
    return done


def after_benchmark(before: set[str] | None) -> Removed:
    """Remove what a benchmark left behind since ``before`` (its snapshot)."""
    if before is None:
        return Removed()
    if other_runs_active():
        return Removed(skipped="another BenchLab run is active; `benchlab cleanup` once idle")
    left = [c for c in containers() if c.id not in before and c.from_benchmark]
    return _remove(left)


def idle_cleanup(dry_run: bool = False) -> tuple[list[Container], Removed]:
    """Everything a benchmark could have left, from any run: benchmark containers and
    finished BenchLab run containers. Only when no BenchLab run is active."""
    if other_runs_active():
        return [], Removed(skipped="a BenchLab run is active; try again when it has finished")
    targets = [
        c
        for c in containers()
        if c.from_benchmark or (RUN_CONTAINER.match(c.name) and not c.running)
    ]
    if dry_run:
        return targets, Removed()
    return targets, _remove(targets)
