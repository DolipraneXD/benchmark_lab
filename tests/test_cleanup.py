"""Cleanup after a benchmark: only containers that appeared during it and are a benchmark's
own (Harbor task environments, Inspect sandboxes, Aider exercises) are removed, with their
compose networks and volumes. Everything else stays."""

import json
import subprocess

from benchlab import cleanup


class FakeDocker:
    def __init__(self, containers):
        self.containers = {c["id"]: c for c in containers}
        self.networks = {"caffe__x__env_default": "caffe__x__env", "benchlab": "benchlab"}
        self.volumes = {"caffe__x__env_data": "caffe__x__env", "jenkins_jenkins-data": "jenkins"}
        self.removed = []

    def __call__(self, *args, timeout=60):
        ok = lambda out="": subprocess.CompletedProcess(args, 0, out, "")  # noqa: E731
        if args[0] == "info":
            return ok("id")
        if args[0] == "ps":
            rows = [
                json.dumps(
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "image": c["image"],
                        "state": c.get("state", "exited"),
                        "project": c.get("project", ""),
                    }
                )
                for c in self.containers.values()
            ]
            return ok("\n".join(rows))
        if args[0] == "rm":
            self.removed.append(self.containers.pop(args[-1])["name"])
            return ok()
        if args[0] in ("network", "volume") and args[1] == "ls":
            pool = self.networks if args[0] == "network" else self.volumes
            project = args[-1].split("=", 2)[-1]
            return ok("\n".join(n for n, p in pool.items() if p == project))
        if args[0] in ("network", "volume") and args[1] == "rm":
            pool = self.networks if args[0] == "network" else self.volumes
            pool.pop(args[2])
            self.removed.append(args[2])
            return ok()
        raise AssertionError(args)


def test_only_new_benchmark_containers_are_removed(monkeypatch):
    docker = FakeDocker(
        [
            {
                "id": "p1",
                "name": "benchlab-docker-runs",
                "image": "tecnativa/docker-socket-proxy",
                "state": "running",
                "project": "benchlab",
            },
            {"id": "u1", "name": "my-postgres", "image": "postgres:16", "state": "running"},
        ]
    )
    monkeypatch.setattr(cleanup, "_docker", docker)
    monkeypatch.setattr(cleanup, "other_runs_active", lambda own_name=None: False)
    before = cleanup.snapshot()
    # during the benchmark: a Harbor task env, an Inspect sandbox, an Aider exercise,
    # and something the user started meanwhile
    docker.containers.update(
        {
            "h1": {
                "id": "h1",
                "name": "caffe__x__env-main-1",
                "image": "alexgshaw/caffe",
                "project": "caffe__x__env",
            },
            "i1": {
                "id": "i1",
                "name": "inspect-gaia-abc-default-1",
                "image": "python:3.12",
                "project": "inspect-gaia-abc",
            },
            "a1": {"id": "a1", "name": "brave_aider", "image": "benchlab-aider:v0.86.2"},
            "u2": {"id": "u2", "name": "my-redis", "image": "redis:7", "state": "running"},
        }
    )
    removed = cleanup.after_benchmark(before)
    assert sorted(removed.containers) == [
        "brave_aider",
        "caffe__x__env-main-1",
        "inspect-gaia-abc-default-1",
    ]
    assert removed.networks == ["caffe__x__env_default"] and removed.volumes == [
        "caffe__x__env_data"
    ]
    left = {c["name"] for c in docker.containers.values()}
    assert left == {"benchlab-docker-runs", "my-postgres", "my-redis"}  # yours and ours stay
    assert "jenkins_jenkins-data" in docker.volumes and "benchlab" in docker.networks


def test_cleanup_waits_while_another_run_is_active(monkeypatch):
    docker = FakeDocker([])
    monkeypatch.setattr(cleanup, "_docker", docker)
    before = cleanup.snapshot()
    docker.containers["h1"] = {
        "id": "h1",
        "name": "t__x__env-main-1",
        "image": "x",
        "project": "t__x__env",
    }
    docker.containers["j2"] = {
        "id": "j2",
        "name": "benchlab-job-other-0",
        "image": "benchlab-core",
        "state": "running",
    }
    monkeypatch.setenv("BENCHLAB_ADVERTISE_HOST", "benchlab-job-mine-0")
    removed = cleanup.after_benchmark(before)
    assert not removed and "another BenchLab run is active" in removed.skipped
    assert "h1" in docker.containers  # might be the other run's task: left for later


def test_idle_cleanup_and_no_docker(monkeypatch):
    docker = FakeDocker(
        [
            {"id": "h1", "name": "t__x__env-main-1", "image": "x", "project": "t__x__env"},
            {"id": "j1", "name": "benchlab-job-old-0", "image": "benchlab-core", "state": "exited"},
            {"id": "u1", "name": "rancher", "image": "rancher/rancher", "state": "exited"},
        ]
    )
    monkeypatch.setattr(cleanup, "_docker", docker)
    monkeypatch.setattr(cleanup, "other_runs_active", lambda own_name=None: False)
    targets, _ = cleanup.idle_cleanup(dry_run=True)
    assert sorted(c.name for c in targets) == ["benchlab-job-old-0", "t__x__env-main-1"]
    assert len(docker.containers) == 3  # dry run removes nothing
    cleanup.idle_cleanup()
    assert set(docker.containers) == {"u1"}

    monkeypatch.setattr(cleanup, "available", lambda: False)
    assert cleanup.snapshot() is None and not cleanup.after_benchmark(None)
