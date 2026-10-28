"""benchlab ui: access guard, model editing, connection test, plan, and background jobs
that run the real `benchlab run` against the mock endpoint (start, finish, stop, resume)."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from benchlab.ui import jobs as J
from benchlab.ui.app import create_app

TOKEN = "test-token-123"
BASE = "http://127.0.0.1:8765"


@pytest.fixture
def client(models_yaml):
    c = TestClient(create_app(TOKEN, root=models_yaml.parent, port=8765), base_url=BASE)
    r = c.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303 and "benchlab_ui" in r.cookies
    return c


def _wait(rd: Path, job_id: str, want: set[str], timeout: float = 90) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = J.status(rd, J.load(rd, job_id))
        if st in want:
            return st
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} still {st} after {timeout}s")


def _job_id(r) -> str:
    return r.headers["location"].split("/jobs/")[1].split("?")[0]


def test_guard(models_yaml):
    app = create_app(TOKEN, root=models_yaml.parent, port=8765)
    anon = TestClient(app, base_url=BASE)
    assert anon.get("/").status_code == 403
    assert anon.get("/?token=nope", follow_redirects=False).status_code == 403
    assert TestClient(app, base_url="http://evil.example:8765").get("/").status_code == 403
    ok = TestClient(app, base_url=BASE)
    ok.get(f"/?token={TOKEN}")
    r = ok.post("/models/mock/delete", headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    assert "id: mock" in models_yaml.read_text()


def test_pages_render(client):
    for path in (
        "/",
        "/models",
        "/models/new",
        "/models/mock/edit",
        "/run",
        "/jobs",
        "/runs",
        "/import",
    ):
        r = client.get(path)
        assert r.status_code == 200, path
    assert "Mock OK" in client.get("/models").text


def test_model_add_edit_delete_keeps_comments(client, models_yaml):
    models_yaml.write_text("# top comment kept\n" + models_yaml.read_text())
    form = {
        "original_id": "",
        "id": "newm",
        "name": "New model",
        "provider": "openai_compatible",
        "kind": "chat",
        "base_url": "http://127.0.0.1:1/v1",
        "model": "m",
        "api_key_env": "MOCK_API_KEY",
        "max_concurrency": "2",
        "price_in_per_mtok": "0.5",
        "price_out_per_mtok": "1.5",
        "suites": "core",
        "extra_params": "{temperature: 0.2}",
    }
    r = client.post("/models/save", data=form, follow_redirects=False)
    assert r.status_code == 303, r.text
    text = models_yaml.read_text()
    assert text.startswith("# top comment kept") and "id: newm" in text
    assert "extra_params: {temperature: 0.2}" in text
    assert (models_yaml.parent / "models.yaml.bak").exists()

    r = client.post("/models/save", data={**form, "original_id": "newm", "name": "Renamed"})
    assert "Renamed" in models_yaml.read_text()
    bad = client.post("/models/save", data={**form, "original_id": "newm", "base_url": ""})
    assert "needs base_url" in bad.text and "Renamed" in models_yaml.read_text()

    client.post("/models/newm/delete")
    assert "id: newm" not in models_yaml.read_text()
    r = client.post("/models/claude/delete", follow_redirects=False)  # the baseline
    assert "err=" in r.headers["location"] and "id: claude" in models_yaml.read_text()


def test_model_check(client):
    r = client.post("/models/mock/check")
    assert r.status_code == 200 and "works" in r.text
    assert "works" in client.get("/models").text  # remembered on the models page


def test_plan(client):
    r = client.post("/run/plan", data={"models": ["mock"], "benches": ["ttft"], "smoke": "on"})
    assert "Ready" in r.text and "mock" in r.text
    r = client.post("/run/plan", data={"models": [], "benches": ["ttft"]})
    assert "pick at least one model" in r.text


def test_job_runs_like_the_cli(client, models_yaml):
    rd = models_yaml.parent / "runs"
    r = client.post(
        "/run/start",
        data={"models": ["mock"], "benches": ["ttft"], "smoke": "on", "budget": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    jid = _job_id(r)
    job = J.load(rd, jid)
    assert job.argv[3:] == [
        "run",
        "-c",
        str(models_yaml),
        "-m",
        "mock",
        "-b",
        "ttft",
        "--smoke",
        "--budget-usd",
        "1.0",
    ]
    assert _wait(rd, jid, {"done", "failed", "lost"}) == "done", J.log_tail(rd, job)
    run = next(rd.glob("*_mock_ttft"))
    res = json.loads((run / "result.json").read_text())
    assert res["bench"] == "ttft" and res["model_id"] == "mock"
    page = client.get(f"/fragments/jobs/{jid}").text
    assert "Finished" in page and run.name in page
    assert client.get(f"/runs/{run.name}").status_code == 200


def test_job_stop_and_resume(client, models_yaml):
    rd = models_yaml.parent / "runs"
    r = client.post(
        "/run/start",
        data={"models": ["slow"], "benches": ["ttft"], "smoke": "on"},
        follow_redirects=False,
    )
    jid = _job_id(r)
    time.sleep(4)  # inside the slow requests
    dup = client.post(
        "/run/start",
        data={"models": ["slow"], "benches": ["ttft"], "smoke": "on"},
        follow_redirects=False,
    )
    assert "already+running" in dup.headers["location"]
    client.post(f"/jobs/{jid}/stop")
    assert _wait(rd, jid, {"stopped", "done", "failed"}) == "stopped"
    r = client.post(f"/jobs/{jid}/resume", follow_redirects=False)
    jid2 = _job_id(r)
    assert jid2 != jid
    assert _wait(rd, jid2, {"done", "failed", "lost"}) == "done"
    assert len(list(rd.glob("*_slow_ttft"))) == 1  # resumed into the same run folder


def test_import_upload(client, models_yaml):
    csv_text = (
        '"Test ID","Plugin","Plugin ID","Strategy","Target","Prompt","Response","Pass","Score","Reason","Timestamp"\n'
        '"1","Gender Bias","bias:gender","basic","t","p","ok","Pass (1)","1","All assertions passed","2026-10-27T15:43:38.506Z"\n'
        '"2","Gender Bias","bias:gender","basic","t","p","","Fail","","","2026-10-27T15:43:38.506Z"\n'
    )
    r = client.post(
        "/import", data={"model": "mock"}, files={"file": ("lab.csv", csv_text, "text/csv")}
    )
    assert r.status_code == 200 and "100.0" in r.text
    run = next((models_yaml.parent / "runs").glob("*_mock_redteam"))
    assert (run / "raw" / "lab.csv").exists()


def test_job_env_puts_venv_first(monkeypatch):
    import os
    import sys

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    first = J.job_env()["PATH"].split(os.pathsep)[0]
    assert first == str(Path(sys.executable).parent)


def test_delete_explains_roles_and_roles_can_move(client, models_yaml):
    text = models_yaml.read_text().replace(
        "  tau2_user_model: claude", "  tau2_user_model: claude\n  grader_model: mock"
    )
    models_yaml.write_text(text)
    models_yaml.chmod(0o664)
    r = client.post("/models/mock/delete", follow_redirects=False)
    assert "BrowseComp+grader" in r.headers["location"] and "id: mock" in models_yaml.read_text()
    r = client.post(
        "/models/roles",
        data={"baseline": "claude", "grader": "", "user_model": "claude"},
        follow_redirects=False,
    )
    assert "Roles+saved" in r.headers["location"]
    assert "grader_model" not in models_yaml.read_text()
    client.post("/models/mock/delete")
    assert "id: mock" not in models_yaml.read_text()
    assert models_yaml.stat().st_mode & 0o777 == 0o664  # saving keeps permissions
    r = client.post(
        "/models/roles",
        data={"baseline": "nope", "grader": "", "user_model": "claude"},
        follow_redirects=False,
    )
    assert "Roles+not+saved" in r.headers["location"]
