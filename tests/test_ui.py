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
        "key_mode": "var:MOCK_API_KEY",
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
    assert (models_yaml.parent / "workspace" / "backups" / "models.yaml.bak").exists()

    r = client.post("/models/save", data={**form, "original_id": "newm", "name": "Renamed"})
    assert "Renamed" in models_yaml.read_text()
    r = client.post(
        "/models/save",
        data={
            **form,
            "original_id": "newm",
            "name": "Renamed",
            "base_url": "http://h:1/v1/chat/completions",
        },
    )
    assert "base_url: http://h:1/v1\n" in models_yaml.read_text() and "was removed" in r.text
    bad = client.post("/models/save", data={**form, "original_id": "newm", "base_url": ""})
    assert "needs base_url" in bad.text and "Renamed" in models_yaml.read_text()

    client.post("/models/newm/delete")
    assert "id: newm" not in models_yaml.read_text()


def test_each_model_gets_its_own_key(client, models_yaml):
    import os
    import stat

    env = models_yaml.parent / ".env"
    key = "sk-newm-own-key-1234567890abcd"
    form = {
        "original_id": "",
        "id": "newm",
        "name": "New model",
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:1/v1",
        "model": "m",
        "key_mode": "own",
        "api_key": key,
    }
    r = client.post("/models/save", data=form)
    assert "Saved newm" in r.text and key not in r.text
    text = models_yaml.read_text()
    assert "api_key_env: BENCHLAB_KEY_NEWM" in text and key not in text
    assert f"BENCHLAB_KEY_NEWM={key}\n" in env.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert os.environ["BENCHLAB_KEY_NEWM"] == key

    page = client.get("/models/newm/edit").text
    assert "saved ••••abcd" in page and key not in page

    # rename: the key moves to the new id's variable, the old one is dropped
    r = client.post(
        "/models/save", data={**form, "original_id": "newm", "id": "newer", "api_key": ""}
    )
    assert "api_key_env: BENCHLAB_KEY_NEWER" in models_yaml.read_text()
    assert f"BENCHLAB_KEY_NEWER={key}" in env.read_text()
    assert "BENCHLAB_KEY_NEWM" not in env.read_text()

    # share another model's key instead: its own variable goes away
    shared = {
        **form,
        "original_id": "newer",
        "id": "newer",
        "api_key": "",
        "key_mode": "var:MOCK_API_KEY",
    }
    client.post("/models/save", data=shared)
    assert "id: newer" in models_yaml.read_text()
    assert "BENCHLAB_KEY_NEWER" not in env.read_text()

    # refusals keep the file unchanged and never echo the pasted key
    before = models_yaml.read_text()
    for data, why in [
        ({**form, "id": "x1", "api_key": "two words"}, "spaces"),
        ({**form, "id": "x2", "key_mode": "none"}, "No key needed but pasted a key"),
        ({**form, "id": "x3", "key_mode": "var:PATH"}, "Choose which key"),
    ]:
        r = client.post("/models/save", data=data)
        assert why in r.text and "sk-newm" not in r.text, why
    assert models_yaml.read_text() == before

    # deleting a model removes its own key from .env
    client.post("/models/save", data={**form, "id": "gone", "api_key": key})
    assert "BENCHLAB_KEY_GONE" in env.read_text()
    client.post("/models/gone/delete")
    assert "BENCHLAB_KEY_GONE" not in env.read_text()


def test_key_names_are_unique_per_config():
    from benchlab.ui.env_store import key_name

    assert key_name("models.yaml", "ml4") == "BENCHLAB_KEY_ML4"
    assert key_name("/x/models.xkiro.yaml", "ml4") == "BENCHLAB_KEY_XKIRO_ML4"


def test_plan_blocks_pairs_whose_helper_has_no_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")  # claude is the τ² customer in the test config
    r = client.post(
        "/run/plan", data={"models": ["mock", "nokey"], "benches": ["t2air", "ttft"], "smoke": "on"}
    )
    assert "Can't run: missing key" in r.text
    assert "customer (user simulator) is claude, which has no API key" in r.text
    assert "nokey has no API key (DEFINITELY_UNSET_KEY_ENV)" in r.text
    r = client.post("/run/start", data={"models": ["nokey"], "benches": ["ttft"], "smoke": "on"})
    assert "Nothing to run. nokey has no API key" in r.text


def test_plan(client):
    r = client.post("/run/plan", data={"models": ["mock"], "benches": ["ttft"], "smoke": "on"})
    assert "Ready" in r.text and "mock" in r.text and "Estimated time" in r.text
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


def test_api_keys_saved_in_env_safely(models_yaml, monkeypatch):
    import os
    import stat

    env = models_yaml.parent / ".env"
    env.write_text("# my keys\nOTHER_KEY=keep-me\n")
    env.chmod(0o644)
    c = TestClient(create_app(TOKEN, root=models_yaml.parent, port=8765), base_url=BASE)
    c.get(f"/?token={TOKEN}")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600  # locked when the panel starts

    page = c.get("/models").text
    assert "MOCK_API_KEY" in page and "from your shell" in page
    assert "sk-mock-secret-key-1234567890" not in page

    new = "sk-new-panel-key-abcdef9876"
    r = c.post("/keys/save", data={"name": "MOCK_API_KEY", "value": f" {new} "})
    assert r.status_code == 200 and "Saved MOCK_API_KEY" in r.text
    text = env.read_text()
    assert text.startswith("# my keys\nOTHER_KEY=keep-me\n") and f"MOCK_API_KEY={new}\n" in text
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert os.environ["MOCK_API_KEY"] == new  # live for checks and new jobs
    page = c.get("/models").text
    assert "in .env ••••9876" in page and new not in page

    c.post("/keys/save", data={"name": "MOCK_API_KEY", "value": "sk-second-key-000011112222"})
    assert env.read_text().count("MOCK_API_KEY=") == 1  # replaced, not appended

    before = env.read_text()
    for data, why in [
        ({"name": "PATH", "value": "x" * 20}, "not used by any model"),
        ({"name": "MOCK_API_KEY", "value": "two words here"}, "spaces"),
        ({"name": "MOCK_API_KEY", "value": "XKIRO_API_KEY"}, "not the variable name"),
        ({"name": "MOCK_API_KEY", "value": ""}, "empty"),
    ]:
        r = c.post("/keys/save", data=data)
        assert why in r.text, why
    assert env.read_text() == before

    r = c.post("/keys/MOCK_API_KEY/remove")
    assert "Removed MOCK_API_KEY" in r.text
    assert "MOCK_API_KEY" not in env.read_text() and "OTHER_KEY=keep-me" in env.read_text()
    assert "MOCK_API_KEY" not in os.environ


def test_job_page_says_why_a_pair_was_skipped(client, models_yaml):
    rd = models_yaml.parent / "runs"
    job = J.Job(
        id="20261028-150008-a6b7",
        argv=["benchlab", "run"],
        config="models.yaml",
        pairs=[["mock", "t2tel"], ["mock", "gpqa"]],
        started_at=time.time() - 60,
    )
    jpath, log, exit_file = J._paths(rd, job.id)
    jpath.write_text(json.dumps(J.asdict(job)))
    exit_file.write_text("1")
    # what `benchlab run` prints when the τ² user simulator has no key (rich wraps at 80)
    log.write_text(
        "─────────────────── mock x t2tel ───────────────────\n"
        "harness not ready for t2tel: t2tel: user simulator model ml4: model ml4: env var\n"
        "MISTRAL_API_KEY is not set (put it in .env)\n"
    )
    page = client.get(f"/fragments/jobs/{job.id}").text
    assert "Skipped" in page and "env var MISTRAL_API_KEY is not set" in page
    assert "Not started" in page  # gpqa: the job ended before it
    assert "Waiting its turn" not in page


def test_benchmarks_have_hover_descriptions(client):
    from benchlab.bench_about import about
    from benchlab.registry import BENCHES

    assert all(about(k)["text"] for k in BENCHES)  # every benchmark has a description
    page = client.get("/run").text
    assert 'class="bench-tip"' in page
    assert "https://github.com/sierra-research/tau2-bench" in page
    assert "198 graduate-level multiple-choice questions" in page
    plan = client.post("/run/plan", data={"models": ["mock"], "benches": ["ttft"], "smoke": "on"})
    assert "Time to first token" in plan.text


def test_run_page_lists_each_test_with_prompts(client, models_yaml):
    """A finished BFCL run (real Inspect dump from the mock) shows its tests, filters them
    and opens each one on the prompt, the model's answer and the grader's reason."""
    import shutil

    from benchlab.runs import ErrorCounts, NormalizedResult, RunConfig, RunDir

    rd = RunDir.create(models_yaml.parent / "runs", "mock", "bfcl")
    rd.write_config(
        RunConfig(
            model_id="mock",
            bench="bfcl",
            model={},
            request_params={},
            per_model_exceptions={},
            harness={},
            subset={},
            trials=1,
            smoke=True,
        )
    )
    shutil.copy("tests/fixtures/bfcl_smoke_mock.log_dump.json", rd.raw_dir / "log_dump.json")
    rd.write_result(
        NormalizedResult(
            bench="bfcl",
            model_id="mock",
            metrics={"accuracy": 0.0},
            primary_metric="accuracy",
            n_tasks=3,
            trials=1,
            errors=ErrorCounts(),
        )
    )
    page = client.get(f"/runs/{rd.name}").text
    assert "Tests" in page and "Prompts, answers and grader notes" in page
    assert "irrelevance_0" in page and "Calculate the area of a triangle" in page
    assert "Incorrectly produced a function call for irrelevant query" in page  # grader reason
    assert "determine_body_mass_index" in page  # the model's (wrong) tool call
    r = client.post(f"/runs/{rd.name}/tests", data={"status": "pass", "q": "", "page": "1"})
    assert "0 tests match" in r.text
    r = client.post(f"/runs/{rd.name}/tests", data={"status": "all", "q": "triangle", "page": "1"})
    assert "1 test matches" in r.text and "irrelevance_0" in r.text
    assert (rd.path / "samples.json").exists()  # cached for the next view
    results = client.get("/results").text  # the comparison counts the same tests
    assert "0/3 passed" in results and f'href="/runs/{rd.name}"' in results
    r = client.post("/report", follow_redirects=True)
    assert "Exported report.html" in r.text
    export = client.get("/report").text
    assert "Failed tests" in export and "Calculate the area of a triangle" in export
