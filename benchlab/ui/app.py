"""The control panel server (FastAPI + Jinja2 pages, a little polling JS).

Local only, one user: it binds to 127.0.0.1 and every request must carry the
session cookie set from the one-time link ``benchlab ui`` prints, so other
websites open in the browser can't read pages or start runs (cookie is
SameSite=Strict, POSTs also need a same-origin ``Origin``, the Host header is
checked against DNS rebinding). Runs are started as ``benchlab run`` processes
(see jobs.py), so they behave exactly like CLI runs.
"""

from __future__ import annotations

import hmac
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from benchlab import __version__, containers, registry, samples
from benchlab.config import BenchConfig, load_config
from benchlab.providers import clean_base_url
from benchlab.redact import redact
from benchlab.runs import RunDir
from benchlab.ui import env_store, models_store, render
from benchlab.ui import jobs as J
from benchlab.usage import UsageLog

HERE = Path(__file__).parent
COOKIE = "benchlab_ui"
CFG_COOKIE = "benchlab_cfg"
KINDS = ("chat", "embedding", "reranker")
CsvFile = File(...)
NON_PCT_BENCHES = {"ttft", "serving"}  # seconds / tokens per second: no 0-100 scale


def discover_configs(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("models*.yaml"))


def create_app(
    token: str, root: str | Path | None = None, port: int = 8765, hosts: tuple[str, ...] = ()
) -> FastAPI:
    root = Path(root or Path.cwd()).resolve()
    app = FastAPI(title="BenchLab", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    tpl = Jinja2Templates(directory=HERE / "templates")
    render.install(tpl.env)
    # Host header allowlist: this machine, plus the names it is opened at when it runs in a
    # container or on a server (``--allow-host``); stops DNS-rebinding pages
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    for h in hosts:
        allowed_hosts |= {h} if ":" in h else {h, f"{h}:{port}"}
    last_checks: dict[tuple[str, str], dict[str, Any]] = {}
    env_path = root / ".env"
    env_store.lock(env_path)  # keys live here: owner-only, like an ssh key

    # ---- guard: host, session cookie, same-origin POSTs ------------------------------
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if host not in allowed_hosts:
            return PlainTextResponse("bad host", status_code=403)
        q = request.query_params.get("token")
        if q is not None:
            if not hmac.compare_digest(q, token):
                return PlainTextResponse("invalid link", status_code=403)
            r = RedirectResponse(request.url.path, status_code=303)
            r.set_cookie(COOKIE, token, httponly=True, samesite="strict")
            return r
        if not request.url.path.startswith("/static/"):
            if not hmac.compare_digest(request.cookies.get(COOKIE, ""), token):
                return PlainTextResponse(
                    "Open the link `benchlab ui` printed in the terminal.", status_code=403
                )
        if request.method not in ("GET", "HEAD"):
            origin = request.headers.get("origin")
            if origin and origin.split("://", 1)[-1] not in allowed_hosts:
                return PlainTextResponse("cross-origin request refused", status_code=403)
        return await call_next(request)

    # ---- helpers ----------------------------------------------------------------------
    def cfg_name(request: Request) -> str:
        names = discover_configs(root)
        c = request.cookies.get(CFG_COOKIE)
        if c in names:
            return c
        return "models.yaml" if "models.yaml" in names else (names[0] if names else "models.yaml")

    def cfg_path(request: Request) -> Path:
        return root / cfg_name(request)

    def load(request: Request) -> tuple[BenchConfig | None, str | None]:
        try:
            return load_config(cfg_path(request)), None
        except Exception as e:  # noqa: BLE001
            return None, redact(str(e))

    def runs_dir(cfg: BenchConfig) -> Path:
        return root / cfg.settings.runs_dir

    def page(request: Request, template: str, **ctx: Any) -> HTMLResponse:
        cfg = ctx.get("cfg")
        base = {
            "configs": discover_configs(root),
            "baseline_id": cfg.settings.baseline_model if cfg else "",
            "current_cfg": cfg_name(request),
            "version": __version__,
            "flash": request.query_params.get("msg"),
            "flash_err": request.query_params.get("err"),
            "active_jobs": sum(
                1 for j in J.list_jobs(runs_dir(cfg)) if J.status(runs_dir(cfg), j) == "running"
            )
            if cfg
            else 0,
        }
        return tpl.TemplateResponse(request, template, {**base, **ctx})

    def back(url: str, msg: str | None = None, err: str | None = None) -> RedirectResponse:
        from urllib.parse import urlencode

        q = urlencode({k: v for k, v in (("msg", msg), ("err", err)) if v})
        return RedirectResponse(url + (("?" + q) if q else ""), status_code=303)

    def broken(request: Request, error: str) -> HTMLResponse:
        return page(request, "broken.html", error=error)

    # ---- config switch -----------------------------------------------------------------
    @app.post("/config")
    def switch_config(name: str = Form(...)):
        r = back("/")
        if name in discover_configs(root):
            r.set_cookie(CFG_COOKIE, name, httponly=True, samesite="strict")
        return r

    # ---- dashboard ---------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        rd = runs_dir(cfg)
        rows = _run_rows(rd, cfg.settings.baseline_model)
        running = [j for j in J.list_jobs(rd) if J.status(rd, j) == "running"]
        recent = rows[:12]
        return page(
            request,
            "dashboard.html",
            cfg=cfg,
            recent=recent,
            running=running,
            spend=_spend_today(rd),
        )

    @app.get("/fragments/jobs", response_class=HTMLResponse)
    def jobs_fragment(request: Request, limit: int = 10):
        cfg, e = load(request)
        if not cfg:
            return HTMLResponse(f"<p class=err>{e}</p>")
        rd = runs_dir(cfg)
        rows = [(j, J.status(rd, j)) for j in J.list_jobs(rd)[:limit]]
        return tpl.TemplateResponse(request, "_jobs.html", {"rows": rows})

    # ---- models ------------------------------------------------------------------------
    @app.get("/models", response_class=HTMLResponse)
    def models_page(request: Request):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        name = cfg_name(request)
        rows = []
        for m in cfg.models:
            key = (
                "n/a"
                if not m.api_key_env
                else ("set" if os.environ.get(m.api_key_env) else "missing")
            )
            rows.append({"m": m, "key": key, "check": last_checks.get((name, m.id))})
        return page(request, "models.html", cfg=cfg, rows=rows, keys=key_rows(cfg))

    def key_names(cfg: BenchConfig) -> set[str]:
        return {m.api_key_env for m in cfg.models if m.api_key_env and not m.reference}

    def key_rows(cfg: BenchConfig) -> list[dict[str, Any]]:
        """One row per key variable the config uses. Values never leave this function:
        only where the key comes from and its last 4 characters."""
        saved = env_store.in_file(env_path)
        out = []
        for var in sorted(key_names(cfg)):
            live = os.environ.get(var)
            source = "missing" if not live else "env" if saved.get(var) == live else "shell"
            out.append(
                {
                    "var": var,
                    "used_by": [m.id for m in cfg.models if m.api_key_env == var],
                    "source": source,
                    "in_file": var in saved,
                    "tail": env_store.masked(live),
                }
            )
        return out

    @app.post("/keys/save")
    def key_save(request: Request, name: str = Form(...), value: str = Form("")):
        cfg, e = load(request)
        if not cfg:
            return back("/models", err=e)
        try:
            env_store.set_key(env_path, name, value, key_names(cfg))
        except ValueError as ex:
            return back("/models", err=f"Key not saved: {ex}")
        return back("/models", msg=f"Saved {name} in .env. New checks and runs use it now.")

    @app.post("/keys/{name}/remove")
    def key_remove(request: Request, name: str):
        cfg, e = load(request)
        if not cfg:
            return back("/models", err=e)
        try:
            env_store.remove_key(env_path, name, key_names(cfg))
        except ValueError as ex:
            return back("/models", err=f"Key not removed: {ex}")
        return back("/models", msg=f"Removed {name} from .env.")

    def key_choices(request: Request, cfg: BenchConfig | None, model_id: str | None):
        """Options for the model form's key menu, plus what each variable holds now
        (last 4 characters only)."""
        own = env_store.key_name(cfg_name(request), model_id) if model_id else None
        opts = [("own", f"Its own key ({own})" if own else "Its own key, named after the id")]
        for var in sorted(key_names(cfg) - {own}) if cfg else []:
            ids = ", ".join(m.id for m in cfg.models if m.api_key_env == var)
            opts.append((f"var:{var}", f"Same key as {ids} ({var})"))
        opts.append(("none", "No key needed (local or self-hosted endpoint)"))
        saved = {v: env_store.masked(os.environ.get(v)) for v in key_names(cfg)} if cfg else {}
        return {"options": opts, "own": own, "saved": saved}

    def key_mode(request: Request, model_id: str | None, var: str | None) -> str:
        if not var:
            return "none" if model_id else "own"
        if model_id and var == env_store.key_name(cfg_name(request), model_id):
            return "own"
        return f"var:{var}"

    def form_page(request, cfg, model_id, form, err=None):
        return page(
            request,
            "model_form.html",
            cfg=cfg,
            m=model_id,
            form=form,
            err=err,
            keys=key_choices(request, cfg, model_id),
        )

    @app.get("/models/new", response_class=HTMLResponse)
    def model_new(request: Request):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        return form_page(request, cfg, None, {**_form_values({}), "key_mode": "own"})

    @app.get("/models/{model_id}/edit", response_class=HTMLResponse)
    def model_edit(request: Request, model_id: str):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        try:
            raw = models_store.raw_model(cfg_path(request), model_id)
        except KeyError:
            return back("/models", err=f"unknown model {model_id}")
        form = {
            **_form_values(raw),
            "key_mode": key_mode(request, model_id, raw.get("api_key_env")),
        }
        return form_page(request, cfg, model_id, form)

    def drop_unused_own_key(request: Request, var: str | None) -> None:
        """Remove a model's own BENCHLAB_KEY_ variable from .env once no model in this
        config uses it (after a delete, a rename or a switch to another key)."""
        if not var or not var.startswith(env_store.PREFIX):
            return
        cfg, _ = load(request)
        if cfg is not None and var not in key_names(cfg) and var in env_store.in_file(env_path):
            env_store.remove_key(env_path, var, {var})

    @app.post("/models/save")
    async def model_save(request: Request):
        form = dict(await request.form())
        original = form.pop("original_id", "") or None
        pasted = str(form.pop("api_key", "") or "")  # never stored in `form`: it is re-rendered
        mode = str(form.get("key_mode") or "own")
        cfg, e = load(request)
        old_var = None
        if cfg is not None and original:
            old_var = next((m.api_key_env for m in cfg.models if m.id == original), None)
        try:
            data = _parse_model_form(form)
            trimmed = None
            if data.get("base_url"):
                data["base_url"], trimmed = clean_base_url(data["base_url"])
            if mode == "own":
                var = env_store.key_name(cfg_name(request), data["id"])
            elif mode == "none":
                var = None
                if pasted.strip():
                    raise ValueError("You chose No key needed but pasted a key. Pick a key option.")
            elif mode.startswith("var:") and cfg is not None and mode[4:] in key_names(cfg):
                var = mode[4:]
            else:
                raise ValueError("Choose which key this model uses.")
            value = env_store.clean_value(pasted) if pasted.strip() else None
            data["api_key_env"] = var
            models_store.save_model(cfg_path(request), data, original)
        except ValueError as ex:
            return form_page(request, cfg, original, form, err=redact(str(ex)))
        if value is None and var and old_var and old_var != var and os.environ.get(old_var):
            if not os.environ.get(var) and mode == "own":
                value = os.environ[old_var]  # renamed, or moved to its own key: carry it over
        if value is not None and var:
            env_store.set_key(env_path, var, value, {var})
        if old_var != var:
            drop_unused_own_key(request, old_var)
        msg = f"Saved {data['id']} in {cfg_name(request)}."
        if trimmed:
            msg += (
                f" The Base URL ended in {trimmed}, which is added to every request "
                f"automatically, so it was removed: {data['base_url']}."
            )
        if var and not os.environ.get(var):
            msg += f" It has no key yet: paste one on its Edit page or under API keys ({var})."
        return back("/models", msg=msg)

    @app.post("/models/{model_id}/delete")
    def model_delete(request: Request, model_id: str):
        cfg, _ = load(request)
        var = next((m.api_key_env for m in cfg.models if m.id == model_id), None) if cfg else None
        try:
            models_store.delete_model(cfg_path(request), model_id)
        except ValueError as ex:
            return back("/models", err=redact(str(ex)))
        drop_unused_own_key(request, var)
        return back("/models", msg=f"Deleted {model_id} from {cfg_name(request)}.")

    @app.post("/models/roles")
    def model_roles(
        request: Request,
        baseline: str = Form(...),
        grader: str = Form(""),
        user_model: str = Form(...),
    ):
        try:
            models_store.save_roles(cfg_path(request), baseline, grader or None, user_model)
        except ValueError as ex:
            return back("/models", err=f"Roles not saved: {redact(str(ex))}")
        return back("/models", msg="Roles saved.")

    @app.post("/models/{model_id}/check", response_class=HTMLResponse)
    def model_check(request: Request, model_id: str):
        from benchlab.check import run_check

        cfg, e = load(request)
        if not cfg:
            return HTMLResponse(f"<span class=err>{e}</span>")
        try:
            spec = cfg.get(model_id)
            log = UsageLog(runs_dir(cfg) / "checks.jsonl")
            rep = run_check(cfg, spec, timeout=60.0, usage_log=log)
            probes = []
            for name, res in (("chat", rep.chat), ("tool call", rep.tool), ("stream", rep.stream)):
                if res is None:
                    continue
                probes.append(
                    {
                        "name": name,
                        "ok": res.ok,
                        "status": res.status,
                        "latency": res.latency_s,
                        "ttft": res.ttft_s,
                        "cost": res.cost_usd,
                        "detail": redact(res.error_msg or "")[:160],
                    }
                )
            result = {
                "ok": rep.ok,
                "at": time.time(),
                "tool_calling": rep.tool_calling,
                "problems": [redact(p) for p in rep.problems],
                "probes": probes,
            }
        except Exception as ex:  # noqa: BLE001 - shown to the user, not raised
            result = {"ok": False, "at": time.time(), "problems": [redact(str(ex))], "probes": []}
        last_checks[(cfg_name(request), model_id)] = result
        return tpl.TemplateResponse(request, "_check.html", {"c": result})

    # ---- new run -----------------------------------------------------------------------
    @app.get("/run", response_class=HTMLResponse)
    def run_page(request: Request):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        benches = [b for b in registry.BENCHES.values() if b.adapter is not None]
        return page(
            request,
            "run.html",
            cfg=cfg,
            models=cfg.chat_models() + [m for m in cfg.models if m.kind != "chat"],
            benches=benches,
            stages=registry.STAGE_BENCHES,
        )

    def _selection(form: dict[str, Any]) -> dict[str, Any]:
        models = form.getlist("models") if hasattr(form, "getlist") else form.get("models", [])
        benches = form.getlist("benches") if hasattr(form, "getlist") else form.get("benches", [])
        trials = (form.get("trials") or "").strip()
        budget = (form.get("budget") or "").strip()
        return {
            "models": [m for m in models if m],
            "benches": [b for b in benches if b],
            "trials": int(trials) if trials else None,
            "subset": (form.get("subset") or "").strip() or None,
            "smoke": form.get("smoke") == "on",
            "budget": float(budget) if budget else None,
        }

    def _blocked(cfg: BenchConfig, rows) -> dict[tuple[str, str], str]:
        from benchlab.planning import blocked_reason

        out = {}
        for m, b, adapter_cls, *_ in rows:
            if adapter_cls is not None and (why := blocked_reason(cfg, m, b)):
                out[(m.id, b.key)] = why
        return out

    def _plan(cfg: BenchConfig, sel: dict[str, Any]):
        from benchlab.planning import plan_rows

        if not sel["models"] or not sel["benches"]:
            raise ValueError("pick at least one model and one benchmark")
        specs = [cfg.get(m) for m in sel["models"]]
        benches = registry.select_benches(",".join(sel["benches"]))
        return plan_rows(cfg, specs, benches, sel["trials"], sel["subset"], sel["smoke"])

    @app.post("/run/plan", response_class=HTMLResponse)
    async def run_plan(request: Request):
        cfg, e = load(request)
        if not cfg:
            return HTMLResponse(f"<p class=err>{e}</p>")
        try:
            sel = _selection(await request.form())
            rows = _plan(cfg, sel)
        except (ValueError, KeyError) as ex:
            return HTMLResponse(f"<p class=err>{redact(str(ex))}</p>")
        busy = J.running_pairs(runs_dir(cfg))
        blocked = _blocked(cfg, rows)
        go = [r for r in rows if (r.model.id, r.bench.key) not in blocked and r.adapter_cls]
        total = sum(r.est_usd for r in go if r.est_usd is not None)
        total_s = sum(r.est_s or 0 for r in go)
        return tpl.TemplateResponse(
            request,
            "_plan.html",
            {
                "rows": rows,
                "total": total,
                "total_s": total_s,
                "measured": any(r.time_source == "last run" for r in go),
                "unknown": any(r.est_usd is None for r in rows),
                "busy": busy,
                "blocked": blocked,
                "sel": sel,
            },
        )

    @app.post("/run/start")
    async def run_start(request: Request):
        cfg, e = load(request)
        if not cfg:
            return back("/run", err=e)
        try:
            sel = _selection(await request.form())
            rows = _plan(cfg, sel)
        except (ValueError, KeyError) as ex:
            return back("/run", err=redact(str(ex)))
        blocked = _blocked(cfg, rows)
        pairs = [
            [m.id, b.key]
            for m, b, adapter_cls, *_ in rows
            if adapter_cls is not None and (m.id, b.key) not in blocked
        ]
        if not pairs:
            why = " ".join(sorted(set(blocked.values()))) or "no integrated (model, benchmark) pair"
            return back("/run", err=f"Nothing to run. {why}")
        rd = runs_dir(cfg)
        clash = {tuple(p) for p in pairs} & J.running_pairs(rd)
        if clash:
            txt = ", ".join(f"{m} × {b}" for m, b in sorted(clash))
            return back("/run", err=f"already running: {txt}")
        job = _start_job(cfg, request, sel, pairs)
        left = ", ".join(f"{m} × {b}" for m, b in sorted(blocked))
        msg = "Started." + (f" Left out for a missing key: {left}." if left else "")
        return back(f"/jobs/{job.id}", msg=msg)

    def _run_args(request: Request, sel: dict[str, Any], models, benches) -> list[str]:
        args = ["run", "-c", str(cfg_path(request).resolve()), "-m", ",".join(models)]
        args += ["-b", ",".join(benches)]
        if sel["trials"] is not None:
            args += ["--trials", str(sel["trials"])]
        if sel["subset"]:
            args += ["--subset", sel["subset"]]
        if sel["smoke"]:
            args += ["--smoke"]
        if sel["budget"] is not None:
            args += ["--budget-usd", str(sel["budget"])]
        return args

    def _start_job(
        cfg: BenchConfig, request: Request, sel: dict[str, Any], pairs, extra=None
    ) -> J.Job:
        models = sorted({m for m, _ in pairs})
        benches = sorted({b for _, b in pairs})
        # the same command a terminal would run; shown on the job page
        argv = [sys.executable, "-m", "benchlab.cli", *_run_args(request, sel, models, benches)]
        job_id = J.new_id()
        steps: list[list[str]] = []
        names: list[str] = []
        mode = containers.executor(cfg)
        if mode == "docker":  # one container per benchmark family, run one after another
            for i, (family, ms, bs) in enumerate(containers.split_by_family(pairs)):
                name = containers.container_name(job_id, i)
                names.append(name)
                steps.append(
                    containers.run_argv(
                        cfg,
                        name=name,
                        family=family,
                        cli_args=_run_args(request, sel, ms, bs),
                        root=root,
                        env_names=containers.key_vars(cfg, ms, bs),
                        needs_docker=containers.needs_docker(bs),
                        user=containers.host_user(),
                    )
                )
        job = J.Job(
            id=job_id,
            argv=argv,
            config=cfg_name(request),
            pairs=pairs,
            started_at=time.time(),
            budget_usd=sel["budget"],
            smoke=sel["smoke"],
            label=f"{','.join(models)} × {','.join(benches)}"
            + (" (smoke)" if sel["smoke"] else ""),
            extra={"selection": sel, **(extra or {})},
            executor=mode,
            steps=steps,
            containers=names,
        )
        return J.start(runs_dir(cfg), job, cwd=root)

    # ---- jobs --------------------------------------------------------------------------
    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        return page(request, "jobs.html", cfg=cfg)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        try:
            job = J.load(runs_dir(cfg), job_id)
        except FileNotFoundError:
            return back("/jobs", err=f"no job {job_id} in {cfg_name(request)}")
        return page(request, "job.html", cfg=cfg, job=job, cmd=_cmd_text(job.argv, root))

    @app.get("/fragments/jobs/{job_id}", response_class=HTMLResponse)
    def job_fragment(request: Request, job_id: str):
        cfg, e = load(request)
        if not cfg:
            return HTMLResponse(f"<p class=err>{e}</p>")
        rd = runs_dir(cfg)
        job = J.load(rd, job_id)
        by_run = {x["name"]: x for x in _run_rows(rd, cfg.settings.baseline_model)}
        return tpl.TemplateResponse(
            request,
            "_job_progress.html",
            {
                "job": job,
                "status": J.status(rd, job),
                "rows": J.progress(rd, job),
                "log": redact(J.log_tail(rd, job)),
                "elapsed": J.age(job),
                "by_run": by_run,
                "baseline_id": cfg.settings.baseline_model,
            },
        )

    @app.post("/jobs/{job_id}/stop")
    def job_stop(request: Request, job_id: str):
        cfg, e = load(request)
        if not cfg:
            return back("/jobs", err=e)
        job = J.load(runs_dir(cfg), job_id)
        ok = J.stop(runs_dir(cfg), job)
        return back(
            f"/jobs/{job_id}",
            msg="stop sent: the run saves its progress and exits" if ok else None,
            err=None if ok else "job is not running",
        )

    @app.post("/jobs/{job_id}/resume")
    def job_resume(request: Request, job_id: str):
        cfg, e = load(request)
        if not cfg:
            return back("/jobs", err=e)
        rd = runs_dir(cfg)
        old = J.load(rd, job_id)
        if J.status(rd, old) == "running":
            return back(f"/jobs/{job_id}", err="still running")
        clash = {tuple(p) for p in old.pairs} & J.running_pairs(rd)
        if clash:
            return back(f"/jobs/{job_id}", err="another job is running these pairs")
        sel = old.extra.get("selection")
        if sel and old.config == cfg_name(request):  # rebuilt: new container names, same run
            job = _start_job(cfg, request, sel, old.pairs, extra={"resumes": old.id})
        else:
            job = J.Job(
                id=J.new_id(),
                argv=old.argv,
                config=old.config,
                pairs=old.pairs,
                started_at=time.time(),
                budget_usd=old.budget_usd,
                smoke=old.smoke,
                label=old.label,
                extra={**old.extra, "resumes": old.id},
            )
            J.start(rd, job, cwd=root)
        return back(f"/jobs/{job.id}", msg=f"resumed from job {old.id}")

    # ---- runs / results ----------------------------------------------------------------
    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(request: Request, model: str = "", bench: str = ""):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        rows = [
            r
            for r in _run_rows(runs_dir(cfg), cfg.settings.baseline_model)
            if (not model or r["model"] == model) and (not bench or r["bench"] == bench)
        ]
        rep = root / cfg.settings.results_dir / "report.html"
        return page(
            request,
            "runs.html",
            cfg=cfg,
            rows=rows,
            model=model,
            bench=bench,
            report_at=rep.stat().st_mtime if rep.exists() else None,
        )

    @app.get("/runs/{name}", response_class=HTMLResponse)
    def run_detail(request: Request, name: str):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        p = runs_dir(cfg) / name
        if Path(name).name != name or not (p / RunDir.CONFIG).exists():
            return back("/runs", err=f"no run {name}")
        r = RunDir(p)
        res = r.read_result() if r.has_result() else None
        me = next(
            (x for x in _run_rows(runs_dir(cfg), cfg.settings.baseline_model) if x["name"] == name),
            {},
        )
        tests = samples.load(r) if res else None
        return page(
            request,
            "run_detail.html",
            cfg=cfg,
            name=name,
            conf=r.read_config(),
            res=res,
            state=r.state(),
            usage=r.usage_log.summarize().as_dict(),
            raw=sorted(x.name for x in r.raw_dir.iterdir()) if r.raw_dir.exists() else [],
            me=me,
            tests=tests,
            listing=_tests_page(
                tests, "fail" if tests and tests["counts"]["fail"] else "all", "", "", 1
            )
            if tests
            else None,
        )

    @app.post("/runs/{name}/tests", response_class=HTMLResponse)
    async def run_tests(request: Request, name: str):
        """One page of a run's tests, filtered by status, category and text."""
        cfg, e = load(request)
        p = runs_dir(cfg) / name if cfg else None
        if not cfg or Path(name).name != name or not p or not (p / RunDir.RESULT).exists():
            return HTMLResponse("<p class=err>no such finished run</p>")
        f = await request.form()
        listing = _tests_page(
            samples.load(RunDir(p)),
            str(f.get("status") or "all"),
            str(f.get("category") or ""),
            str(f.get("q") or ""),
            int(str(f.get("page") or "1") or 1),
        )
        return tpl.TemplateResponse(request, "_tests.html", {"listing": listing})

    @app.get("/results", response_class=HTMLResponse)
    def results_page(request: Request):
        from benchlab.report import collect, comparison, suspicious

        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        col = collect(cfg)
        rep = root / cfg.settings.results_dir / "report.html"
        return page(
            request,
            "results.html",
            cfg=cfg,
            cmp=comparison(cfg, col),
            susp=suspicious(cfg, col),
            report_at=rep.stat().st_mtime if rep.exists() else None,
        )

    @app.post("/report")
    def report_build(request: Request):
        from benchlab.report import build_report

        cfg, e = load(request)
        if not cfg:
            return back("/results", err=e)
        out = root / cfg.settings.results_dir / "report.html"
        try:
            rep = build_report(cfg, out)
        except Exception as ex:  # noqa: BLE001
            return back("/results", err=f"Export failed: {redact(str(ex))}")
        return back(
            "/results",
            msg=f"Exported {out.name}: {len(rep.collected.entries)} results, "
            f"{len(rep.suspicious)} to look at. Open it with the button below.",
        )

    @app.get("/report")
    def report_view(request: Request):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        p = root / cfg.settings.results_dir / "report.html"
        if not p.exists():
            return back("/results", err="No report exported yet.")
        return FileResponse(p, media_type="text/html")

    # ---- import ------------------------------------------------------------------------
    @app.get("/import", response_class=HTMLResponse)
    def import_page(request: Request):
        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        return page(request, "import.html", cfg=cfg, models=cfg.chat_models(), result=None)

    @app.post("/import", response_class=HTMLResponse)
    def import_post(request: Request, model: str = Form(...), file: UploadFile = CsvFile):
        from benchlab.adapters.redteam import import_csv
        from benchlab.results import append_result

        cfg, e = load(request)
        if not cfg:
            return broken(request, e)
        fname = Path(file.filename or "redteam.csv").name
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / fname
            with open(tmp, "wb") as f:
                shutil.copyfileobj(file.file, f)
            try:
                rd, res = import_csv(cfg, cfg.get(model), tmp)
            except (ValueError, KeyError) as ex:
                return page(
                    request,
                    "import.html",
                    cfg=cfg,
                    models=cfg.chat_models(),
                    result=None,
                    error=redact(str(ex)),
                )
        append_result(root / cfg.settings.results_dir, res, rd)
        return page(
            request,
            "import.html",
            cfg=cfg,
            models=cfg.chat_models(),
            result={"run": rd.name, "res": res},
        )

    return app


# --------------------------------------------------------------------------- #
TESTS_PER_PAGE = 40


def _tests_page(
    data: dict[str, Any], status: str, category: str, q: str, page: int
) -> dict[str, Any]:
    """Filter a run's samples and cut one page; text search covers id, prompts and answers."""
    q = q.strip().lower()
    rows = [
        s
        for s in data["samples"]
        if (status == "all" or s["status"] == status)
        and (not category or s["category"] == category)
        and (
            not q
            or q
            in " ".join((s["id"], s["prompt"], s["response"], s["reason"], s["system"])).lower()
        )
    ]
    pages = max(1, -(-len(rows) // TESTS_PER_PAGE))
    page = min(max(1, page), pages)
    return {
        "rows": rows[(page - 1) * TESTS_PER_PAGE : page * TESTS_PER_PAGE],
        "total": len(rows),
        "page": page,
        "pages": pages,
        "status": status,
        "category": category,
        "q": q,
    }


def _cmd_text(argv: list[str], root: Path) -> str:
    """The equivalent terminal command, for copy/paste."""
    a = list(argv)
    if a[:3] == [sys.executable, "-m", "benchlab.cli"]:
        a = ["benchlab", *a[3:]]
    return " ".join(
        str(Path(x).relative_to(root)) if x.startswith(str(root) + "/") else x for x in a
    )


def _run_rows(rd: Path, baseline_id: str = "") -> list[dict[str, Any]]:
    """Run folders newest first, each with the baseline model's score on the same bench
    (its newest finished run, a real run preferred over a smoke run) for the readout."""
    rows = []
    for r in sorted(RunDir.list_runs(rd), key=lambda r: r.name, reverse=True):
        try:
            c = r.read_config()
        except Exception:  # noqa: BLE001
            continue
        row = {
            "name": r.name,
            "model": c.get("model_id"),
            "bench": c.get("bench"),
            "smoke": c.get("smoke"),
            "imported": bool((c.get("extra") or {}).get("imported")),
            "date": (c.get("created_at") or "")[:16].replace("T", " "),
            "done": False,
        }
        if r.has_result():
            try:
                res = r.read_result()
            except Exception:  # noqa: BLE001
                rows.append(row)
                continue
            row.update(
                done=True,
                metric=res.get("primary_metric"),
                value=res["metrics"].get(res.get("primary_metric")),
                n=res.get("n_tasks"),
                cost=res.get("cost_usd"),
                errors=(res.get("errors") or {}).get("total", 0),
                warnings=len(res.get("warnings") or []),
                pct=row["bench"] not in NON_PCT_BENCHES,
            )
        rows.append(row)
    base: dict[str, float] = {}
    for smoke_ok in (False, True):
        for r in rows:
            if (
                r["done"]
                and r["model"] == baseline_id
                and r.get("value") is not None
                and (smoke_ok or not r["smoke"])
            ):
                base.setdefault(r["bench"], r["value"])
    for r in rows:
        r["is_base"] = r["model"] == baseline_id
        r["base"] = base.get(r["bench"])
    return rows


def _spend_today(rd: Path) -> list[dict[str, Any]]:
    """Requests / tokens / cost per model since local midnight, from every usage log."""
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    logs = [rd / "checks.jsonl", rd / "proxy_usage.jsonl"]
    if rd.exists():
        logs += [
            d / RunDir.USAGE
            for d in rd.iterdir()
            if d.is_dir()
            and (d / RunDir.USAGE).exists()
            and (d / RunDir.USAGE).stat().st_mtime >= midnight
        ]
    per: dict[str, dict[str, Any]] = {}
    for p in logs:
        for rec in UsageLog(p).records():
            if (rec.get("ts") or 0) < midnight:
                continue
            d = per.setdefault(
                rec.get("model_id", "?"),
                {"requests": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "known": True},
            )
            d["requests"] += 1
            d["tokens_in"] += int(rec.get("tokens_in") or 0)
            d["tokens_out"] += int(rec.get("tokens_out") or 0)
            if rec.get("cost_usd") is None:
                d["known"] = d["known"] and not (rec.get("tokens_in") or rec.get("tokens_out"))
            else:
                d["cost"] += float(rec["cost_usd"])
    return [{"model": k, **v} for k, v in sorted(per.items())]


def _form_values(raw: dict[str, Any]) -> dict[str, str]:
    """Model entry -> strings for the form."""

    def s(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "on" if v else ""
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        if isinstance(v, dict):
            return yaml.safe_dump(v, default_flow_style=True, sort_keys=False).strip() if v else ""
        return str(v)

    out = {k: s(raw.get(k)) for k in models_store.EDITABLE}
    out.setdefault("provider", "openai_compatible")
    if not raw:
        out.update(provider="openai_compatible", kind="chat", max_concurrency="4")
    return out


def _parse_model_form(form: dict[str, Any]) -> dict[str, Any]:
    """Form strings -> values for models_store.save_model (None removes a key)."""

    def text(k: str) -> str | None:
        v = str(form.get(k) or "").strip()
        return v or None

    def num(k: str, cast=float):
        v = text(k)
        if v is None:
            return None
        try:
            return cast(v)
        except ValueError:
            raise ValueError(f"{k}: {v!r} is not a number") from None

    def lst(k: str) -> list[str] | None:
        v = text(k)
        return [x.strip() for x in v.split(",") if x.strip()] if v else None

    ep = text("extra_params")
    try:
        extra = yaml.safe_load(ep) if ep else None
    except yaml.YAMLError as e:
        raise ValueError(f"extra_params is not valid YAML: {e}") from None
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra_params must be a mapping, e.g. {temperature: 0.0}")
    if not text("id") or not text("name") or not text("model"):
        raise ValueError("id, name and model are required")
    return {
        "id": text("id"),
        "name": text("name"),
        "provider": text("provider") or "openai_compatible",
        "base_url": text("base_url"),
        "api_version": text("api_version"),
        "region": text("region"),
        "model": text("model"),
        "kind": None if text("kind") in (None, "chat") else text("kind"),
        "self_hosted": True if form.get("self_hosted") == "on" else None,
        "price_in_per_mtok": num("price_in_per_mtok"),
        "price_out_per_mtok": num("price_out_per_mtok"),
        "max_concurrency": num("max_concurrency", int),
        "suites": lst("suites"),
        "extras": lst("extras"),
        "extra_params": extra if extra is not None else ({} if ep is not None else None),
        "knowledge_cutoff": text("knowledge_cutoff"),
        "notes": text("notes"),
    }


def serve(
    port: int = 8765,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    hosts: tuple[str, ...] = (),
) -> None:  # pragma: no cover - CLI entry
    import secrets

    import uvicorn

    # BENCHLAB_UI_TOKEN keeps the link stable across restarts (a container, a server)
    token = os.environ.get("BENCHLAB_UI_TOKEN") or secrets.token_urlsafe(18)
    url = f"http://127.0.0.1:{port}/?token={token}"
    print(f"BenchLab control panel: {url}\n(this link is your session key)", flush=True)
    if open_browser:
        _open_quietly(url)
    uvicorn.run(
        create_app(token, port=port, hosts=hosts), host=host, port=port, log_level="warning"
    )


def _open_quietly(url: str) -> None:  # pragma: no cover - desktop only
    """Open the browser without its own start-up chatter (Chromium's "zygote" lines)."""
    import shutil
    import subprocess
    import webbrowser

    opener = shutil.which("xdg-open") if sys.platform.startswith("linux") else None
    if opener:
        subprocess.Popen(
            [opener, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        webbrowser.open(url)
