"""BenchLab command line."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from benchlab import __version__, registry
from benchlab.config import BenchConfig, ModelSpec, load_config
from benchlab.llm import litellm_model_name, per_model_exceptions, request_params
from benchlab.redact import redact
from benchlab.runs import RunConfig, RunDir
from benchlab.usage import BudgetExceeded, BudgetGuard, UsageLog

app = typer.Typer(
    help="BenchLab: run a fixed benchmark suite against any model.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()
err = Console(stderr=True)

ConfigOpt = typer.Option(None, "--config", "-c", help="models.yaml path (default ./models.yaml)")


def _cfg(path: str | None) -> BenchConfig:
    try:
        return load_config(path)
    except Exception as e:  # noqa: BLE001
        err.print(f"[red]config error:[/red] {redact(e)}")
        raise typer.Exit(2) from None


def _usd(v) -> str:
    if v is None:
        return "-"
    return f"${v:.4f}" if v < 0.01 else f"${v:.2f}"


def _fmt(v, nd=2, suffix=""):
    if v is None:
        return "-"
    return f"{v:.{nd}f}{suffix}"


# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the benchlab version."""
    console.print(f"benchlab {__version__}")


@app.command()
def models(
    config: str | None = ConfigOpt,
    keys: bool = typer.Option(False, help="Show whether each key is set"),
) -> None:
    """Validate models.yaml and list the models."""
    cfg = _cfg(config)
    t = Table(
        title=f"models.yaml ({len(cfg.models)} models, baseline={cfg.settings.baseline_model})"
    )
    for c in (
        "id",
        "name",
        "kind",
        "provider",
        "litellm model",
        "self-hosted",
        "$/Mtok in/out",
        "conc",
        "suites",
        "extras",
        "exceptions",
    ):
        t.add_column(c)
    if keys:
        t.add_column("key set?")
    for m in cfg.models:
        row = [
            m.id,
            m.name,
            m.kind,
            m.provider,
            litellm_model_name(m),
            "yes" if m.self_hosted else "",
            f"{m.price_in_per_mtok:g}/{m.price_out_per_mtok:g}" if m.has_prices else "n/a",
            str(m.max_concurrency),
            ",".join(m.suites or []),
            ",".join(m.extras),
            json.dumps(per_model_exceptions(m, cfg.settings)) if m.kind == "chat" else "",
        ]
        if keys:
            row.append(
                "yes"
                if (m.api_key_env and os.environ.get(m.api_key_env))
                else ("n/a" if not m.api_key_env else "[red]NO[/red]")
            )
        t.add_row(*row)
    console.print(t)


@app.command()
def check(
    model: str = typer.Option(..., "--model", "-m", help="model id, comma list, or 'all'"),
    config: str | None = ConfigOpt,
    timeout: float = typer.Option(60.0, help="per-request timeout in seconds"),
    no_stream: bool = typer.Option(False, help="skip the streaming/TTFT probe"),
) -> None:
    """Send one chat, one tool-call and one streamed request; print latency, tool-calling status and usage."""
    from benchlab.check import run_check

    cfg = _cfg(config)
    specs = cfg.select(model)
    log = UsageLog(Path(cfg.settings.runs_dir) / "checks.jsonl")
    failed = 0
    for spec in specs:
        console.rule(f"[bold]{spec.id}[/bold] - {spec.name} ({litellm_model_name(spec)})")
        rep = run_check(cfg, spec, timeout=timeout, usage_log=log, with_stream=not no_stream)
        t = Table(show_header=True, header_style="bold")
        for c in ("probe", "status", "latency", "ttft", "tokens in/out", "cost", "detail"):
            t.add_column(c)

        def row(name, res, detail="", t=t):
            if res is None:
                t.add_row(name, "skipped", "-", "-", "-", "-", detail)
                return
            st = f"[green]{res.status}[/green]" if res.ok else f"[red]{res.status}[/red]"
            tok = (
                f"{res.tokens_in}/{res.tokens_out}"
                if res.usage_returned
                else "[yellow]none[/yellow]"
            )
            t.add_row(
                name,
                st,
                _fmt(res.latency_s, 2, "s"),
                _fmt(res.ttft_s, 3, "s"),
                tok,
                "-" if res.cost_usd is None else f"${res.cost_usd:.6f}",
                detail or (res.error_msg or "")[:100] if not res.ok else detail,
            )

        row(
            "chat",
            rep.chat,
            (rep.chat.content or "").strip()[:60] if rep.chat and rep.chat.ok else "",
        )
        row("tool call", rep.tool, f"{rep.tool_calling}: {rep.tool_detail}")
        row("stream", rep.stream, "TTFT measured" if rep.stream and rep.stream.ok else "")
        console.print(t)
        console.print(f"params: {rep.params}  per-model exceptions: {rep.exceptions or 'none'}")
        if rep.ok:
            console.print("[green]OK[/green]: endpoint works, tool calling works, usage returned")
        else:
            failed += 1
            for p in rep.problems:
                console.print(f"[red]PROBLEM[/red]: {redact(p)}")
            if spec.self_hosted:
                console.print(
                    "[yellow]hint[/yellow]: for self-hosted models check the chat template, "
                    "the tool-call parser (--tool-call-parser / --enable-auto-tool-choice) "
                    "and the quantization."
                )
    console.print(f"\nusage log: {log.path}")
    raise typer.Exit(1 if failed else 0)


@app.command("proxy-config")
def proxy_config(
    config: str | None = ConfigOpt,
    out: str = typer.Option("litellm_config.yaml", help="output path"),
) -> None:
    """Write the LiteLLM proxy config (model ids -> providers, prices, usage callback)."""
    from benchlab.proxy import proxy_base_url, write_proxy_config

    cfg = _cfg(config)
    p = write_proxy_config(cfg, out)
    console.print(f"wrote {p}  ->  proxy will serve {proxy_base_url(cfg)}")
    console.print("start it with: benchlab proxy")


@app.command()
def proxy(
    config: str | None = ConfigOpt,
    out: str = typer.Option("litellm_config.yaml", help="config path to (re)generate"),
) -> None:
    """Regenerate the proxy config and run the LiteLLM proxy in the foreground."""
    from benchlab.proxy import launch_proxy, proxy_base_url, write_proxy_config

    cfg = _cfg(config)
    p = write_proxy_config(cfg, out)
    os.environ.setdefault(
        "BENCHLAB_USAGE_LOG", str(Path(cfg.settings.runs_dir) / "proxy_usage.jsonl")
    )
    console.print(
        f"serving {proxy_base_url(cfg)} from {p}; usage -> {os.environ['BENCHLAB_USAGE_LOG']}"
    )
    rc = launch_proxy(cfg, p)
    raise typer.Exit(rc if isinstance(rc, int) else 0)


# --------------------------------------------------------------------------- #
def _plan_rows(
    cfg: BenchConfig,
    specs: list[ModelSpec],
    benches: list[registry.BenchInfo],
    trials: int | None,
    subset: str | None,
    smoke: bool,
):
    rows = []
    for b in benches:
        adapter_cls = registry.load_adapter(b)
        k = trials if trials is not None else (b.trials_default or 1)
        if trials is None and b.key.startswith("t2") and cfg.settings.tau2_trials:
            k = cfg.settings.tau2_trials  # profile: lite -> 1
        for m in specs:
            if not registry.applies_to(b, m):
                continue
            n = b.n_tasks_default
            notes = []
            if adapter_cls is not None:
                try:
                    plan = adapter_cls(cfg).plan(m, subset, k, smoke)
                    n, tin, tout, notes = (
                        plan.n_tasks,
                        plan.est_tokens_in_per_task,
                        plan.est_tokens_out_per_task,
                        plan.notes,
                    )
                except Exception as e:  # noqa: BLE001
                    tin, tout = b.tokens_in_per_task, b.tokens_out_per_task
                    notes = [f"plan failed: {redact(e)}"]
            else:
                tin, tout = b.tokens_in_per_task, b.tokens_out_per_task
                if smoke:
                    n = min(n or 3, 3)
                notes = ["adapter not integrated yet"]
            est = m.cost_usd(tin * n * k, tout * n * k) if n else None
            rows.append((m, b, adapter_cls, n, k, est, notes))
    return rows


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="model id, comma list, or 'all'"),
    bench: str | None = typer.Option(None, "--bench", "-b", help="comma list of bench keys"),
    suite: str | None = typer.Option(
        None, "--suite", help="core | coder | extras | retrieval | serving"
    ),
    stage: int | None = typer.Option(None, "--stage", help="1-4"),
    trials: int | None = typer.Option(
        None, help="trials per task (default: bench default; τ² uses 3 for pass^k)"
    ),
    subset: str | None = typer.Option(None, help="subset name/size understood by the adapter"),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="abort cleanly past this spend"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the plan and estimated cost only"),
    smoke: bool = typer.Option(False, "--smoke", help="2-3 tasks per bench"),
    resume: bool = typer.Option(True, help="continue the newest unfinished run for (model, bench)"),
    config: str | None = ConfigOpt,
) -> None:
    """Run benchmarks. Every run writes runs/<timestamp>_<model>_<bench>/."""
    cfg = _cfg(config)
    specs = cfg.select(model)
    try:
        benches = registry.select_benches(bench, suite, stage)
    except (KeyError, ValueError) as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from None
    rows = _plan_rows(cfg, specs, benches, trials, subset, smoke)
    if not rows:
        err.print("nothing to run: no (model, bench) pair matches the selectors")
        raise typer.Exit(1)

    t = Table(title="plan" + (" (dry run)" if dry_run else ""))
    for c in ("model", "bench", "harness", "tasks", "trials", "est. cost", "docker", "status"):
        t.add_column(c)
    total = 0.0
    unknown = False
    for m, b, adapter_cls, n, k, est, notes in rows:
        if est is None:
            unknown = True
        else:
            total += est
        status = "ready" if adapter_cls else "[yellow]not integrated yet[/yellow]"
        t.add_row(
            m.id,
            b.key,
            b.harness,
            str(n or "?"),
            str(k),
            "-" if est is None else f"${est:.2f}",
            "yes" if b.needs_docker else "",
            status + ("; " + "; ".join(notes) if notes and adapter_cls else ""),
        )
    console.print(t)
    console.print(
        f"estimated total: ${total:.2f}"
        + (" (+ models with unknown prices)" if unknown else "")
        + "  - rough, from per-task token profiles; real cost comes from the usage logs"
    )
    if budget_usd is not None and total > budget_usd:
        console.print(f"[yellow]estimate exceeds --budget-usd {budget_usd}[/yellow]")
    if dry_run:
        raise typer.Exit(0)

    rc = 0
    for m, b, adapter_cls, _n, k, _est, _notes in rows:
        if adapter_cls is None:
            console.print(
                f"[yellow]skip[/yellow] {m.id} x {b.key}: adapter not integrated yet ({b.notes or b.harness})"
            )
            continue
        rc |= _run_one(cfg, m, b, adapter_cls, k, subset, smoke, budget_usd, resume)
    raise typer.Exit(rc)


def _run_one(cfg, m, b, adapter_cls, trials, subset, smoke, budget_usd, resume) -> int:
    from benchlab.adapters.base import HarnessNotReady
    from benchlab.results import append_result

    adapter = adapter_cls(cfg)
    console.rule(f"{m.id} x {b.key}")
    try:
        harness = adapter.prepare()
    except HarnessNotReady as e:
        err.print(f"[red]harness not ready for {b.key}:[/red] {redact(e)}")
        return 1
    run_dir = RunDir.find_resumable(cfg.settings.runs_dir, m.id, b.key) if resume else None
    if run_dir is not None:
        console.print(
            f"resuming {run_dir.path} ({len(run_dir.completed_ids())} tasks already done)"
        )
    else:
        run_dir = RunDir.create(cfg.settings.runs_dir, m.id, b.key)
        plan = adapter.plan(m, subset, trials, smoke)
        run_dir.write_config(
            RunConfig(
                model_id=m.id,
                bench=b.key,
                model=m.public_dict(),
                request_params=request_params(m, cfg.settings),
                per_model_exceptions=per_model_exceptions(m, cfg.settings),
                harness=harness,
                subset={"name": subset, "ids": plan.subset_ids, "n": plan.n_tasks},
                trials=trials,
                smoke=smoke,
                user_model=cfg.settings.tau2_user_model if b.key.startswith("t2") else None,
            )
        )
        console.print(f"run dir: {run_dir.path}")
    os.environ["BENCHLAB_USAGE_LOG"] = str(run_dir.path / RunDir.USAGE)
    budget = BudgetGuard(budget_usd, logs=[run_dir.usage_log])
    t0 = time.time()
    try:
        adapter.run(m, run_dir, subset, trials, smoke, budget=budget)
    except BudgetExceeded as e:
        run_dir.add_elapsed(time.time() - t0)
        err.print(f"[red]aborted:[/red] {e}. Re-run the same command to resume.")
        return 3
    except KeyboardInterrupt:
        run_dir.add_elapsed(time.time() - t0)
        err.print("interrupted; progress is saved, re-run to resume")
        return 130
    run_dir.add_elapsed(time.time() - t0)
    res = adapter.parse(run_dir)
    res.wall_time_s = run_dir.state().get("elapsed_s", time.time() - t0)
    run_dir.write_result(res)
    p = append_result(cfg.settings.results_dir, res, run_dir)
    console.print(
        f"[green]{b.key} {m.id}[/green]: {res.primary_metric}={_fmt(res.primary_value, 1)}  "
        f"n={res.n_tasks} trials={res.trials} errors={res.errors.total} "
        f"cost={_usd(res.cost_usd)}  -> {p}"
    )
    for w in res.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    return 0


@app.command()
def runs(config: str | None = ConfigOpt) -> None:
    """List run folders and their status."""
    cfg = _cfg(config)
    t = Table(title=f"runs in {cfg.settings.runs_dir}")
    for c in ("run", "model", "bench", "status", "done", "primary", "cost"):
        t.add_column(c)
    for r in RunDir.list_runs(cfg.settings.runs_dir):
        c = r.read_config()
        st = r.state()
        if r.has_result():
            res = r.read_result()
            t.add_row(
                r.name,
                c["model_id"],
                c["bench"],
                "[green]done[/green]",
                str(len(st.get("completed", []))),
                _fmt(res["metrics"].get(res["primary_metric"]), 1),
                "-" if res.get("cost_usd") is None else f"${res['cost_usd']:.2f}",
            )
        else:
            s = r.usage_log.summarize()
            t.add_row(
                r.name,
                c["model_id"],
                c["bench"],
                "[yellow]incomplete[/yellow]",
                str(len(st.get("completed", []))),
                "-",
                "-" if not s.cost_known else f"${s.cost_usd:.2f}",
            )
    console.print(t)


@app.command()
def usage(
    config: str | None = ConfigOpt,
    log: str | None = typer.Option(None, help="usage JSONL (default: all run logs + checks)"),
) -> None:
    """Summarize tokens, cost, latency and error rates from the usage logs."""
    cfg = _cfg(config)
    logs = (
        [Path(log)]
        if log
        else [r.path / RunDir.USAGE for r in RunDir.list_runs(cfg.settings.runs_dir)]
    )
    logs += (
        []
        if log
        else [
            Path(cfg.settings.runs_dir) / "checks.jsonl",
            Path(cfg.settings.runs_dir) / "proxy_usage.jsonl",
        ]
    )
    per_model: dict[str, list] = {}
    for p in logs:
        for rec in UsageLog(p).records():
            per_model.setdefault(rec.get("model_id", "?"), []).append(rec)
    t = Table(title="usage")
    for c in (
        "model",
        "requests",
        "ok",
        "timeouts",
        "rate-limit",
        "errors",
        "malformed tools",
        "tokens in",
        "tokens out",
        "cost",
        "p50 lat",
        "p95 lat",
        "p50 ttft",
    ):
        t.add_column(c)
    from benchlab.usage import UsageSummary

    for mid, recs in sorted(per_model.items()):
        s = UsageSummary()
        for r in recs:
            s.add(r)
        d = s.as_dict()
        t.add_row(
            mid,
            str(d["n_requests"]),
            str(d["n_ok"]),
            str(d["n_timeouts"]),
            str(d["n_rate_limit"]),
            str(d["n_errors"]),
            str(d["n_malformed_tool_calls"]),
            str(d["tokens_in"]),
            str(d["tokens_out"]),
            "-" if d["cost_usd"] is None else f"${d['cost_usd']:.4f}",
            _fmt(d["latency_p50_s"], 2, "s"),
            _fmt(d["latency_p95_s"], 2, "s"),
            _fmt(d["ttft_p50_s"], 3, "s"),
        )
    console.print(t)


@app.command()
def report(
    config: str | None = ConfigOpt,
    out: str = typer.Option("report.html", "--out", "-o", help="where to write the HTML report"),
    include_smoke: bool = typer.Option(
        True, "--include-smoke/--no-smoke", help="use smoke runs when nothing else exists (flagged)"
    ),
) -> None:
    """Generate report.html and results/tracker.json from all finished runs (build order step 8)."""
    from benchlab.report import build_report

    cfg = _cfg(config)
    rep = build_report(cfg, out, include_smoke=include_smoke)
    col = rep.collected
    if not col.entries:
        err.print(
            f"[yellow]no finished runs under {cfg.settings.runs_dir}; wrote an empty report[/yellow]"
        )
    t = Table(title="tracker.json")
    for c in ("key", "value", "note"):
        t.add_column(c, overflow="fold")
    for k, v in rep.tracker.items():
        t.add_row(k, v["v"], v["note"][:110] + ("…" if len(v["note"]) > 110 else ""))
    console.print(t)
    if rep.suspicious:
        console.print(f"[yellow]suspicious results: {len(rep.suspicious)}[/yellow]")
        for s in rep.suspicious[:30]:
            console.print(
                f"  - {s['model']} / {s['bench']}: {s['why']}"
                + (f"  -> {s['hint']}" if s["hint"] else "")
            )
    console.print(
        f"report: {rep.html_path}\ntracker: {rep.tracker_path}  ({len(rep.tracker)} keys, {len(col.entries)} results, {len(col.smoke_only)} smoke-only)"
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
