# Design: BenchLab control panel (`benchlab ui`)

## Intent
Today BenchLab is driven entirely from the terminal. The user wants a web control
panel, for themself only and on their own machine, to:
- manage models
- start benchmark runs
- set budgets
- import external CSVs
- watch long runs live

without typing CLI commands. Success: everything they do day to day with
`benchlab models / check / run / runs / usage / report / import-redteam` can be
done from the browser, and a run started there behaves exactly like one started in
the terminal.

## Goals
- Pick a config: `models.yaml` or `models.xkiro.yaml`. A header switcher; each config
  keeps its own `runs_dir` / `results_dir`.
- **Models:**
  - list them with key status (set / missing, never the value)
  - add, edit and delete models, written back to the YAML
  - a "test connection" button that runs `benchlab check` and shows its probes
- **New run:**
  - choose models × benches (stage / suite / bench), trials, subset, smoke, budget
  - see the dry-run plan and cost estimate before starting
- **Jobs:**
  - runs keep going in the background when the tab or the panel is closed
  - reopening the panel shows live progress: log tail, tasks done, requests, tokens,
    spend vs budget
  - a Stop button saves progress, so the run can be resumed later
- **Results:**
  - run history, each run's result and warnings
  - the generated report, usage and error rates
- **Import:** upload a red-team CSV for a chosen model (existing `import-redteam` logic).

## Non-goals (for now)
- Multiple users, login, remote access, a shared server.
- Editing API keys in the UI. Keys stay in `.env`.
- Changing how benches run or how they're scored.

## Anti-goals (would make it a failure even if it works)
- A run started from the panel that differs from the same CLI run (other params,
  other folders, different resume behaviour).
- API keys appearing in pages, logs or the browser.
- Any website open in another tab being able to start a costly run through the
  panel.
- Saving a model edit that silently wipes the comments and field docs in
  `models.yaml`.

## Constraints
- Local only: bind to `127.0.0.1`.
- Reuse BenchLab code. No logic is duplicated between the CLI and the panel.
- Long jobs: up to hours, some in Docker, and they must survive the panel restarting.
- Same Python environment. FastAPI, uvicorn, Jinja2 and multipart are already
  installed (through `litellm[proxy]`) and become explicit dependencies.

## Approaches
- **A. Python server + server-rendered pages + htmx: chosen.** `benchlab ui` serves
  the pages from inside the package and reuses its modules directly. Live parts
  refresh with htmx polling or server-sent events. One language, no JS build. The UI
  is plainer.
- **B. FastAPI JSON API + React app.** Richest UI, but two codebases and a Node
  toolchain. Rejected: too much weight for a single-user local tool.
- **C. Streamlit.** Fastest first screen, but its rerun model handles background
  jobs and live logs poorly, and those are core here. Rejected.

## How it works (decided)
- **Start:** `benchlab ui [--port 8765]` opens `http://127.0.0.1:8765`.
- **Jobs are separate processes.** "Start" launches the *same* CLI command
  (`benchlab run -c <cfg> -m … -b … --budget-usd …`) as a detached process in its own
  process group, with stdout and stderr going to a log file. This gives identical
  behaviour to the CLI for free, plus its existing resume and budget handling.
  - A small job record (id, command, pid, config, start time, log path, exit code)
    lives in `<runs_dir>/.jobs/`.
  - A thin launcher wrapper records the exit code, so status still works after the
    panel restarts.
- **Stop** sends SIGINT to the job's process group. The CLI already catches it,
  saves state and exits cleanly. Resume = start the same run again.
- **Progress comes from files BenchLab already writes:**
  - the job log tail
  - `state.json` (completed count vs the planned `n` in `config.json`)
  - `usage.jsonl` (requests, tokens, cost so far, errors), shown against the budget

  How detailed progress is depends on the bench. Harness benches like Terminal-Bench
  report through the log and the usage records rather than per-task counts, and the
  UI says so instead of showing a fake percentage.
- **No duplicate runs:** the panel refuses to start a (model, bench) pair that
  already has a running job in the same config, because both would resume into the
  same run folder.
- **Shared logic:** the dry-run planning that lives in the CLI's private helper
  (`_plan_rows`, `benchlab/cli.py`) moves to a shared module that both the CLI and
  the panel call. Run listing reuses `RunDir.list_runs`, usage reuses `UsageLog`,
  the report reuses `build_report`, import reuses `benchlab/adapters/redteam.py`,
  and the connection test reuses the `check` logic in `benchlab/llm.py`.
- **Model editing keeps comments:** the YAML is round-tripped with `ruamel.yaml`
  (a new dependency), so comments and order survive. Every edit is validated with
  BenchLab's existing config models before saving, and a backup copy is kept.
- **Security:**
  - the server binds to localhost only
  - a random session token is generated at startup and put into the URL it opens;
    every state-changing request checks that token and the `Origin` header, so
    another site can't trigger runs
  - all output (logs, errors) goes through `benchlab.redact`

## Screens
1. **Dashboard:** running jobs with live progress, recent runs, today's spend per
   model.
2. **Models:** table with key status and last check result. Add / edit / delete
   forms. Test-connection button.
3. **New run:** model and bench pickers that only offer valid pairs (registry
   `applies_to`), options, then a live dry-run plan with cost estimate, then Start.
4. **Job:** live log, progress, spend vs budget, Stop / Resume.
5. **Runs & results:** history with filters, a detail page per run (metrics,
   warnings, errors, harness versions), the latest `report.html`, and a "rebuild
   report" button.
6. **Import:** upload a CSV, pick a model, see the summary and warnings.

## Left to the builder
Page styling, the htmx-versus-SSE choice per widget, the exact job-file format, the
module layout under `benchlab/ui/`, and test structure (FastAPI TestClient plus the
existing mock OpenAI server in `tests/conftest.py`).

## Calls I made (please check)
- Jobs run the CLI as a subprocess rather than calling adapters inside the server.
  This is what lets runs survive the panel and stay identical to the CLI.
- `ruamel.yaml` is added so model edits keep the YAML comments.
- The session token plus Origin check, even though it's local only.
- Port 8765 by default.

## After approval
1. Save this design as `docs/superpowers/specs/2026-10-28-benchlab-control-panel-design.md`.
2. Write the implementation plan.
3. Build in this order: server skeleton and config switcher → models → new run and
   jobs → results → import.

## Verification
- Start a smoke run from the panel and check that it produces the same run folder
  and `result.json` as the equivalent CLI command.
- Close the panel during a run, reopen it, and check that progress continues.
- Stop a run, then resume it.
- Edit a model and confirm the YAML comments are intact.
- Confirm that a cross-origin POST is rejected.
- `pytest -q` and `ruff check` stay green.
