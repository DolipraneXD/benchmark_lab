# BenchLab

BenchLab runs a fixed set of benchmarks against any model you can reach through an API
(endpoint + model name + key): add the model and its key in the web panel and run it and gives comparable, reproducible results. Each benchmark
wraps its **official harness**; BenchLab never scores anything itself. It records every
request (tokens, cost, latency, errors), so you can tell wrong answers apart from timeouts
and API failures.

You can drive it two ways:
- **Terminal:** the `benchlab` command.
- **Web control panel:** `benchlab ui`. It starts the same commands for you and shows them
  live.

Every score in the report is compared to a baseline model, chosen in `models.yaml`
(`settings.baseline_model`) or in the panel under **Models → Roles**.

## Setup

You need Python 3.11+. Docker is only needed for benchmarks that run model-written code
(see [What each benchmark needs](#what-each-benchmark-needs)).

```bash
uv sync                          # with uv
# or, without uv:
python3 -m venv .venv
.venv/bin/pip install -e . pytest pytest-asyncio ruff   # also installs uv, used for the harnesses

cp .env.example .env             # API keys live here (git-ignored); the panel can fill it
source .venv/bin/activate        # makes `benchlab` available in this terminal
benchlab models --keys           # validate models.yaml and show which keys are set
```

The rest of this README assumes the venv is active. Without activating it, put
`.venv/bin/` in front of the command (`.venv/bin/benchlab ...`) or use `uv run benchlab ...`.

Each official harness gets its own environment under `harnesses/<name>/.venv`. It is
created the first time that benchmark runs, with exact versions recorded in
`harnesses/<name>/manifest.json` and in every run's `config.json`.

## Models (`models.yaml`)

All models are listed in `models.yaml`. The top of the file documents every field. An
entry looks like this:

```yaml
settings:
  profile: lite                  # same benches on smaller fixed subsets (see "Cheap mode")
  baseline_model: minimax        # every score is compared to this model
  tau2_user_model: minimax       # the fixed customer simulator for τ²-bench
  # grader_model: <id>           # BrowseComp grader; defaults to the baseline

models:
  - id: minimax
    name: MiniMax M3.1 Flash
    provider: openai_compatible
    base_url: https://api.xkiro.com/v1
    model: minimax/minimax-m3.1-flash
    api_key_env: BENCHLAB_KEY_MINIMAX  # the model's own key, stored in .env
    price_in_per_mtok: 0.15            # USD per 1M tokens; 0 = unknown, cost shows as n/a
    price_out_per_mtok: 0.6
    max_concurrency: 4
```

### Adding a model

The easy way is the panel: **Models → Add a model**. Fill in the endpoint and model name,
paste the API key, and save. Then click **Test** on the model's row.

Each model gets **its own key**, stored in `.env` under a name made from the config and the
model id:

| Config | Model id | Key variable |
|---|---|---|
| `models.yaml` | `minimax` | `BENCHLAB_KEY_MINIMAX` |
| `models.xkiro.yaml` | `minimax` | `BENCHLAB_KEY_XKIRO_MINIMAX` |

Ids are unique within a config, so adding a model never overwrites another model's key, even
when both configs have a model with the same id. When several models use one provider
account, choose **Same key as …** in the form so they share one variable. Renaming a model
moves its key to the new name; deleting a model removes its key from `.env`.

By hand: add an entry to `models.yaml` with `api_key_env: BENCHLAB_KEY_<ID>`, and add the
line `BENCHLAB_KEY_<ID>=<your key>` to `.env`. Any other variable name works too. Then test
the connection:

```bash
benchlab check --model <id>
```

`check` sends three small requests: a chat, a tool call and a streamed answer. It prints
the latency, time to first token, whether tool calling works (and how it failed if it
didn't), the tokens used and the cost. It exits with an error on any problem, so a broken
endpoint fails in seconds rather than an hour into a run.

Useful fields:
- `provider`: `openai_compatible` (needs `base_url`), `anthropic` or `bedrock`.
- `extra_params: {temperature: null}` removes the default temperature for models that reject it.
- `kind: embedding | reranker` marks retrieval models. They run only in the retrieval suite.
- `suites` and `extras` choose which benchmarks a model gets. Chat models default to `core`.
- `self_hosted: true` enables the serving benchmark for that model.
- `reference: true` marks a model that is never called: only its `published` scores appear
  in the report.

### Roles

Three settings name a model:
- `baseline_model`: the model every score is compared to.
- `grader_model`: grades BrowseComp. If unset, the baseline does it. A model never grades
  itself.
- `tau2_user_model`: plays the customer in τ²-bench, and stays the same for every run.

A model that holds a role can't be deleted or renamed until another model takes it. In the
panel, change roles under **Models → Roles**.

### Second config: `models.xkiro.yaml`

A trial config for models reached through xkiro.com, to try the pipeline cheaply. It
starts with only `minimax`, which shares `BENCHLAB_KEY_MINIMAX` with `models.yaml`; add
other models from the panel. It writes to `runs_xkiro/` and
`results_xkiro/`, so trial numbers never mix with real ones. Pass `-c models.xkiro.yaml`
to any command, or switch config in the panel header. A reseller can't prove it serves the
official weights, so use these runs to test the setup, not to decide.

## Running from the terminal

```bash
benchlab check --model all                              # test every model's connection first
benchlab run --model minimax --stage 1 --dry-run            # show the plan and estimated cost
benchlab run --model minimax --stage 1 --smoke              # 2-3 tasks per bench, to check the setup
benchlab run --model minimax --stage 1 --budget-usd 10      # the real run, stopped at $10
benchlab run --model minimax --bench gpqa,bfcl              # specific benchmarks
benchlab runs                                           # list runs and their status
benchlab usage                                          # tokens, cost, latency and error rates
benchlab report                                         # build report.html and results/tracker.json
```

Options for `benchlab run`:

| Option | Meaning |
|---|---|
| `-m, --model` | a model id, a comma list, or `all` (every chat model) |
| `-b, --bench` | comma list of benchmark keys (see the table below) |
| `--stage 1-4` | every benchmark of that stage |
| `--suite` | `core`, `coder`, `extras`, `retrieval` or `serving` |
| `--trials N` | trials per task (τ²-bench defaults to 3) |
| `--subset` | subset name or size understood by the benchmark |
| `--budget-usd X` | stop cleanly once the run has spent X |
| `--dry-run` | print the plan and the estimated cost, run nothing |
| `--smoke` | 2-3 tasks per benchmark |
| `--no-resume` | start a new run instead of continuing the last unfinished one |
| `-c, --config` | another config file (default `models.yaml`) |

**Stopping and resuming.** Press Ctrl+C: progress is saved. Run the same command again to
continue where it stopped.

**Where results go.** Each run gets its own folder,
`runs/<timestamp>_<model>_<bench>/`, with these files:
- `config.json`: the exact settings and harness versions
- `state.json`: the finished tasks, used to resume
- `usage.jsonl`: one line per request
- `raw/`: the harness's own output
- `result.json`: the scores

Every result is also added to `results/results.csv`. `benchlab report` collects all
finished runs into `report.html`, flags suspicious results and lists them in
`results/suspicious.json`.

### Stages

Run them in order. Each stage is more expensive than the last, so a broken model gets
caught early.

| Stage | Command | Benchmarks |
|---|---|---|
| 1 | `benchlab run -m all --stage 1` | GPQA Diamond, BFCL, TTFT probe (cheap) |
| 2 | `benchlab run -m all --stage 2` | τ²-bench airline, retail, telecom |
| 3 | `benchlab run -m all --stage 3` | Terminal-Bench 2.0, SWE-bench Verified (Docker) |
| 4 | `benchlab run -m <ids> --stage 4` | GAIA, BrowseComp, coding, retrieval and serving benches, for chosen models |

### Benchmark keys

| Key | Benchmark | Stage |
|---|---|---|
| `gpqa` | GPQA Diamond | 1 |
| `bfcl` | BFCL (tool calling) | 1 |
| `ttft` | time to first token probe | 1 |
| `t2air`, `t2ret`, `t2tel` | τ²-bench airline, retail, telecom | 2 |
| `tb` | Terminal-Bench 2.0 | 3 |
| `swe` | SWE-bench Verified | 3 |
| `gaia` | GAIA | 4 |
| `bcomp` | BrowseComp (graded by `grader_model`) | 4 |
| `lcb` | LiveCodeBench | 4 |
| `aider` | Aider Polyglot | 4 |
| `mteb_ret`, `mteb_rr`, `beir`, `own` | retrieval: MTEB retrieval and rerank, BEIR, your own data | 4 |
| `serving` | serving benchmark (self-hosted models only) | 4 |
| `redteam` | external red-team results, imported (see below) | — |

`benchlab/registry.py` lists them all, including model-specific extras.

### Cheap mode (`profile: lite`)

With `profile: lite` in `settings:`, every benchmark runs on a smaller **fixed, seeded**
subset that is the same for every model. That costs roughly a fifth of the full suite. Lite
scores are comparable with each other but not with full runs or published numbers, so the
report marks them. Set `profile: full` for the decision runs.

### Importing an external red-team report

A promptfoo red-team CSV from an outside lab can be imported as the `redteam` benchmark
for one model:

```bash
benchlab import-redteam ~/Downloads/report.csv -m minimax
benchlab report
```

The main score is `pass_rate`: the share of graded attacks the model resisted. Rows that
were never graded (empty response and empty reason) count as API errors. Import again to
replace the result: the newest run wins.

### Other commands

```bash
benchlab models [--keys]     # validate the config and list the models
benchlab proxy-config        # write the LiteLLM proxy config (no secrets inside)
benchlab proxy               # run the LiteLLM proxy in the foreground
benchlab version
```

You don't have to start the proxy yourself: `benchlab run` starts it when a harness needs
it.

## Web control panel

### Starting it

```bash
benchlab ui                      # prints a link and opens it in your browser
benchlab ui --no-browser         # only print the link
benchlab ui --port 8766          # if 8765 is taken
```

Open the printed link (`http://127.0.0.1:8765/?token=...`). Start the panel from the repo
folder so it finds `.env` and `models.yaml`.

### Stopping and restarting

Press Ctrl+C in the terminal where it runs. If it runs somewhere else, use:

```bash
pkill -f "benchlab ui"
```

Run `benchlab ui` again to restart. Each start prints a **new link**, and the old one stops
working. Benchmark runs started from the panel are separate processes: they keep running
while the panel is stopped and show up again when it comes back. Restart the panel after
changing its code.

### What it does

- **Dashboard:** running jobs, recent results, and today's spend per model.
- **Models:** add, edit and delete models, set the roles, and test a connection.
- **API keys** (on the Models page): paste a key for each variable the models use, replace
  it or remove it. See [API keys in the panel](#api-keys-in-the-panel).
- **New run:** pick models and benchmarks, see the plan and the cost estimate, then start
  with a budget. A benchmark whose model, or whose helper model, has no key is marked
  "Can't run: missing key" with the reason, and Start leaves it out. Helper models are the
  τ²-bench customer and the BrowseComp grader.
- **Jobs:** live progress, requests, tokens, spend against the budget, and the log. Stop
  saves progress, and Resume continues the run.
- **Results:** every run with its scores and warnings. Rebuild and open the report.
- **Import:** upload a red-team CSV for a model.

A run started from the panel is the exact `benchlab run ...` command shown on its job page.
It writes the same folders a terminal run would. The panel won't start a model and
benchmark pair that is already running. Model edits are checked before saving; comments in
the YAML are kept, and the previous file is saved as `models.yaml.bak`. Job records and
logs are in `runs/.jobs/`.

### API keys in the panel

You can paste a key when you add or edit a model (see [Adding a model](#adding-a-model)).
The **API keys** section on the Models page lists every key variable the models use, with
the models using each one. Paste a key there and click Save to replace it:
- The key is written to `.env` in the repo folder, the same file terminal runs read, so it
  works for both. It replaces any older value for that variable.
- Checks and new runs use it right away, with no panel restart.
- A saved key is never sent back to the page; you only see its last 4 characters.
- **Remove** deletes the line from `.env`.

How the keys are protected:
- `.env` is git-ignored, and the panel makes it readable by you only (`chmod 600`) when it
  starts and on every save. No backup copy of `.env` is made.
- Only variables used by a model in the current config can be written, so the panel can't
  be used to set things like `PATH`.
- The key is not encrypted on disk: anyone who can log in as you, or read your backups, can
  read `.env`.

A key exported in your shell shows as "from your shell". In the terminal, that value wins
over `.env`.

### Security

- The panel listens on 127.0.0.1 only.
- The link sets a session cookie, and every page needs it.
- Changes must come from the panel itself, so another website open in your browser can't
  start runs.

## Tests

The tests never call a real model and cost nothing. `tests/conftest.py` starts a small
OpenAI-compatible mock server, so requests, streaming, tool calls and usage logging run for
real over HTTP.

```bash
pytest -q                                   # all tests
pytest -q tests/test_ui.py                  # one file
pytest -q -k redteam                        # tests whose name matches
ruff check . && ruff format --check .       # lint and formatting
```

With uv, use `uv run pytest -q` instead. Without activating the venv, use
`.venv/bin/pytest -q`.

What the test files cover:

| File | Covers |
|---|---|
| `test_config.py` | loading and validating `models.yaml`, including the shipped configs |
| `test_llm.py` | chat, tool calls, streaming and error types against the mock server, and `check` |
| `test_proxy_logger.py` | the usage records the proxy writes (fields, TTFT, error types) |
| `test_runs_and_usage.py` | the usage summary and budget, run folders and resume, benchmark selection, no secrets in the proxy config |
| `test_cli.py` | `models`, `check`, `run --dry-run` and `proxy-config` from the command line |
| `test_inspect_adapter.py` | GPQA, BFCL, GAIA, BrowseComp: harness commands and parsing |
| `test_tau2_adapter.py` | τ²-bench |
| `test_harbor_adapter.py` | Terminal-Bench and SWE-bench |
| `test_stage5_adapters.py` | Aider Polyglot and LiveCodeBench |
| `test_retrieval_adapter.py` | the retrieval pipeline |
| `test_serving_adapters.py` | the TTFT probe and the serving benchmark |
| `test_stage9_adapters.py` | AutomationBench, Toolathlon, OSWorld |
| `test_report.py` | `report.html`, `tracker.json` and the suspicious-results list |
| `test_redteam_import.py` | importing a red-team CSV |
| `test_ui.py` | the control panel: access check, pages, model edits and roles, starting, stopping and resuming jobs, CSV upload |

Parser tests run on saved harness outputs in `tests/fixtures/`. A few tests are skipped
unless a harness environment is installed (retrieval, serving, AutomationBench). That is
expected.

## What each benchmark needs

| Benchmark | Needs |
|---|---|
| GPQA, BFCL, TTFT probe | API access only |
| τ²-bench | API access for the model and the user-simulator model |
| GAIA | Docker and web access |
| Terminal-Bench, SWE-bench | Docker, about 4 CPU and 16 GB RAM per parallel task, tens of GB of images |
| Aider Polyglot, LiveCodeBench | Docker |
| Retrieval | embedding and rerank endpoints, or a GPU |
| Serving | a self-hosted endpoint; `nvidia-smi` on the serving machine for GPU figures |

## Project layout

```
models.yaml, models.xkiro.yaml   the two configs
benchlab/cli.py                  the benchlab command
benchlab/config.py               config validation
benchlab/llm.py                  model access through LiteLLM: check, probes, budget
benchlab/proxy.py                the LiteLLM proxy used by the harnesses
benchlab/registry.py             benchmark keys, stages and suites
benchlab/adapters/               one adapter per benchmark around its official harness
benchlab/report.py               report.html and tracker.json
benchlab/ui/                     the web control panel
subsets/                         the fixed task subsets
tests/                           the test suite
```
