# BenchLab

A command-line runner that executes a fixed benchmark suite against any model
(API endpoint + model name + API key) and produces comparable, reproducible
results. Every benchmark wraps its **official harness**; BenchLab never scores
anything itself. Claude Sonnet 5.5 is the baseline every other model is compared to.

Status by build stage:

| # | Stage | Status |
|---|-------|--------|
| 1 | Skeleton, `models.yaml`, LiteLLM layer, `benchlab check`, cost/latency logging | **done** |
| 2 | Inspect-based benches: GPQA Diamond, BFCL, GAIA/BrowseComp | **done** (proved end to end on BFCL; GPQA/GAIA need network/Docker not available in the build sandbox) |
| 3 | τ²-bench | planned |
| 4 | Harbor: Terminal-Bench, SWE-bench (Docker) | planned |
| 5 | Aider Polyglot, LiveCodeBench, RULER | planned |
| 6 | Retrieval pipeline (MTEB, BEIR, own data) | planned |
| 7 | Serving benchmark (`vllm bench serve`) + TTFT probe | planned |
| 8 | `benchlab report` → `report.html`, `results/tracker.json` | planned |
| 9 | Optional: Toolathlon, AutomationBench, OSWorld | planned |

## Setup

```bash
uv sync                      # Python 3.11+, installs benchlab + LiteLLM (SDK and proxy)
cp .env.example .env         # put API keys here; .env is git-ignored
uv run benchlab models --keys   # validate models.yaml and see which keys are set
```

Official harnesses are **not** installed into this environment: each one gets
its own `uv` venv under `harnesses/<name>/.venv`, created on first use by the
adapter's `prepare()` with exact pins (recorded in `harnesses/<name>/manifest.json`
and in every run's `config.json`). This is necessary, not cosmetic: e.g.
`inspect-ai` requires `openai>=3.4` while LiteLLM pins `openai<3`.

Docker is required for everything that executes model-written code
(Terminal-Bench, SWE-bench, Aider, LiveCodeBench, GAIA). Stage 1 needs no Docker.

## How to add a model

Add one entry to `models.yaml`:

```yaml
  - id: glm                       # short id: run folders, results.csv, tracker keys (<id>__<bench>)
    name: GLM-5.3
    provider: openai_compatible   # openai_compatible | anthropic | bedrock
    base_url: https://.../v1
    model: glm-5.3
    api_key_env: GLM_API_KEY      # the key is read from this env var (.env); never stored in files
    self_hosted: false
    price_in_per_mtok: 0.0        # USD; 0 = unknown -> cost shows as n/a instead of a fake 0
    price_out_per_mtok: 0.0
    max_concurrency: 8
    extra_params: {temperature: 0.0}
```

Then:

```bash
uv run benchlab check --model glm
```

`check` sends one chat request, one tool-call request and one streamed request
and prints latency, TTFT, whether tool calling works (and *how* it failed:
not used / malformed JSON / wrong tool / error), the token usage returned and
the cost of each probe. It exits non-zero on any problem, so a broken endpoint
fails in seconds, not after an hour.

Notes on fields:

* `extra_params: {temperature: null}` **removes** the global default for that
  model. Use it for models that reject sampling parameters (Claude Sonnet 5.5
  returns 400 for non-default `temperature`). Every such difference is recorded
  as a *per-model exception* in the run's `config.json` and shown by `check`.
* `kind: embedding | reranker` marks retrieval models; they are excluded from
  `--model all` and run through the separate retrieval pipeline (stage 6).
* `suites: [core, coder]`, `extras: [glm_aider]` control which benchmarks a
  model gets. Chat models default to `core`.
* `published: {gpqa: 80.1}` feeds the "suspicious results" list in the report.
* `tracker_id` lets two entries (e.g. Step 5 via API and self-hosted) report
  under one model id; `compare_to` prints the gap and the self-hosting hints.
* Provider mapping: `openai_compatible` → LiteLLM `openai/<model>` with
  `api_base`; `anthropic` → `anthropic/<model>`; `bedrock` → `bedrock/<model>`
  with `region` / `AWS_REGION_NAME` and the standard AWS credential chain.

## Architecture

```
models.yaml ──► benchlab.config (pydantic)
                   │
                   ├─► benchlab.llm      LiteLLM SDK: check, TTFT probe, retrieval, budget guard
                   │                      - usage.jsonl per run (tokens, cost, latency, TTFT, error class)
                   │
                   └─► benchlab.proxy    LiteLLM proxy config: one OpenAI-compatible endpoint
                        benchlab proxy    http://127.0.0.1:4000/v1, model name == benchlab id
                                          - every harness points OPENAI_BASE_URL at it
                                          - benchlab.proxy_logger writes the same usage.jsonl
adapters/<bench>.py   prepare() | plan() | run() | parse()   ── wraps the OFFICIAL harness
runs/<ts>_<model>_<bench>/   config.json  state.json (resume)  usage.jsonl  raw/  result.json
results/results.csv   one row per (model, bench, metric)
```

* **One access layer.** Direct requests go through `litellm.completion`;
  harness requests go through the LiteLLM proxy generated from the same
  `models.yaml` (`benchlab proxy-config`, `benchlab proxy`). Prices from
  `models.yaml` are passed to LiteLLM as `input_cost_per_token` /
  `output_cost_per_token`, so cost is computed identically on both paths.
* **Usage log.** `runs/<run>/usage.jsonl` holds one record per request:
  status (`ok` / `timeout` / `rate_limit` / `auth` / `error` /
  `malformed_tool_call`), latency, TTFT (streamed requests only), tokens, cost.
  Timeouts, API errors and malformed tool calls are therefore counted
  separately from wrong answers. `benchlab usage` summarizes it.
* **Keys.** Read from the env var named in `api_key_env`, at call time. The
  generated proxy config uses `os.environ/VAR` references; `config.json`
  stores only the var *name*; every logged string passes through
  `benchlab.redact`.
* **Runs are resumable.** `state.json` tracks completed task ids; re-running
  the same command continues the newest unfinished run for (model, bench).
* **Budget.** `--budget-usd` tails the run's usage log and aborts cleanly
  (progress saved) when the spend passes the cap. `--dry-run` prints what
  would run and a rough cost estimate from per-task token profiles.

## Stage 2: Inspect-based benchmarks (GPQA Diamond, BFCL, GAIA, BrowseComp)

Harness: [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals)
0.24.0 on `inspect-ai` 0.3.279 (pinned in `benchlab/adapters/inspect_bench.py`).
Every run starts a per-run LiteLLM proxy and calls
`inspect eval-set <task> --model openai/<id> --model-base-url <proxy>` with a
fixed `--log-dir` inside the run folder, which is Inspect's own resumable mode:
re-running the same command continues an interrupted run. The `.eval` log is
the raw artifact; `result.json` is parsed from it (metrics come from Inspect's
scorers, never recomputed).

| Bench | Task | Default subset | Notes |
|---|---|---|---|
| `gpqa` | `inspect_evals/gpqa_diamond` | all 198 | task defaults: CoT on, choices shuffled with seed 42; the task's default of **4 epochs** is overridden to `--trials` (1) |
| `bfcl` | `inspect_evals/bfcl` | `single_turn` = 13 V1/V2 AST + relevance categories, 3,641 samples | `--subset leaderboard` adds the four V3 multi-turn categories (+800, much longer); `--subset multi_turn`; or `--subset simple_python+irrelevance`. Per-category accuracies are in `result.json`; the primary number is Inspect's sample-weighted accuracy, with the leaderboard-style unweighted mean as `accuracy_unweighted_categories`. exec_* (retired upstream), sql and multi_turn_composite are excluded by default |
| `gaia` | `inspect_evals/gaia` | fixed 50-question subset of the 165 validation questions, seed 20261013, saved to `subsets/gaia_validation_50.json` on first run | needs `HF_TOKEN` with access to the gated `gaia-benchmark/GAIA` dataset and Docker (sandbox). `--subset 165` = full validation |
| `bcomp` | `inspect_evals/browse_comp` | first 100 | alternative to GAIA, run with `--bench bcomp`; needs `TAVILY_API_KEY` and Docker. The grader is the **fixed** `settings.grader_model` (default: the baseline) through the proxy, never the model under test |

What counts as an error: Inspect runs with `--no-fail-on-error`, so a sample
that times out or hits an API error is logged with an error instead of
aborting the run. `result.json` reports the harness accuracy (over scored
samples) *and* `accuracy_errors_as_wrong`, plus timeouts / API errors /
malformed tool calls separately; an error rate above 2% raises a warning.

Verified in this build: BFCL end to end (`benchlab run --model <id> --bench bfcl --smoke`)
against a mock OpenAI endpoint, including resume and parsing. GPQA's dataset
host (`openaipublic.blob.core.windows.net`) and HuggingFace were not reachable
from the build sandbox and Docker was not running, so GPQA/GAIA/BrowseComp are
verified by command-building and parser tests only; run `--smoke` on them first.

## Commands

```bash
benchlab models [--keys]                  # validate models.yaml
benchlab check --model glm|all            # endpoint / tool-calling / usage probe
benchlab proxy-config                     # write litellm_config.yaml (no secrets inside)
benchlab proxy                            # run the LiteLLM proxy in the foreground
benchlab run --model glm --suite core --stage 1
benchlab run --model all --bench bfcl,gpqa [--trials 3] [--budget-usd 20] [--dry-run] [--smoke]
benchlab runs                             # list run folders and their status
benchlab usage                            # tokens / cost / latency / error rates per model
benchlab report                           # stage 8
```

`--model all` means every chat model. `--stage N` runs that stage's benches for
every model that has them; `--suite` and `--bench` intersect with it.

## How to run each stage

| Stage | Command | What it runs |
|---|---|---|
| 1 | `benchlab run --model all --stage 1` | GPQA Diamond + BFCL (cheap; catches broken setups) |
| 2 | `benchlab run --model all --stage 2 --trials 3` | τ²-bench airline/retail/telecom, 20 tasks/domain × 3 trials |
| 3 | `benchlab run --model all --stage 3` | Terminal-Bench then SWE-bench Verified, fixed 50-task subset (seeded once, ids saved and reused) |
| 4 | `benchlab run --model <ids> --stage 4` | full runs / extras, only for named models |

Always run `benchlab check --model all` first, and `--smoke` (2–3 tasks per
bench) before a full stage.

## Rough cost per stage per model

Filled in as each stage lands and real numbers come out of the usage logs.
Until then `benchlab run --dry-run` gives an estimate from per-task token
profiles (see `benchlab/registry.py`). For Claude Sonnet 5.5 at $2/$10 per
Mtok: GPQA Diamond ≈ $3–4 (198 q, CoT); BFCL `single_turn` ≈ $15–20 (3,641
short prompts with tool schemas); BFCL `multi_turn` adds roughly as much again
(long stateful conversations). Open-weight APIs are typically 5–20× cheaper.

## Hardware and Docker resources

| Bench | Needs |
|---|---|
| GPQA, BFCL, TTFT probe | nothing beyond API access |
| τ²-bench | API access for the model *and* the fixed user-simulator model |
| GAIA | Docker (Inspect sandbox), web access |
| Terminal-Bench, SWE-bench (Harbor) | Docker, ~4 CPU / 16 GB RAM per concurrent task, tens of GB of images |
| Aider Polyglot, LiveCodeBench | Docker |
| RULER | long-context endpoint (up to 128k) |
| Retrieval (MTEB / BEIR / own) | embedding + rerank endpoints or a GPU for sentence-transformers |
| Serving | runs `vllm bench serve` against the self-hosted box; `nvidia-smi` on it |

Details per benchmark are added with each stage.

## Tests

```bash
uv run pytest -q          # unit tests; the LiteLLM layer is exercised against an in-process mock OpenAI server
uv run ruff check . && uv run ruff format --check .
```

Later stages add parser tests on saved harness outputs under `tests/fixtures/`.
