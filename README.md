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
| 3 | τ²-bench (airline, retail, telecom) | **done** (proved end to end on airline against a mock endpoint) |
| 4 | Harbor: Terminal-Bench 2.0, SWE-bench Verified (Docker) | **done** (no Docker in the build sandbox: verified by command/parse tests; run `--smoke` on a Docker host first) |
| 5 | Aider Polyglot, LiveCodeBench, RULER | **done** for Aider + LiveCodeBench (structural verification: Docker/HF not available in the build sandbox); **RULER skipped** - see below |
| 6 | Retrieval pipeline (MTEB, BEIR, own data) | **done** (own-data path proved end to end with mteb's evaluator; HF datasets not reachable in the build sandbox) |
| 7 | Serving benchmark (guidellm / `vllm bench serve`) + TTFT probe | **done** (both proved end to end against a mock endpoint; GPU figures come from `nvidia-smi` on the serving box) |
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

## Stage 3: τ²-bench (airline, retail, telecom)

Harness: [`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench)
pinned to tag **v1.0.1** (`settings.tau2_ref`), installed the official way - a
git clone + `uv sync` under `harnesses/tau2/repo` (the PyPI package named
`tau2` is an unrelated project). Bench keys `t2air`, `t2ret`, `t2tel`
(`--stage 2`).

* **Comparability warning.** The repo is now τ³-bench: v1.0.0 shipped 75+ task
  fixes to airline/retail/banking. Scores from v1.x are therefore **not
  comparable with τ²-era published numbers** (model cards, the 2025 paper); only
  compare models run by BenchLab with the same `tau2_ref`. To reproduce the τ²
  task set, set `tau2_ref` to the last pre-τ³ tag (`v0.2.0`) - note the CLI of
  older tags may differ from what the adapter was written against.
* **Fixed subset.** Default 20 tasks per domain (`settings.tau2_tasks_per_domain`)
  drawn once with seed 20261013 from the `base` split (airline 50, retail 114,
  telecom 114 tasks) and saved to `subsets/tau2_<domain>_20.json`; every model
  gets the same ids. `--subset all` runs the whole base split, `--subset N`
  another fixed size.
* **Trials.** Default 3 (`trials_default` in the registry) so tau2 computes
  pass^1, pass^2, pass^3; `--trials K` overrides. `result.json` holds every
  pass^k, the tracker gets pass^1.
* **User simulator.** Always `settings.tau2_user_model` (default `claude`),
  with `settings.tau2_user_model_params` applied on top of that model's own
  params (Claude: no temperature). It is recorded in `config.json`
  (`harness.user_model`, `harness.user_llm_args`) and in every note. Both the
  agent and the simulator go through the per-run proxy, so their tokens and
  cost are logged separately: `cost_usd` is the model under test only,
  `user_sim_cost_usd` is reported alongside.
* **Scoring.** `pass^k`, average reward and termination counts come from tau2's
  own `compute_metrics()`. tau2 excludes `infrastructure_error` simulations
  from pass^k; BenchLab reports their count, plus timeouts / API errors /
  abnormal terminations (`max_steps`, `too_many_errors`, …) separately and
  warns when they exceed 2% / 30% of simulations.
* **Resume.** tau2's `--auto-resume` continues from `raw/tau2_<domain>/results.json`.
* **Smoke.** `--smoke` = 2 tasks, 1 trial, `--max-steps 20`.
* Other fixed settings: `--seed 300` (tau2 default), `--max-steps 200`,
  `--max-concurrency` = the model's `max_concurrency`.

Cost: a τ² task is a long multi-turn conversation (tens of k input tokens per
simulation). Rough estimate for Claude Sonnet 5.5: ~$0.15 per simulation ⇒ 3
domains × 20 tasks × 3 trials ≈ $25-30 for the agent, plus a similar amount for
the Claude user simulator *per evaluated model*.

## Stage 4: Terminal-Bench 2.0 and SWE-bench Verified via Harbor

Harness: [Harbor](https://github.com/laude-institute/harbor) pinned to **0.24.0**
in `harnesses/harbor/.venv` - the official harness for Terminal-Bench 2.0, with
SWE-bench Verified (and Aider Polyglot, LiveCodeBench, …) as registry datasets.
Bench keys `tb` and `swe` (`--stage 3`); `lag_tb21` for the Laguna extra.

* **Datasets** come from Harbor's registry (`registry.json` in the harbor repo):
  `terminal-bench@2.0` (89 tasks, git `laude-institute/terminal-bench-2`) and
  `swebench-verified@1.0` (500 tasks). `settings.tb_dataset` / `swe_dataset`
  pin the `name@version`; the dataset git commit is recorded in `config.json`.
  There is **no `terminal-bench@2.1` in the registry today**; `lag_tb21` will
  say so in `prepare()` until one appears (or point `tb21_dataset` elsewhere).
* **Agent**: Harbor's reference agent `terminus-2` for every model (same
  prompts/tools), model `openai/<id>` through the per-run proxy
  (`--ak api_base=…`), `temperature` from the model's params when it has one.
  `max_tokens` is deliberately *not* capped for these agentic tasks (recorded
  in the note). One attempt per task (`-k 1`).
* **Fixed 50-task subsets** (`settings.harbor_subset_size`), drawn once with
  seed 20261013 and committed: `subsets/tb_terminal-bench_2.0_50.json`,
  `subsets/swe_swebench-verified_1.0_50.json`. Every model runs the same ids
  (`-i <task>` filters). `--subset all` = whole dataset.
* **Scoring**: each task's own verifier (the dataset's tests) writes
  `verifier_result.rewards` in `<job>/<trial>/result.json`; BenchLab only
  aggregates. Primary metric `resolved` = solved / all trials in the subset
  (**an errored trial counts as unsolved**, which is how "% tasks solved" is
  reported); `resolved_excl_errors`, timeouts / API errors / environment
  errors are reported separately with a warning above 2%.
* **Resume**: a run whose `raw/jobs/<bench>/config.json` exists is continued
  with `harbor job resume -p …` (completed trials are kept).
* **Smoke**: `--smoke` = the first 2 subset tasks.

Hardware: Docker with ~4 CPU / 16 GB RAM per concurrent trial and tens of GB
for images; set the model's `max_concurrency` accordingly (it becomes
`-n`). Harbor refuses to start - even `--dry-run` - without a Docker daemon.

Cost (Claude Sonnet 5.5, rough): Terminal-Bench tasks run long agent loops,
~$1-3 per task ⇒ ~$50-150 for the 50-task subset; SWE-bench Verified is
similar or higher. Open-weight APIs: a fraction of that. Use `--budget-usd`.

## Stage 5: Aider Polyglot, LiveCodeBench, RULER

### Aider Polyglot (`aider`, `glm_aider`, `ml4_aider`)
Official harness: [`Aider-AI/aider`](https://github.com/Aider-AI/aider) `benchmark/`
pinned to tag **v0.86.2** (`settings.aider_ref`), exercises from
`Aider-AI/polyglot-benchmark` at a pinned commit (`settings.aider_polyglot_ref`).
`prepare()` clones both under `harnesses/aider/` and builds the
`benchlab-aider:<ref>` image from the repo's `benchmark/Dockerfile` (Python,
Go, Rust, Java, Node toolchains). Each run copies the exercises into
`raw/benchmarks/`, starts the per-run proxy and runs
`./benchmark/benchmark.py benchlab --model openai/<id> --edit-format diff --threads <max_concurrency> --tries 2 --cont`
inside the container with `OPENAI_API_BASE` pointing at the proxy
(`--network=host`; set `settings.aider_docker_network` to `bridge` and a
non-loopback proxy host on macOS). Scoring is aider's own: after the run
`benchmark.py --stats` is executed in the container and parsed (`pass_rate_2`,
the leaderboard number, is the primary metric; `pass_rate_1`,
`percent_cases_well_formed`, error outputs, malformed responses, syntax errors
and test timeouts are reported too), cross-checked against the per-exercise
`.aider.results.json` files. `--cont` makes re-runs resume. `--smoke` =
`--num-tests 3`. Same edit format (`diff`) for every model, recorded in the note.

### LiveCodeBench (`lcb`, `q27_lcb`)
Official repo [`LiveCodeBench/LiveCodeBench`](https://github.com/LiveCodeBench/LiveCodeBench)
pinned to commit `28fef95` (`settings.lcb_ref`), dataset
`livecodebench/code_generation_lite` **release_v6** (May 2023 - Apr 2025,
1055 problems; `settings.lcb_release_version`). The model under test is
registered through the repo's documented extension point (`lm_styles`,
`LMStyle.OpenAIChat`) at runtime by a wrapper - no harness code is edited -
and the OpenAI runner is pointed at the proxy with `OPENAI_KEY` /
`OPENAI_BASE_URL`. Sampling follows LCB's protocol (**n=10, temperature 0.2**,
`settings.lcb_n` / `lcb_temperature`), a deliberate, recorded deviation from
the global temperature 0. **Cut-off window:** set `knowledge_cutoff:
YYYY-MM-DD` on the model in `models.yaml`; it becomes `--start_date`, so only
problems released after the model's cut-off are used, and the actual
first/last contest dates are recorded in `result.json` and the note. Without
a cutoff the whole release is used and the note says so. Scoring is the
repo's evaluator (`--evaluate`; `pass@1` per problem, per-difficulty, the
harness summary's pass@k); `--continue_existing_with_eval` resumes. The
harness venv installs the repo's declared deps **except `vllm`/`torch`**
(GPU inference stack, never imported on the API path) - a documented install
deviation, not a code change. The evaluator executes generated code in local
subprocesses: run it on a box you don't mind that on (or inside a VM).

### RULER - skipped (not cleanly integrable)
`NVIDIA/RULER` deprecated its original pipeline; the current one
(`rulerv1-ns` branch) is NeMo-Skills with a cluster config and a model server it
launches itself (vLLM/SGLang/TRT-LLM, GPU only). The legacy OpenAI client
hardcodes Azure environment variables and a fixed model/length table, so it
cannot target an OpenAI-compatible endpoint without modifying the harness.
`lag_ruler` and `mimo_long` therefore resolve to `benchlab/adapters/ruler.py`,
which fails `prepare()` with that explanation instead of faking a result.
Options: run NeMo-Skills on the GPU box serving Laguna/MiMo (official, heavy);
NeMo-Skills' OpenAI-server mode against the proxy; or accept a RULER port in
another framework (violates the official-harness rule). Your call.

## Stage 6: retrieval (embedding + reranker)

Harness: [`mteb`](https://github.com/embeddings-benchmark/mteb) pinned to
**2.24.12** in `harnesses/retrieval/.venv` (brings sentence-transformers and
pytrec_eval). These models do not chat, so this pipeline calls the embedding /
rerank endpoints **directly** (mteb's `OpenAIAPIEncodeWrapper` ->
`<host>/v1/embeddings`, `OpenAIAPIRerankWrapper` -> `<host>/v1/rerank`, as
served by vLLM / TEI) or loads local weights (`base_url: local` ->
`SentenceTransformerEncoderWrapper` / `CrossEncoderWrapper`). No proxy, so cost
is reported as n/a (self-hosted). Run with `--model qemb`; the reranker is
`settings.reranker_model` (`qrr`).

| key | what | tracker |
|---|---|---|
| `mteb_ret` | MTEB(eng, v2) retrieval tasks (10, `settings.mteb_retrieval_tasks`), embedding only, mean nDCG@10 | `qemb__mteb_ret` |
| `mteb_rr` | MTEB(eng, v2) reranking tasks + SciDocsRR, scored by the embedding model (standard MTEB) and by the cross-encoder | `qemb__mteb_rr`, `qrr__mteb_rr` |
| `beir` | BEIR datasets as packaged by mteb (`settings.beir_tasks`: the 8 with corpora an API embedder can afford; `--subset all` = all 15), nDCG@10 embedding-only, then **re-rank the top-100** with the reranker (`convert_to_reranking`) | `qemb__beir`, `qemb__beir_rr` |
| `own` | your `retrieval/queries.jsonl` + `retrieval/docs/` through the same evaluator (custom `AbsTaskRetrieval`), recall@10 / nDCG@10 with and without the reranker | `qemb__own_rec`, `qemb__own_ndcg`, `qemb__own_rec_rr`, `qemb__own_ndcg_rr` |

Scoring is mteb's retrieval evaluator in every case (pytrec_eval nDCG@k /
recall@k); BenchLab only averages task scores. mteb's result cache lives in
the run folder, so a re-run resumes at task granularity. `--smoke` uses the
NanoBEIR tasks (retrieval) / SciDocsRR (reranking). Instruction prompts for
models that need them go in the model's `extra_params: {prompts: {query: "...",
document: "..."}}`. Verified here end to end on own data with deterministic
test models (`tests/test_retrieval_adapter.py`); MTEB/BEIR datasets come from
HuggingFace, which the build sandbox could not reach.

## Stage 7: serving benchmark and TTFT probe

### TTFT probe (`ttft`, every model, stage 1)

BenchLab's own measurement, not a harness: 10 fixed short prompts
(`benchlab/adapters/ttft.py`) × 3 streamed repeats, `max_tokens 64`, one
untimed warm-up, identical for every model, through the same LiteLLM layer as
`benchlab check`. TTFT = time to the first streamed content chunk.
`<id>__ttft` = **p50 seconds** (p95, mean, p50 latency and decode tokens/s in
`result.json`; failures counted as API errors; warning when p95 > 3× p50).
Resumable per sample; cost comes from the usage log (a few cents at most).

### Serving benchmark (`serving`, self-hosted models only, stage 4)

Harness: [guidellm](https://github.com/vllm-project/guidellm) **0.8.1** (the
vLLM project's load generator) in `harnesses/serving/.venv`; or `vllm bench
serve` (`settings.serving_tool: vllm`, needs the `vllm` package on the client,
command built from its CLI but not exercised here). The endpoint is hit
**directly** (no proxy, so no proxy latency in the numbers) at
`/v1/chat/completions`, one guidellm run per concurrency level
(`settings.serving_concurrencies`, default 1 / 8 / 32), `serving_requests_per_level`
(64) requests each, synthetic prompts of `serving_prompt_tokens` (512) →
`serving_output_tokens` (256) tokens generated with the model's HF tokenizer
(`serving.tokenizer` when the served name is not a HF id). The whole spec goes
into a guidellm scenario file; the API key is written only to a 0600 temp file
deleted after the run (a redacted copy of the scenario is kept in `raw/`).
Set `settings.serving_prompts_file` (JSONL `{"prompt": ...}`) to use a fixed
prompt set instead of synthetic text, the same for every model.

| key | value | from |
|---|---|---|
| `<id>__srv_ttft` | TTFT median at concurrency 1, seconds (p95 in `result.json`) | `time_to_first_token_ms.successful.median` |
| `<id>__srv_tps` | tokens/s per request at concurrency 1 = 1000 / median inter-token latency | `inter_token_latency_ms.successful.median` |
| `<id>__srv_c1`, `srv_c8`, `srv_c32` | total output tokens/s at that concurrency | `output_tokens_per_second.successful.mean` |
| `<id>__srv_mem`, `srv_gpus` | GPU memory used (GB) and GPU count | `nvidia-smi` (`serving.nvidia_smi: local` or `ssh:<host>`), else `serving.gpu_mem_gb` / `serving.gpus` |
| `<id>__srv_quant` | quantization | `serving.quantization` in models.yaml (vLLM does not expose it over the API) |
| `<id>__srv_match` | self-hosted vs API score gap | computed by `benchlab report` (stage 8) |

Per-level results stay as `raw/c<N>.json`, so an interrupted run resumes at the
next level. Errored requests are counted as API errors and flagged. `--smoke` =
concurrency 1, 4 requests. Verified end to end against the mock endpoint
(`tests/test_serving_adapters.py`); synthetic prompts need the HF tokenizer,
which the build sandbox could not download, so the fixed-prompt path was used
there.

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
benchlab run --model all --bench ttft      # TTFT probe, every model (stage 1)
benchlab run --model all --bench serving   # serving benchmark, self-hosted models only
benchlab report                           # stage 8
```

`--model all` means every chat model. `--stage N` runs that stage's benches for
every model that has them; `--suite` and `--bench` intersect with it.

## How to run each stage

| Stage | Command | What it runs |
|---|---|---|
| 1 | `benchlab run --model all --stage 1` | GPQA Diamond + BFCL + TTFT probe (cheap; catches broken setups) |
| 2 | `benchlab run --model all --stage 2` | τ²-bench airline/retail/telecom, 20 tasks/domain × 3 trials (trials default to 3 for τ²) |
| 3 | `benchlab run --model all --stage 3` | Terminal-Bench 2.0 then SWE-bench Verified, fixed 50-task subsets (committed under `subsets/`) |
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
| Serving | guidellm (default) or `vllm bench serve` from any box that can reach the self-hosted endpoint; `nvidia-smi` on the serving box (local or over ssh) |

Details per benchmark are added with each stage.

## Tests

```bash
uv run pytest -q          # unit tests; the LiteLLM layer is exercised against an in-process mock OpenAI server
uv run ruff check . && uv run ruff format --check .
```

Later stages add parser tests on saved harness outputs under `tests/fixtures/`.
