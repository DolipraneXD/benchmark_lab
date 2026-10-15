# BenchLab

A command-line runner that executes a fixed benchmark suite against any model
(API endpoint + model name + API key) and produces comparable, reproducible
results. Every benchmark wraps its **official harness**; BenchLab never scores
anything itself. Mistral Large 4 (`ml4`) is the baseline every other model is compared to
(`settings.baseline_model`); it also plays the τ²-bench user simulator, and
`dsflash` grades BrowseComp. A model is never graded by itself: `benchlab run`
refuses a model-graded bench when the model under test is the grader.

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
| 8 | `benchlab report` → `report.html`, `results/tracker.json` | **done** |
| 9 | Optional: AutomationBench, Toolathlon, OSWorld | **done** (AutomationBench proved end to end; Toolathlon and OSWorld wrap the official runners but need a host you set up: app stack + accounts, or KVM/VMs) |

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
* **User simulator.** Always `settings.tau2_user_model` (`ml4` in models.yaml),
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

## Stage 8: `benchlab report` (report.html + tracker.json)

`benchlab report` reads every finished run under `runs/` (newest non-smoke
run per model × bench wins; a smoke run is used only when nothing else exists
and is flagged), and writes:

* `results/tracker.json` - `{"<modelId>__<benchKey>": {"v": "...", "note": "..."}}`.
  Adapter entries are merged as written (scores one decimal, TTFT seconds,
  cost USD; notes carry harness, subset size, trials, deviations). The report
  derives the remaining keys:

  | key | value |
  |---|---|
  | `<id>__cost` | total USD over the core benches that have a result, from the usage logs |
  | `claude__coder_lcb/aider/swe/tb` | copies of the baseline's coder-suite scores |
  | `gptoss__gpt_sanity` | worst deviation (points) vs the model's `published:` scores; note starts with `OFF by >3 pts` when any bench is off by more than `settings.sanity_threshold_pts` |
  | `<id>__srv_match` | self-hosted minus API counterpart (`compare_to`) in points, mean over common core benches; falls back to `published`; n/a otherwise |
  | `step5__s5_api` / `s5_self` | mean of core scores on the API / self-hosted entry over the benches both have, with the gap in the note |
  | `step5__s5_tok` | output tokens per task-attempt self-hosted, with the API figure and ratio in the note |
  | `dsflash__ds_cps` / `ds_lps` | USD and wall seconds per solved core task (solved = tasks × trials × score) |
  | not-integrated benches (`mimo_tool`, `mimo_osw`, `ds_auto`) | `n/a` with the reason |

* `report.html` - one static page: models × benchmarks with the delta against
  the baseline (green/red; TTFT lower-is-better), n/trials/cost/errors under
  each cell, cost per task and tokens per task, the serving table
  (quant/GPUs/memory/TTFT/tok/s/throughput/vs-API), retrieval results, the
  **suspicious results** list, the derived keys with their notes, and the run
  index (run folder, date, harness version, note) for every number shown.
* `results/suspicious.json` - the same list as data.

Suspicious-result rules: only a smoke run; error rate (timeouts + API errors
+ malformed tool calls) above `settings.suspicious_error_rate` (5%); a 0 or
100 score on ≥5 tasks; no cost on a priced model (harness bypassed the proxy);
any adapter warning; more than `sanity_threshold_pts` (3) from a published
score; a self-hosted model more than `self_hosted_gap_pts` (5) below its API
counterpart, or 20+ points below the baseline on a tool-calling bench (hint:
chat template, `--tool-call-parser`, quantization).

## Stage 9 (optional): AutomationBench, Toolathlon, OSWorld

All three wrap the official harness, pinned by commit in `settings`; none of them
re-implements scoring. Each is an extra (`mimo_tool`, `mimo_osw`, `ds_auto`), so
add the key to a model's `extras:` to run it on another model.

### AutomationBench (`ds_auto`) - verified

[zapier/AutomationBench](https://github.com/zapier/AutomationBench) is public
(MIT, white paper arXiv 2604.18934); the repo ships the **public** 600-task set
(6 domains × 100) plus a 200-task `simple` warm-up domain that is not scored.
The official leaderboard uses a held-out private set, so local scores agree
directionally only (the note says so). `uv sync` into
`harnesses/automationbench/repo` (Python 3.13), then `auto-bench --model <id>
--base-url <proxy> --api-key-var BENCHLAB_PROXY_KEY --domains sales,...,hr
--max-steps 50 --max-concurrent <max_concurrency> --toolset api --export-json`.
Score = strict pass rate (every assertion satisfied) from the export; partial
credit, tool calls and steps per task, per-domain pass rates, aborted / errored
tasks are reported alongside. Cost comes from the proxy usage log. `--subset
sales,hr` restricts domains; `--smoke` = 3 sales tasks. Verified end to end
through the proxy against the mock (`tests/test_stage9_adapters.py`). For
`dsflash`, `ds_cps` / `ds_lps` in the report use this run when it exists.

### Toolathlon-Verified (`mimo_tool`) - official runner, host setup is yours

[hkust-nlp/Toolathlon](https://github.com/hkust-nlp/Toolathlon) (108 tasks, 32
apps, 604 MCP tools) needs, per its README: Docker/Podman and the task image,
`global_preparation/deploy_containers.sh` (local app stack), Node 22, and
accounts/tokens for the external services in `configs/token_key_session.py`
(`global_preparation/how2register_accounts.md`). BenchLab clones the pinned
commit into `harnesses/toolathlon/repo` and refuses to run until
`bash global_preparation/install_env_minimal.sh false` and
`bash global_preparation/check_installation_containerized.sh` have been run there
and you `touch .benchlab_setup_verified` in the repo. The run is the official
`scripts/run_parallel.sh <id> <dump> unified <workers>` with
`TOOLATHLON_OPENAI_BASE_URL/_API_KEY` = BenchLab's proxy; score = the harness'
`eval_stats.json` success rate; timeouts / run failures / preprocess failures
are reported separately and count as failed. The harness fixes max_tokens 65536,
50 turns, 5400 s per task and provider-default sampling: recorded as a deviation.
The authors' public evaluation service is **not** used because it would send your
model API key to a third-party server. Not exercised here (no Docker in the build
sandbox): command building and parsing are tested on a synthetic `eval_stats.json`.

### OSWorld-Verified (`mimo_osw`) - needs VMs

[xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld), 369 tasks
(`test_all.json`; `settings.osworld_meta: test_nogdrive.json` = 361 when Google
Drive is not configured, which must be reported). The docker provider needs
KVM (`/dev/kvm`); `prepare()` refuses otherwise. Official
`scripts/python/run_multienv.py --provider_name docker --observation_type
screenshot --max_steps 15 --num_envs <max_concurrency>`, results read exactly as
`show_result.py` does (`result.txt` per task; re-running resumes unfinished
tasks). The baseline `PromptAgent` dispatches on the model-name prefix and only
the `gpt*` branch speaks plain OpenAI chat at `OPENAI_BASE_URL`, so BenchLab
registers the alias `gpt-benchlab-<id>` on the per-run proxy and passes that
name; the note records it. The model must accept screenshots. Harness-fixed
temperature 1.0 / top_p 0.9 / max_tokens 1500 are recorded as deviations.
Not exercised here (no KVM): command building and result parsing are tested on a
synthetic result tree.

## Cheap mode (`profile: lite`)

The full core suite costs roughly $160 per model at Claude Sonnet 5.5 prices
(BFCL 3,641 samples, τ² 20 tasks × 3 trials per domain, 50-task Terminal-Bench
and SWE-bench subsets, LiveCodeBench with 10 samples per problem). Set
`profile: lite` in `settings:` (or override any knob individually) to run the
same benchmarks on smaller **fixed, seeded** subsets, identical for every model:

| knob | full | lite |
|---|---|---|
| `bfcl_subset` | `single_turn` (3,641) | `lite`: 9 single-turn categories, 1,064 samples |
| `tau2_tasks_per_domain` × `tau2_trials` | 20 × 3 | 10 × 1 (pass^1 only) |
| `harbor_subset_size` (TB, SWE) | 50 | 20 |
| `gaia_subset_size` | 50 | 20 |
| `lcb_n` | 10 samples/problem | 1 |
| `autobench_domains` | 6 domains, 600 tasks | `sales`, 100 tasks |
| `serving_requests_per_level` | 64 | 32 |

Rough cost per model at Claude prices: stage 1 ≈ $9, stage 2 ≈ $4.5 (+ the
same again on the Claude user simulator), stage 3 ≈ $42, GAIA ≈ $3; LCB and
Aider ≈ $40. About a fifth of the full suite. Lite scores are **not** comparable
with full-profile or published numbers (different subset sizes and BFCL
category mix); every note records the subset, and the report flags published
comparisons on lite runs as deviations. Other savings that do not change the
subsets: point `tau2_user_model` at a cheaper strong model, and run `--stage 1`
on everything before paying for stages 2-3 on models that already look broken.

## Trial run through xkiro.com (`models.xkiro.yaml`)

`models.xkiro.yaml` points every chat model at one OpenAI-compatible reseller
endpoint (xkiro.com, free tier ~1M tokens/day) so the whole pipeline can be tried
with a single key (`XKIRO_API_KEY`). It uses `profile: lite` and writes to
`runs_xkiro/` and `results_xkiro/`, so trial numbers never mix with real ones.

```bash
curl -s https://api.xkiro.com/v1/models -H "Authorization: Bearer $XKIRO_API_KEY"  # copy exact ids into model:
uv run benchlab check -c models.xkiro.yaml -m all
uv run benchlab run   -c models.xkiro.yaml -m ml4 --stage 1 --smoke
uv run benchlab run   -c models.xkiro.yaml -m ml4 -b gpqa --budget-usd 0.9
```

Prices in that file are 1.0/1.0, so "USD" there means millions of tokens:
`--budget-usd 0.9` stops a run at ~0.9M tokens, under the daily quota, and the run
resumes the next day. Lite needs about 2.2M tokens per model for stage 1 and 1.9M
for stage 2, so the free tier covers roughly one bench per model per day; stage 3
(Terminal-Bench / SWE-bench) does not fit. Use these runs to test the setup, not
for the decision: a reseller can't prove it serves the official weights, and
quota errors surface as API errors in the error column.

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
benchlab report [--out report.html] [--no-smoke]   # report.html + results/tracker.json + results/suspicious.json
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
| AutomationBench | nothing beyond API access (simulated SaaS tools run in-process; Python 3.13 via uv) |
| Toolathlon | Docker/Podman, the Toolathlon task image, its locally deployed apps (poste.io, Canvas, k8s, ...), Node 22, real accounts/tokens for the external services |
| OSWorld-Verified | VMs: Docker + KVM on the host (or AWS/VMware per its README), tens of GB for the Ubuntu image, a model that accepts screenshots |
| Serving | guidellm (default) or `vllm bench serve` from any box that can reach the self-hosted endpoint; `nvidia-smi` on the serving box (local or over ssh) |

Details per benchmark are added with each stage.

## Tests

```bash
uv run pytest -q          # unit tests; the LiteLLM layer is exercised against an in-process mock OpenAI server
uv run ruff check . && uv run ruff format --check .
```

Later stages add parser tests on saved harness outputs under `tests/fixtures/`.
