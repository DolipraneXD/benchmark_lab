"""models.yaml loading and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from benchlab import paths, providers
from benchlab.redact import register_secret_env

Kind = Literal["chat", "embedding", "reranker"]

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class MissingKeyError(RuntimeError):
    pass


class ModelSpec(BaseModel):
    id: str
    name: str
    provider: str  # a key of benchlab.providers.PROVIDERS
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    kind: Kind = "chat"
    self_hosted: bool = False
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    max_concurrency: int = Field(default=4, ge=1)
    extra_params: dict[str, Any] = Field(default_factory=dict)
    region: str | None = None  # bedrock
    api_version: str | None = None  # azure
    extra_headers: dict[str, str] = Field(default_factory=dict)
    suites: list[str] | None = None
    extras: list[str] = Field(default_factory=list)
    published: dict[str, float] = Field(default_factory=dict)
    tracker_id: str | None = None
    compare_to: str | None = None
    knowledge_cutoff: str | None = None  # YYYY-MM-DD; LiveCodeBench uses problems after it
    reference: bool = False  # published scores only: shown in the report, never called or run
    serving: dict[str, Any] = Field(
        default_factory=dict
    )  # self-hosted: {quantization, gpus, gpu_mem_gb, nvidia_smi: local|ssh host}
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _id_ok(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"model id {v!r} must match {_ID_RE.pattern}")
        return v

    @model_validator(mode="after")
    def _provider_requirements(self) -> ModelSpec:
        prov = providers.get(self.provider)  # unknown provider -> ValueError
        if prov.base_url == "required" and not self.base_url:
            raise ValueError(f"model {self.id}: {self.provider} needs base_url")
        if self.reference:
            self.suites = []
            return self
        if prov.key_required and not self.api_key_env:
            raise ValueError(f"model {self.id}: {self.provider} needs an API key (api_key_env)")
        if prov.needs_api_version and not self.api_version:
            raise ValueError(f"model {self.id}: {self.provider} needs api_version")
        for k in ("api_key", "key", "token"):
            if k in self.model_fields_set:  # pragma: no cover - defensive
                raise ValueError("never put keys in models.yaml; use api_key_env")
        if self.suites is None:
            self.suites = {"chat": ["core"], "embedding": ["retrieval"], "reranker": ["retrieval"]}[
                self.kind
            ]
        register_secret_env(self.api_key_env)
        return self

    # -- helpers ----------------------------------------------------------
    @property
    def tracker(self) -> str:
        return self.tracker_id or self.id

    @property
    def has_prices(self) -> bool:
        return self.price_in_per_mtok > 0 or self.price_out_per_mtok > 0

    def api_key(self, required: bool = True) -> str | None:
        """Read the key from the environment at call time. Never cached, never logged."""
        if not self.api_key_env:
            return None
        v = os.environ.get(self.api_key_env)
        if not v and required:  # the model names a key variable, so it needs it
            raise MissingKeyError(
                f"model {self.id}: env var {self.api_key_env} is not set (put it in .env)"
            )
        return v or None

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float | None:
        if not self.has_prices:
            return None
        return tokens_in * self.price_in_per_mtok / 1e6 + tokens_out * self.price_out_per_mtok / 1e6

    def public_dict(self) -> dict[str, Any]:
        """Everything about the model that is safe to write into a run's config.json."""
        d = self.model_dump()
        d["api_key_env"] = self.api_key_env  # the *name* only
        return d


class Settings(BaseModel):
    profile: Literal["full", "lite"] = (
        "full"  # lite = same benches, smaller fixed subsets (see LITE_PROFILE)
    )
    baseline_model: str = "claude"
    litellm_proxy_port: int = 4000
    litellm_proxy_host: str = "127.0.0.1"
    default_temperature: float | None = 0.0
    default_max_tokens: int = 4096
    request_timeout_s: float = 300
    # rate limits, 5xx and dropped connections are retried with growing waits (2, 4, 8, 16 s)
    request_retries: int = 4
    retry_backoff_s: float = 2.0
    tau2_user_model: str = "claude"
    tau2_user_model_params: dict[str, Any] = Field(default_factory=dict)
    grader_model: str | None = None  # LLM grader for model-graded benches (default: baseline)
    tau2_ref: str = "v1.0.1"  # git tag/commit of sierra-research/tau2-bench to pin
    tau2_tasks_per_domain: int = 20  # fixed seeded subset of the `base` split per domain
    tau2_trials: int | None = None  # trials when --trials is not given (default: 3 for pass^k)
    bfcl_subset: str | None = (
        None  # BFCL category set when --subset is not given (default single_turn)
    )
    gaia_subset_size: int = 50  # fixed seeded subset of the GAIA validation split
    tau2_max_steps: int = 200  # tau2 default
    harbor_subset_size: int = 50  # fixed seeded subset for Terminal-Bench / SWE-bench
    tb_dataset: str = "terminal-bench@2.0"  # harbor registry name@version
    swe_dataset: str = "swebench-verified@1.0"
    aider_ref: str = "v0.86.2"  # Aider-AI/aider tag (benchmark harness + Docker image)
    aider_polyglot_ref: str = (
        "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"  # Aider-AI/polyglot-benchmark
    )
    aider_edit_format: str = "diff"  # same edit format for every model (leaderboard default)
    aider_tries: int = 2  # pass_rate_2 is the leaderboard number
    aider_docker_network: str = "host"  # how the container reaches the proxy (host on Linux)
    lcb_ref: str = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"  # LiveCodeBench/LiveCodeBench commit
    lcb_release_version: str = "release_v6"  # problems May 2023 - Apr 2025
    lcb_n: int = 10  # LCB official: n=10 samples, temperature 0.2 (pass@1 estimate)
    lcb_temperature: float = 0.2
    lcb_default_start_date: str | None = None  # used when a model has no knowledge_cutoff
    # retrieval pipeline (mteb 2.x is the harness for MTEB, BEIR and own data)
    reranker_model: str = "qrr"  # model id (kind: reranker) used for the "+ reranker" variants
    rerank_top_k: int = 100  # rerank the top-100 of the embedding stage
    mteb_retrieval_tasks: list[str] = Field(
        default_factory=lambda: [  # MTEB(eng, v2) retrieval
            "ArguAna",
            "CQADupstackGamingRetrieval",
            "CQADupstackUnixRetrieval",
            "ClimateFEVERHardNegatives",
            "FEVERHardNegatives",
            "FiQA2018",
            "HotpotQAHardNegatives",
            "SCIDOCS",
            "TRECCOVID",
            "Touche2020Retrieval.v3",
        ]
    )
    mteb_reranking_tasks: list[str] = Field(
        default_factory=lambda: [  # MTEB(eng, v2) reranking + SciDocsRR
            "AskUbuntuDupQuestions",
            "MindSmallReranking",
            "SciDocsRR",
        ]
    )
    beir_tasks: list[str] = Field(
        default_factory=lambda: [  # BEIR datasets with corpora an API embedder can afford
            "SciFact",
            "NFCorpus",
            "FiQA2018",
            "ArguAna",
            "SCIDOCS",
            "TRECCOVID",
            "Touche2020",
            "CQADupstackRetrieval",
        ]
    )  # `--subset all` = mteb's full BEIR benchmark (adds NQ, HotpotQA, FEVER, ClimateFEVER, DBPedia, Quora, MSMARCO)
    own_retrieval_dir: str = "data/retrieval"  # queries.jsonl + docs/
    # serving benchmark (guidellm against the self-hosted endpoint) and the TTFT probe
    serving_tool: str = (
        "guidellm"  # guidellm | vllm  (vllm = `vllm bench serve`, needs vllm installed)
    )
    serving_concurrencies: list[int] = Field(default_factory=lambda: [1, 8, 32])
    serving_requests_per_level: int = 64  # requests per concurrency level
    serving_prompt_tokens: int = 512  # synthetic prompt length
    serving_output_tokens: int = 256  # synthetic output length
    serving_prompts_file: str | None = (
        None  # JSONL {"prompt": ...} to use instead of synthetic text (same file for every model)
    )
    ttft_prompts: int = 10  # fixed prompt set size for the TTFT probe (all models)
    ttft_repeats: int = 3  # streamed repeats per prompt
    apptasks_dir: str = "data/evals"  # app tasks: your own cases, *.yaml
    # AutomationBench (opt-in extra: add `autobench` to a model's extras)
    autobench_ref: str = (
        "4a8e1061254004d9dac807054eed33fad7d1ff14"  # zapier/AutomationBench (v1.0.6)
    )
    autobench_toolset: str = "api"  # api | zapier | limited_zapier (README default: api)
    autobench_max_steps: int = 50
    autobench_domains: list[str] | None = None  # default: the 6 public scored domains
    # report / tracker (stage 8)
    sanity_threshold_pts: float = (
        3.0  # |ours - published| above this is flagged in the suspicious list
    )
    self_hosted_gap_pts: float = (
        5.0  # self-hosted below its API counterpart by more than this -> hint
    )
    suspicious_error_rate: float = (
        0.05  # (timeouts + API errors + malformed tool calls) / tasks above this -> flagged
    )
    # how the panel starts runs: local = a subprocess here; docker = one container per run
    # (BENCHLAB_EXECUTOR overrides it, which is what docker/compose.yaml does)
    executor: Literal["local", "docker"] = "local"
    docker_network: str = "benchlab"
    docker_image_prefix: str = "benchlab"
    docker_image_tag: str = "latest"
    container_limits: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "core": {"cpus": 2, "memory": "4g"},
            "code": {"cpus": 4, "memory": "8g"},
            "retrieval": {"cpus": 4, "memory": "8g"},
            "serving": {"cpus": 2, "memory": "4g"},
        }
    )
    # under the workspace (BENCHLAB_WORKSPACE moves it, e.g. inside a run container)
    runs_dir: str = Field(default_factory=lambda: str(paths.workspace() / "runs"))
    results_dir: str = Field(default_factory=lambda: str(paths.workspace() / "results"))

    @model_validator(mode="after")
    def _apply_profile(self) -> Settings:
        """`profile: lite` sets the LITE_PROFILE knobs unless the user set them explicitly."""
        if self.profile == "lite":
            for k, v in LITE_PROFILE.items():
                if k not in self.model_fields_set:
                    setattr(self, k, v)
        return self


LITE_PROFILE: dict[str, Any] = {
    # same benchmarks, smaller fixed subsets: roughly 1/5 of the full cost
    "bfcl_subset": "lite",  # 9 single-turn categories, 1064 samples (not the leaderboard mix)
    "tau2_tasks_per_domain": 10,
    "tau2_trials": 1,  # pass^1 only; no pass^k
    "harbor_subset_size": 20,  # Terminal-Bench / SWE-bench
    "gaia_subset_size": 20,
    "lcb_n": 1,  # one sample per problem instead of 10
    "autobench_domains": ["sales"],  # 100 tasks instead of 600
    "serving_requests_per_level": 32,
}


class BenchConfig(BaseModel):
    settings: Settings = Field(default_factory=Settings)
    models: list[ModelSpec]

    @model_validator(mode="after")
    def _unique_ids(self) -> BenchConfig:
        seen: set[str] = set()
        for m in self.models:
            if m.id in seen:
                raise ValueError(f"duplicate model id {m.id}")
            seen.add(m.id)
        for ref in (
            self.settings.baseline_model,
            self.settings.tau2_user_model,
            self.settings.grader_model,
        ):
            if ref is not None and ref not in seen:
                raise ValueError(f"settings reference unknown model id {ref!r}")
            if ref is not None and self.get(ref).reference:
                raise ValueError(
                    f"{ref!r} is a reference model (published scores only); "
                    "baseline, user simulator and grader must be benchmarked models"
                )
        for m in self.models:
            if m.compare_to and m.compare_to not in seen:
                raise ValueError(f"model {m.id}: compare_to {m.compare_to!r} is unknown")
        return self

    def get(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model id {model_id!r}; known: {[m.id for m in self.models]}")

    def chat_models(self) -> list[ModelSpec]:
        """Chat models that are benchmarked (reference models excluded)."""
        return [m for m in self.models if m.kind == "chat" and not m.reference]

    def reference_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.reference]

    def select(self, selector: str) -> list[ModelSpec]:
        """'all' -> every benchmarked chat model; otherwise a comma-separated list of ids."""
        if selector == "all":
            return self.chat_models()
        out = [self.get(x.strip()) for x in selector.split(",") if x.strip()]
        refs = [m.id for m in out if m.reference]
        if refs:
            raise ValueError(
                f"{refs} are reference models (published scores only); not benchmarked"
            )
        return out

    @property
    def baseline(self) -> ModelSpec:
        return self.get(self.settings.baseline_model)


def find_config_path(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("BENCHLAB_MODELS")
    if env:
        return Path(env)
    return Path.cwd() / "models.yaml"


def load_config(path: str | Path | None = None, dotenv: bool = True) -> BenchConfig:
    p = find_config_path(path)
    if dotenv:
        # .env next to models.yaml, then cwd; never override real env vars
        for candidate in (p.parent / ".env", Path.cwd() / ".env"):
            if candidate.exists():
                load_dotenv(candidate, override=False)
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    if "models" not in raw:
        raise ValueError(f"{p}: missing top-level 'models' list")
    return BenchConfig.model_validate(raw)
