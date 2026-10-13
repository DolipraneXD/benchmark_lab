"""models.yaml loading and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from benchlab.redact import register_secret_env

Provider = Literal["openai_compatible", "anthropic", "bedrock"]
Kind = Literal["chat", "embedding", "reranker"]

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class MissingKeyError(RuntimeError):
    pass


class ModelSpec(BaseModel):
    id: str
    name: str
    provider: Provider
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    kind: Kind = "chat"
    self_hosted: bool = False
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    max_concurrency: int = Field(default=4, ge=1)
    extra_params: dict[str, Any] = Field(default_factory=dict)
    region: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    suites: list[str] | None = None
    extras: list[str] = Field(default_factory=list)
    published: dict[str, float] = Field(default_factory=dict)
    tracker_id: str | None = None
    compare_to: str | None = None
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _id_ok(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"model id {v!r} must match {_ID_RE.pattern}")
        return v

    @model_validator(mode="after")
    def _provider_requirements(self) -> ModelSpec:
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError(f"model {self.id}: openai_compatible needs base_url")
        if self.provider in ("openai_compatible", "anthropic") and not self.api_key_env:
            raise ValueError(f"model {self.id}: {self.provider} needs api_key_env")
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
        if not v and required and self.provider != "bedrock":
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
    baseline_model: str = "claude"
    litellm_proxy_port: int = 4000
    litellm_proxy_host: str = "127.0.0.1"
    default_temperature: float | None = 0.0
    default_max_tokens: int = 4096
    request_timeout_s: float = 300
    tau2_user_model: str = "claude"
    tau2_user_model_params: dict[str, Any] = Field(default_factory=dict)
    grader_model: str | None = None  # LLM grader for model-graded benches (default: baseline)
    tau2_ref: str = "v1.0.1"  # git tag/commit of sierra-research/tau2-bench to pin
    tau2_tasks_per_domain: int = 20  # fixed seeded subset of the `base` split per domain
    tau2_max_steps: int = 200  # tau2 default
    runs_dir: str = "runs"
    results_dir: str = "results"


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
        return [m for m in self.models if m.kind == "chat"]

    def select(self, selector: str) -> list[ModelSpec]:
        """'all' -> every chat model; otherwise a comma-separated list of ids."""
        if selector == "all":
            return self.chat_models()
        return [self.get(x.strip()) for x in selector.split(",") if x.strip()]

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
