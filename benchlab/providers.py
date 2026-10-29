"""The API providers a model can use. Each maps to a LiteLLM route prefix; LiteLLM does
the actual protocol work, so adding a provider here is all BenchLab needs.

``base_url``: "required", "optional" or "none" (the provider has one fixed endpoint).
``key``: whether ``api_key_env`` must be set. Bedrock uses the AWS credential chain and
local servers (Ollama, most self-hosted OpenAI-compatible endpoints) accept no key.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    key: str  # value of `provider:` in models.yaml
    label: str  # shown in the panel
    prefix: str  # LiteLLM route prefix
    base_url: str = "none"  # required | optional | none
    default_base_url: str | None = None
    key_required: bool = True
    example_model: str = ""
    needs_api_version: bool = False
    needs_region: bool = False


PROVIDERS: dict[str, Provider] = {
    p.key: p
    for p in (
        Provider(
            "openai_compatible",
            "OpenAI-compatible endpoint (vLLM, resellers, gateways)",
            "openai",
            base_url="required",
            key_required=False,
            example_model="minimax/minimax-m3.1-flash",
        ),
        Provider("openai", "OpenAI", "openai", example_model="gpt-4.1"),
        Provider(
            "anthropic",
            "Anthropic",
            "anthropic",
            base_url="optional",
            example_model="claude-sonnet-5-5",
        ),
        Provider(
            "mistral", "Mistral AI (La Plateforme)", "mistral", example_model="mistral-large-latest"
        ),
        Provider("gemini", "Google Gemini (AI Studio)", "gemini", example_model="gemini-2.5-pro"),
        Provider(
            "azure",
            "Azure OpenAI",
            "azure",
            base_url="required",
            example_model="my-deployment-name",
            needs_api_version=True,
        ),
        Provider(
            "bedrock",
            "AWS Bedrock",
            "bedrock",
            key_required=False,
            example_model="anthropic.claude-sonnet-5-5-v1:0",
            needs_region=True,
        ),
        Provider("deepseek", "DeepSeek", "deepseek", example_model="deepseek-chat"),
        Provider("groq", "Groq", "groq", example_model="llama-3.3-70b-versatile"),
        Provider("xai", "xAI (Grok)", "xai", example_model="grok-3"),
        Provider("openrouter", "OpenRouter", "openrouter", example_model="qwen/qwen3-235b-a22b"),
        Provider("together_ai", "Together AI", "together_ai", example_model="Qwen/Qwen3-235B-A22B"),
        Provider(
            "fireworks_ai",
            "Fireworks AI",
            "fireworks_ai",
            example_model="accounts/fireworks/models/qwen3-235b",
        ),
        Provider("cohere", "Cohere", "cohere_chat", example_model="command-a-03-2025"),
        Provider(
            "ollama",
            "Ollama (local)",
            "ollama_chat",
            base_url="optional",
            default_base_url="http://localhost:11434",
            key_required=False,
            example_model="qwen3:32b",
        ),
    )
}


def get(key: str) -> Provider:
    try:
        return PROVIDERS[key]
    except KeyError:
        raise ValueError(f"unknown provider {key!r}; known: {', '.join(PROVIDERS)}") from None


def route(provider: str, model: str) -> str:
    """LiteLLM model string, e.g. ``mistral/mistral-large-latest``. A model that already
    carries the prefix is left alone."""
    prefix = get(provider).prefix
    return model if model.startswith(prefix + "/") else f"{prefix}/{model}"
