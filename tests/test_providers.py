"""Providers: every entry routes through LiteLLM with the right endpoint fields, and the
config refuses a model that is missing what its provider needs."""

import pytest
from pydantic import ValidationError

from benchlab.config import ModelSpec
from benchlab.llm import litellm_params
from benchlab.providers import PROVIDERS, route
from benchlab.proxy import _proxy_litellm_params


def _spec(provider, **kw):
    p = PROVIDERS[provider]
    base = {"base_url": "https://example.test/v1"} if p.base_url == "required" else {}
    ver = {"api_version": "2024-10-21"} if p.needs_api_version else {}
    key = {"api_key_env": "PROV_TEST_KEY"} if p.key_required else {}
    return ModelSpec(id="m", name="M", provider=provider, model="x", **{**base, **ver, **key, **kw})


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_every_provider_routes(provider, monkeypatch):
    monkeypatch.setenv("PROV_TEST_KEY", "sk-provider-test-0000000000")
    spec = _spec(provider)
    p = litellm_params(spec)
    assert p["model"] == f"{PROVIDERS[provider].prefix}/x"
    proxied = _proxy_litellm_params(spec)
    assert "sk-provider-test" not in str(proxied)  # the proxy config holds references only
    if PROVIDERS[provider].key_required:
        assert proxied["api_key"] == "os.environ/PROV_TEST_KEY"


def test_provider_specific_fields(monkeypatch):
    monkeypatch.setenv("PROV_TEST_KEY", "sk-provider-test-0000000000")
    az = litellm_params(_spec("azure"))
    assert az["api_version"] == "2024-10-21" and az["api_base"] == "https://example.test/v1"
    assert litellm_params(_spec("ollama"))["api_base"] == "http://localhost:11434"
    assert litellm_params(_spec("bedrock", region="eu-west-3"))["aws_region_name"] == "eu-west-3"
    assert litellm_params(_spec("openai_compatible"))["api_key"] == "EMPTY"  # no key set
    assert route("mistral", "mistral/mistral-large-latest") == "mistral/mistral-large-latest"


def test_config_refuses_incomplete_models():
    with pytest.raises(ValidationError, match="unknown provider 'nope'"):
        ModelSpec(id="m", name="M", provider="nope", model="x")
    with pytest.raises(ValidationError, match="azure needs base_url"):
        ModelSpec(id="m", name="M", provider="azure", model="x", api_key_env="K", api_version="v")
    with pytest.raises(ValidationError, match="azure needs api_version"):
        ModelSpec(id="m", name="M", provider="azure", model="x", api_key_env="K", base_url="u")
    with pytest.raises(ValidationError, match="mistral needs an API key"):
        ModelSpec(id="m", name="M", provider="mistral", model="x")
    ModelSpec(id="m", name="M", provider="ollama", model="x")  # local: no key, no URL needed


def test_clean_base_url_removes_endpoint_paths():
    from benchlab.providers import clean_base_url

    assert clean_base_url("https://llmhub.leyton.com/v1/chat/completions") == (
        "https://llmhub.leyton.com/v1",
        "/chat/completions",
    )
    assert clean_base_url("https://api.example.com/v1/") == ("https://api.example.com/v1", None)
    assert clean_base_url("http://h:8000/v1/completions")[0] == "http://h:8000/v1"
