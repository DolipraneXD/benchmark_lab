import pytest

from benchlab.config import load_config
from benchlab.llm import litellm_model_name, litellm_params, per_model_exceptions, request_params
from benchlab.redact import redact
from tests.conftest import MOCK_KEY


def test_load_and_select(models_yaml):
    cfg = load_config(models_yaml)
    assert cfg.baseline.id == "claude"
    assert [m.id for m in cfg.select("mock,claude")] == ["mock", "claude"]
    assert all(m.kind == "chat" for m in cfg.select("all"))
    assert "qemb" not in [m.id for m in cfg.select("all")]
    assert cfg.get("qemb").suites == ["retrieval"]
    assert cfg.get("mock").suites == ["core"]


def test_litellm_mapping(models_yaml):
    cfg = load_config(models_yaml)
    assert litellm_model_name(cfg.get("mock")) == "openai/mock-model"
    assert litellm_model_name(cfg.get("claude")) == "anthropic/claude-sonnet-5-5"
    assert litellm_model_name(cfg.get("bed")) == "bedrock/us.anthropic.claude-sonnet-5-5"
    p = litellm_params(cfg.get("mock"), cfg.settings)
    assert p["api_base"].endswith("/v1") and p["api_key"] == MOCK_KEY
    assert p["input_cost_per_token"] == pytest.approx(1e-6)
    assert p["output_cost_per_token"] == pytest.approx(2e-6)
    assert litellm_params(cfg.get("bed"), cfg.settings)["aws_region_name"] == "us-east-1"


def test_request_params_and_exceptions(models_yaml):
    cfg = load_config(models_yaml)
    assert request_params(cfg.get("mock"), cfg.settings) == {"max_tokens": 4096, "temperature": 0.0}
    # temperature: null removes the global default and is recorded as an exception
    assert "temperature" not in request_params(cfg.get("claude"), cfg.settings)
    assert per_model_exceptions(cfg.get("claude"), cfg.settings) == {"temperature": "<omitted>"}
    assert per_model_exceptions(cfg.get("mock"), cfg.settings) == {}


def test_cost(models_yaml):
    cfg = load_config(models_yaml)
    assert cfg.get("mock").cost_usd(1_000_000, 500_000) == pytest.approx(2.0)
    assert cfg.get("notools").cost_usd(10, 10) is None  # unknown prices -> n/a, never 0


def test_keys_never_in_public_dict_and_redaction(models_yaml):
    cfg = load_config(models_yaml)
    d = cfg.get("mock").public_dict()
    assert MOCK_KEY not in str(d) and d["api_key_env"] == "MOCK_API_KEY"
    assert MOCK_KEY not in redact(f"Authorization: Bearer {MOCK_KEY} failed")
    assert "sk-abcdef***" == redact("sk-abcdefGHIJKLMNOP")


def test_validation_errors(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "models:\n  - id: x\n    name: X\n    provider: openai_compatible\n    model: m\n    api_key_env: K\n"
    )
    with pytest.raises(Exception, match="base_url"):
        load_config(p)
    p.write_text(
        "settings: {baseline_model: nope}\nmodels:\n  - id: x\n    name: X\n    provider: anthropic\n    model: m\n    api_key_env: K\n"
    )
    with pytest.raises(Exception, match="unknown model id 'nope'"):
        load_config(p)
    p.write_text(
        "models:\n  - id: Bad-Id\n    name: X\n    provider: anthropic\n    model: m\n    api_key_env: K\n"
    )
    with pytest.raises(Exception, match="must match"):
        load_config(p)
