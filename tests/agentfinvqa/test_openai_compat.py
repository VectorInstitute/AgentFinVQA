"""Tests for OpenAI-compat endpoint helpers."""

import pytest

from agentfinvqa.utils.openai_compat import (
    is_qwen35_model,
    qwen35_extra_body,
    resolve_openai_endpoint,
)


# ---------------------------------------------------------------------------
# resolve_openai_endpoint
# ---------------------------------------------------------------------------


def test_resolve_explicit_key_and_base():
    key, base = resolve_openai_endpoint(api_key="sk-test", api_base="http://localhost:8000/v1")
    assert key == "sk-test"
    assert base == "http://localhost:8000/v1"


def test_resolve_env_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    key, base = resolve_openai_endpoint()
    assert key == "sk-from-env"
    assert base is None


def test_resolve_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    key, _ = resolve_openai_endpoint(api_key="sk-explicit")
    assert key == "sk-explicit"


def test_resolve_vllm_injects_empty_key_when_base_set(monkeypatch):
    """VLLM base + no key → inject 'EMPTY' so the OpenAI SDK doesn't reject ''."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    key, base = resolve_openai_endpoint(api_base="http://localhost:8000/v1")
    assert key == "EMPTY"
    assert base == "http://localhost:8000/v1"


def test_resolve_no_args_no_env_returns_empty_key_and_no_base(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    key, base = resolve_openai_endpoint()
    assert key == ""
    assert base is None


# ---------------------------------------------------------------------------
# is_qwen35_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["qwen3.5-72b-instruct", "Qwen3.5", "qwen3_5", "qwen3.6", "QWEN3_6-235B"],
)
def test_is_qwen35_model_true(name):
    assert is_qwen35_model(name)


@pytest.mark.parametrize(
    "name",
    ["gpt-4o", "gemini-1.5-flash", "qwen2.5-vl-72b", "qwen3-235b", ""],
)
def test_is_qwen35_model_false(name):
    assert not is_qwen35_model(name)


# ---------------------------------------------------------------------------
# qwen35_extra_body
# ---------------------------------------------------------------------------


def test_qwen35_extra_body_disables_thinking():
    extra = qwen35_extra_body("qwen3.5-72b-instruct")
    assert extra == {"chat_template_kwargs": {"enable_thinking": False}}


def test_qwen35_extra_body_empty_for_other_models():
    assert qwen35_extra_body("gpt-4o") == {}
    assert qwen35_extra_body("gemini-2.0-flash") == {}
