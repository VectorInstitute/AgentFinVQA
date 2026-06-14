"""Tests for model compatibility helpers."""

import pytest

from agentfinvqa.utils.model_compat import openai_temperature


@pytest.mark.parametrize("model", ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "gpt-5-mini"])
def test_no_temperature_for_reasoning_models(model):
    """Reasoning models reject temperature; helper must return empty dict."""
    assert openai_temperature(model) == {}


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4-turbo", "gemini-1.5-flash", "qwen3.5-72b"])
def test_temperature_zero_for_standard_models(model):
    assert openai_temperature(model) == {"temperature": 0}
