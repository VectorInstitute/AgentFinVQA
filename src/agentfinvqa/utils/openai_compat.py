"""OpenAI-compatible endpoint helpers.

Centralizes the ``api_key`` / ``api_base`` resolution used by every backend
caller (tools + CrewAI agent LLMs). Supports two deployment modes:

1. **Hosted OpenAI** (default) — ``api_base`` is empty, ``OPENAI_API_KEY`` is
   read from env; the official ``api.openai.com`` endpoint is used.
2. **Local OpenAI-compatible server** (e.g. vLLM serving Qwen2.5-VL) — caller
   supplies an explicit ``api_base`` (or sets ``OPENAI_BASE_URL`` in env);
   ``api_key`` may be empty since vLLM ignores it, and we substitute a dummy
   non-empty string so the OpenAI SDK constructor is happy.
"""

import os
from typing import Any, Optional, Tuple

from crewai import LLM
from openai import OpenAI

from .model_compat import openai_temperature


def resolve_openai_endpoint(api_key: str = "", api_base: str = "") -> Tuple[str, Optional[str]]:
    """Resolve effective (api_key, base_url) from explicit args + env fallbacks.

    Parameters
    ----------
    api_key : str
        Explicit key passed by the caller. May be empty.
    api_base : str
        Explicit base URL passed by the caller. May be empty.

    Returns
    -------
    tuple of (str, Optional[str])
        ``api_key`` (never empty — substitutes ``"EMPTY"`` for vLLM) and
        ``base_url`` (``None`` when the default OpenAI endpoint should be used).
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    resolved_base = api_base or os.environ.get("OPENAI_BASE_URL", "") or None
    if resolved_base and not resolved_key:
        # vLLM accepts any non-empty key; the OpenAI SDK rejects "".
        resolved_key = "EMPTY"
    return resolved_key, resolved_base


def build_openai_client(api_key: str = "", api_base: str = "") -> OpenAI:
    """Construct an ``OpenAI`` client honouring explicit args + env fallbacks."""
    key, base = resolve_openai_endpoint(api_key, api_base)
    if base:
        return OpenAI(api_key=key, base_url=base)
    return OpenAI(api_key=key)


def is_qwen35_model(model_name: str) -> bool:
    """Return True iff the model name belongs to the Qwen3.5 / Qwen3.6 family.

    These models enable thinking mode by default, prefixing responses with
    ``<think>...</think>`` blocks that break our strict-JSON parsing path.
    Callers should pass ``enable_thinking=False`` via ``extra_body`` to suppress.
    """
    name = (model_name or "").lower()
    return "qwen3.5" in name or "qwen3_5" in name or "qwen3.6" in name or "qwen3_6" in name


def qwen35_extra_body(model_name: str) -> dict:
    """Return the ``extra_body`` kwarg needed to disable thinking on Qwen3.5 / Qwen3.6.

    Empty dict for other models so the call site can always pass
    ``extra_body=qwen35_extra_body(self.model)`` without conditionals.
    """
    if not is_qwen35_model(model_name):
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def build_crewai_llm(
    backend: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Any:
    """Build a CrewAI ``LLM`` for OpenAI-compat / Gemini routing with Qwen thinking off.

    Centralizes the duplicated ``_build_llm`` helpers in planner/vision/verifier agents.
    For local vLLM, prefixes the model with ``openai/`` so LiteLLM uses the
    custom base URL.
    """
    if backend == "openai":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_base = api_base or os.environ.get("OPENAI_BASE_URL", "") or None
        model_str = f"openai/{model}" if resolved_base and not model.startswith("openai/") else model
        kwargs: dict[str, Any] = {
            "model": model_str,
            "api_key": resolved_key or "EMPTY",
            **openai_temperature(model),
        }
        if resolved_base:
            kwargs["base_url"] = resolved_base
        extra = qwen35_extra_body(model)
        if extra:
            kwargs["extra_body"] = extra
        return LLM(**kwargs)
    if backend == "gemini":
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            temperature=0,
        )
    raise ValueError(f"Unknown LLM backend: {backend!r}")
