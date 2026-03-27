"""Langfuse client helper with optional tracing instrumentation."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Protocol, cast

from dotenv import load_dotenv
from langfuse import Langfuse


class _Instrumentor(Protocol):
    """Minimal interface for instrumentation helpers used at runtime."""

    def instrument(self) -> None:
        """Activate automatic tracing hooks."""


try:
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor as _GoogleInstrumentorImpl
except Exception:  # pragma: no cover - optional dependency
    _GoogleInstrumentor: type[_Instrumentor] | None = None
else:
    _GoogleInstrumentor = cast(type[_Instrumentor], _GoogleInstrumentorImpl)

try:
    from openinference.instrumentation.openai import OpenAIInstrumentor as _OpenAIInstrumentorImpl
except Exception:  # pragma: no cover - optional dependency
    _OpenAIInstrumentor: type[_Instrumentor] | None = None
else:
    _OpenAIInstrumentor = cast(type[_Instrumentor], _OpenAIInstrumentorImpl)


def _build_instrumentor(cls: type[_Instrumentor] | None) -> _Instrumentor | None:
    """Instantiate an instrumentation helper, swallowing runtime errors."""
    if cls is None:
        return None
    try:
        return cls()
    except Exception:
        return None


_google_instrumentor = _build_instrumentor(_GoogleInstrumentor)
_openai_instrumentor = _build_instrumentor(_OpenAIInstrumentor)


_client = None
_initialised = False


def get_client() -> "Langfuse | None":
    """
    Initialize and return a globally cached Langfuse client.

    Retrieves configuration from environment variables and configures
    the SDK for local or cloud usage.

    Returns
    -------
    Langfuse or None
        An active client, or None if configuration is missing or invalid.
    """
    global _client, _initialised  # noqa: PLW0603
    if _initialised:
        return _client

    _initialised = True

    # Load environment variables from .env file
    with suppress(Exception):
        load_dotenv()

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        return None

    try:
        kwargs: dict = {"public_key": public_key, "secret_key": secret_key}
        # Accept LANGFUSE_HOST or LANGFUSE_BASE_URL (both are common)
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "")
        if host:
            kwargs["host"] = host

        _client = Langfuse(**kwargs)
        # Activate OTel auto-instrumentation so provider SDK calls (Google GenAI,
        # OpenAI) are captured as detailed child spans inside Langfuse traces.
        if _google_instrumentor is not None:
            with suppress(Exception):
                _google_instrumentor.instrument()
        if _openai_instrumentor is not None:
            with suppress(Exception):
                _openai_instrumentor.instrument()
    except Exception as exc:
        print(f"[langfuse] client init failed: {exc}")
        _client = None

    return _client


def reset_client() -> None:
    """
    Clear the cached client and reset initialization state.

    Returns
    -------
    None
    """
    global _client, _initialised  # noqa: PLW0603
    _client = None
    _initialised = False
