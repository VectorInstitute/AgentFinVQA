"""Model compatibility helpers for provider-specific quirks."""

# OpenAI models that reject the temperature parameter (only default/1 supported)
_NO_TEMPERATURE_MODELS = frozenset(
    {
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "gpt-5-mini",
    }
)


def openai_temperature(model: str) -> dict:
    """Return ``{"temperature": 0}`` for models that support it, else ``{}``.

    Use as ``**openai_temperature(model)`` when building OpenAI API kwargs.
    """
    if model in _NO_TEMPERATURE_MODELS:
        return {}
    return {"temperature": 0}
