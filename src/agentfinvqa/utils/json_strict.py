"""Strict JSON parsing policy with repair fallback."""

import json
import re
from typing import Any, Optional

from json_repair import repair_json


# Qwen3.x thinking blocks (vLLM reasoning parser / chat template variants).
_LT, _GT = chr(60), chr(62)
_THINKING_BLOCK_RE = re.compile(
    r"(?s)\s*(?:"
    + _LT
    + r"think"
    + _GT
    + r".*?"
    + _LT
    + r"/think"
    + _GT
    + r"|"
    + _LT
    + r"redacted_thinking"
    + _GT
    + r".*?"
    + _LT
    + r"/redacted_thinking"
    + _GT
    + r")\s*",
    re.IGNORECASE,
)


def _strip_thinking_blocks(text: str) -> str:
    return _THINKING_BLOCK_RE.sub("", text).strip()


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_last_json_object(text: str) -> Optional[str]:
    """Return the longest top-level JSON object substring in *text*, if any.

    Qwen3.x Crew agents often emit chain-of-thought, draft JSON fragments, then a
    final JSON answer. Taking the *last* ``{`` can yield a nested fragment; the
    longest valid top-level object is usually the intended payload.
    """
    decoder = json.JSONDecoder()
    best: Optional[str] = None
    best_len = 0
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        candidate = text[i:]
        try:
            obj, end = decoder.raw_decode(candidate)
            if isinstance(obj, dict) and end > best_len:
                best = candidate[:end]
                best_len = end
        except json.JSONDecodeError:
            continue
    return best


def _prepare_json_text(text: str) -> str:
    text = _strip_thinking_blocks(text)
    text = _strip_markdown_fences(text)
    if not text.lstrip().startswith("{"):
        extracted = _extract_last_json_object(text)
        if extracted:
            text = extracted
    return text.strip()


def parse_strict(
    text: str,
    required_keys: Optional[list[str]] = None,
) -> tuple[dict[str, Any], bool]:
    """
    Parse JSON data from a string with automatic cleanup and repair.

    Handles markdown fences, Qwen thinking blocks, and CoT preambles before the
    final JSON object.

    Parameters
    ----------
    text : str
        The raw string content to parse.
    required_keys : list of str, optional
        Keys that must be present in the resulting dictionary.

    Returns
    -------
    result : dict
        The parsed data, or an empty dict if parsing failed.
    parse_ok : bool
        True if the JSON was valid without needing structural repairs.
    """
    text = _prepare_json_text(text)

    # 1. Direct parse
    try:
        result = json.loads(text)
        if _check_keys(result, required_keys):
            return result, True
        raise ValueError("Missing required keys")
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Last JSON object (handles CoT + draft JSON + final JSON)
    extracted = _extract_last_json_object(text)
    if extracted and extracted != text:
        try:
            result = json.loads(extracted)
            if _check_keys(result, required_keys):
                return result, True
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. First JSON block (legacy path)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if _check_keys(result, required_keys):
                return result, True
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Repair fallback
    for candidate in (extracted, text):
        if not candidate:
            continue
        try:
            repaired = repair_json(candidate)
            result = json.loads(repaired)
            if _check_keys(result, required_keys):
                return result, False  # parse_ok=False: needed repair
        except Exception:
            continue

    return {}, False


def _check_keys(result: Any, required_keys: Optional[list[str]]) -> bool:
    """
    Validate that the object is a dictionary containing specific keys.

    Parameters
    ----------
    result : Any
        The object to validate.
    required_keys : list of str, optional
        The minimal set of keys expected.

    Returns
    -------
    bool
        True if the object is a dict and all keys are present.
    """
    if not isinstance(result, dict):
        return False
    return not required_keys or all(k in result for k in required_keys)
