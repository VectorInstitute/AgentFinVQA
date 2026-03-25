"""Strict JSON parsing policy with repair fallback."""

import json
import re
from typing import Any, Optional

from json_repair import repair_json


def parse_strict(
    text: str,
    required_keys: Optional[list[str]] = None,
) -> tuple[dict[str, Any], bool]:
    """
    Parse JSON data from a string with automatic cleanup and repair.

    Handles markdown fences and extracts the first JSON block found.

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
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if _check_keys(result, required_keys):
            return result, True
        raise ValueError("Missing required keys")
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract first JSON block from surrounding text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if _check_keys(result, required_keys):
                return result, True
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Repair fallback
    try:
        repaired = repair_json(text)
        result = json.loads(repaired)
        if _check_keys(result, required_keys):
            return result, False  # parse_ok=False: needed repair
    except Exception:
        pass

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
