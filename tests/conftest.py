"""Pytest fixtures and path setup."""

import sys
from pathlib import Path

import pytest


_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if _SRC_DIR.exists():
    src_path = str(_SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


@pytest.fixture
def my_test_number() -> int:
    """My test number.

    Returns
    -------
        int: A really awesome number.
    """
    return 42
