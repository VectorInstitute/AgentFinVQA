"""Helper utilities for dataset image persistence."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PIL import Image as PILImage


def save_image_data(image: Any, path: Path) -> str:
    """
    Persist dataset-provided image content to ``path``.

    Parameters
    ----------
    image : Any
        A PIL Image, numpy array, HF image dict, or binary bytes.
    path : Path
        Destination file path (parent directories must already exist).

    Returns
    -------
    str
        Absolute path to the saved file, or empty string on failure.
    """
    path = path.resolve()
    result = ""
    if path.exists():
        return str(path)

    try:
        if isinstance(image, PILImage.Image):
            image.save(str(path))
            result = str(path)
        elif isinstance(image, bytes):
            path.write_bytes(image)
            result = str(path)
        elif isinstance(image, dict):
            raw = image.get("bytes") or image.get("path")
            if isinstance(raw, bytes):
                path.write_bytes(raw)
                result = str(path)
            elif isinstance(raw, str) and raw:
                shutil.copy(raw, str(path))
                result = str(path)
            else:
                raise ValueError(f"Unknown image dict keys: {list(image.keys())}")
        else:
            # Fallback: try numpy array-like → PIL
            import numpy as np  # noqa: PLC0415

            PILImage.fromarray(np.array(image)).save(str(path))
            result = str(path)
    except Exception as exc:  # pragma: no cover - log + continue
        print(f"  Warning: could not save image to {path}: {exc}")

    return result
