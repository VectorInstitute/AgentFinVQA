"""color_area_tool — OpenCV pixel-counting tool for chart color area measurement.

Objectively measures the fraction of chart area occupied by each legend color.
Runs between legend grounding and vision stages; no VLM calls, pure CPU.
"""

import re
from typing import Optional

import cv2
import numpy as np

from ..mep.schema import MEPColorArea


# Chart types where color-area measurement is meaningful
_COLOR_AREA_CHART_TYPES = {"bar", "bar_stacked", "bar_grouped", "pie", "donut"}

# Question keywords that indicate a size-comparison intent
_COMPARISON_KEYWORDS = {
    "largest",
    "smallest",
    "highest",
    "lowest",
    "most",
    "least",
    "greatest",
    "fewest",
    "dominant",
    "majority",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_rgb(rgb_val) -> Optional[tuple]:
    """Parse rgb_approximate to a (R, G, B) int tuple.

    Accepts a list/tuple of 3 ints or a string like "(70, 130, 180)".
    Returns None on failure.
    """
    if isinstance(rgb_val, (list, tuple)) and len(rgb_val) >= 3:
        try:
            return tuple(int(v) for v in rgb_val[:3])
        except (TypeError, ValueError):
            return None
    if isinstance(rgb_val, str):
        nums = re.findall(r"\d+", rgb_val)
        if len(nums) >= 3:
            try:
                return tuple(int(n) for n in nums[:3])
            except ValueError:
                return None
    return None


def _rgb_to_hsv(r: int, g: int, b: int) -> tuple:
    """Convert a single RGB pixel to HSV using OpenCV conventions.

    Returns (H, S, V) where H ∈ [0, 179], S/V ∈ [0, 255].
    """
    pixel = np.array([[[r, g, b]]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)
    return int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])


def _hue_distance(h1: int, h2: int) -> int:
    """Circular distance between two hue values in [0, 179]."""
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def color_area_tool(image_path: str, legend_map: list) -> dict:
    """Measure the chart area occupied by each legend color via pixel counting.

    Parameters
    ----------
    image_path : str
        Path to the chart image.
    legend_map : list of dict
        Each entry must have ``label`` (str) and ``rgb_approximate``
        (list of 3 ints or string like "(R, G, B)").

    Returns
    -------
    dict with keys:
        breakdown : dict[label → percentage]  (sorted descending)
        largest   : str | None
        total_pixels_matched : int
        error     : str | None
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "error": "image_load_failed",
            "breakdown": {},
            "largest": None,
            "total_pixels_matched": 0,
        }

    h, w = img.shape[:2]
    # Crop to chart region: top 85 % of height, left 75 % of width
    crop_h = int(h * 0.85)
    crop_w = int(w * 0.75)
    chart_region = img[:crop_h, :crop_w]

    hsv_img = cv2.cvtColor(chart_region, cv2.COLOR_BGR2HSV)

    counts: dict[str, int] = {}

    for entry in legend_map:
        label = entry.get("label", "")
        rgb = _parse_rgb(entry.get("rgb_approximate"))
        if rgb is None:
            counts[label] = 0
            continue

        r, g, b = rgb
        target_h, target_s, target_v = _rgb_to_hsv(r, g, b)

        # Build tolerance bounds; clamp to valid ranges
        lo_h = max(0, target_h - 10)
        hi_h = min(179, target_h + 10)
        lo_s = max(0, target_s - 40)
        hi_s = min(255, target_s + 40)
        lo_v = max(0, target_v - 40)
        hi_v = min(255, target_v + 40)

        lower = np.array([lo_h, lo_s, lo_v], dtype=np.uint8)
        upper = np.array([hi_h, hi_s, hi_v], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)

        counts[label] = int(np.sum(mask > 0))

    total_matched = sum(counts.values())
    if total_matched == 0:
        return {
            "error": "no_pixels_matched",
            "breakdown": {},
            "largest": None,
            "total_pixels_matched": 0,
        }

    breakdown = {label: round(cnt / total_matched * 100, 1) for label, cnt in counts.items()}
    # Sort descending by percentage
    breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))

    largest = max(breakdown, key=lambda lbl: breakdown[lbl])

    return {
        "breakdown": breakdown,
        "largest": largest,
        "total_pixels_matched": total_matched,
        "error": None,
    }


def should_trigger_color_area(chart_type: str, legend_map: list, question: str) -> bool:
    """Return True only when color-area measurement is appropriate.

    Gating conditions (ALL must be true):
    - chart_type is a supported color-area type (bar, pie, donut variants)
    - legend_map has more than one entry
    - question contains at least one size-comparison keyword
    """
    if chart_type not in _COLOR_AREA_CHART_TYPES:
        return False
    if len(legend_map) <= 1:
        return False
    q_lower = question.lower()
    return any(kw in q_lower for kw in _COMPARISON_KEYWORDS)


def check_color_ambiguity(legend_map: list) -> bool:
    """Return True if any two legend entries have HSV hues within 15 units.

    Ambiguous colors cannot be reliably distinguished by pixel matching,
    so the color_area stage is skipped when this returns True.

    Parameters
    ----------
    legend_map : list of dict
        Each entry needs ``rgb_approximate``.

    Returns
    -------
    bool
        True → colors too similar; False → colors are well-separated.
    """
    hues: list[tuple[int, str]] = []
    for entry in legend_map:
        label = entry.get("label", "")
        rgb = _parse_rgb(entry.get("rgb_approximate"))
        if rgb is None:
            continue
        h, _, _ = _rgb_to_hsv(*rgb)
        hues.append((h, label))

    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            if _hue_distance(hues[i][0], hues[j][0]) <= 15:
                return True
    return False


def format_color_area_block_for_vision(color_area: Optional[MEPColorArea]) -> str:
    """Build vision prompt text from color-area results."""
    if color_area is None or not color_area.triggered:
        return ""
    if color_area.low_confidence or color_area.color_ambiguity or color_area.parse_error:
        return ""
    if not color_area.breakdown:
        return ""
    lines = [
        "Pre-computed color area measurement (pixel counting — treat as "
        "objective measurement, more reliable than visual estimation for "
        "size-comparison questions):",
    ]
    sorted_items = sorted(color_area.breakdown.items(), key=lambda x: x[1], reverse=True)
    for label, pct in sorted_items:
        lines.append(f"  {label} → {pct}% of chart area")
    lines.append(
        "\nINSTRUCTION: For size-comparison questions (largest, smallest, "
        "most, least), use these pixel measurements as primary evidence. "
        "Only override if measurements look clearly wrong (e.g. very low "
        "total coverage)."
    )
    return "\n".join(lines)
