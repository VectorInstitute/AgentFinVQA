"""Unit tests for color_area_tool and vision color-area prompt block."""

from unittest.mock import patch

import numpy as np

from agentfinvqa.mep.schema import MEP, MEPColorArea
from agentfinvqa.tools.color_area_tool import (
    check_color_ambiguity,
    color_area_tool,
    format_color_area_block_for_vision,
    should_trigger_color_area,
)


def test_should_trigger_all_gated_types_with_keyword():
    legend = [
        {"label": "A", "rgb_approximate": [255, 0, 0]},
        {"label": "B", "rgb_approximate": [0, 0, 255]},
    ]
    for ct in ("bar", "bar_stacked", "bar_grouped", "pie", "donut"):
        assert should_trigger_color_area(ct, legend, "Which is the largest segment?") is True


def test_should_trigger_false_non_comparison_question():
    legend = [
        {"label": "A", "rgb_approximate": [255, 0, 0]},
        {"label": "B", "rgb_approximate": [0, 0, 255]},
    ]
    assert should_trigger_color_area("bar", legend, "What is the title of the chart?") is False


def test_should_trigger_false_non_gated_chart():
    legend = [
        {"label": "A", "rgb_approximate": [255, 0, 0]},
        {"label": "B", "rgb_approximate": [0, 0, 255]},
    ]
    assert should_trigger_color_area("line", legend, "Which is largest?") is False


def test_should_trigger_false_single_legend():
    legend = [{"label": "Only", "rgb_approximate": [255, 0, 0]}]
    assert should_trigger_color_area("bar", legend, "Which is largest?") is False


def test_check_color_ambiguity_true_similar_hue():
    # Identical RGB → same hue → ambiguous by the hue-distance rule
    legend = [
        {"label": "A", "rgb_approximate": [200, 100, 50]},
        {"label": "B", "rgb_approximate": [200, 100, 50]},
    ]
    assert check_color_ambiguity(legend) is True


def test_check_color_ambiguity_false_well_separated():
    legend = [
        {"label": "A", "rgb_approximate": [255, 0, 0]},
        {"label": "B", "rgb_approximate": [0, 255, 0]},
    ]
    assert check_color_ambiguity(legend) is False


@patch("agentfinvqa.tools.color_area_tool.cv2.imread")
def test_color_area_tool_no_pixels_matched_white(mock_imread):
    mock_imread.return_value = np.full((100, 100, 3), 255, dtype=np.uint8)
    legend = [
        {"label": "X", "rgb_approximate": [255, 0, 0]},
        {"label": "Y", "rgb_approximate": [0, 0, 255]},
    ]
    out = color_area_tool("/fake/path.png", legend)
    assert out["error"] == "no_pixels_matched"
    assert out["breakdown"] == {}
    assert out["largest"] is None
    assert out["total_pixels_matched"] == 0


def test_mep_color_area_serializes_in_to_dict():
    ca = MEPColorArea(
        triggered=True,
        breakdown={"A": 60.0, "B": 40.0},
        largest="A",
        total_pixels_matched=100,
        low_confidence=False,
        color_ambiguity=False,
        parse_error=False,
        tool_trace=[{"tool": "color_area"}],
    )
    mep = MEP(run_id="r1", color_area=ca)
    d = mep.to_dict()
    assert d["color_area"]["triggered"] is True
    assert d["color_area"]["breakdown"] == {"A": 60.0, "B": 40.0}
    assert d["color_area"]["largest"] == "A"
    assert d["color_area"]["total_pixels_matched"] == 100
    assert d["color_area"]["tool_trace"] == [{"tool": "color_area"}]


def test_format_color_area_block_empty_when_low_confidence():
    ca = MEPColorArea(
        triggered=True,
        breakdown={"A": 50.0},
        largest="A",
        total_pixels_matched=10,
        low_confidence=True,
        color_ambiguity=False,
        parse_error=False,
    )
    assert format_color_area_block_for_vision(ca) == ""


def test_format_color_area_block_correct_when_confident():
    ca = MEPColorArea(
        triggered=True,
        breakdown={"B": 30.0, "A": 70.0},
        largest="A",
        total_pixels_matched=100,
        low_confidence=False,
        color_ambiguity=False,
        parse_error=False,
    )
    block = format_color_area_block_for_vision(ca)
    assert "Pre-computed color area measurement" in block
    assert "A → 70.0% of chart area" in block
    assert "B → 30.0% of chart area" in block
    # Descending by percentage: A before B
    assert block.index("A →") < block.index("B →")
    assert "primary evidence" in block
