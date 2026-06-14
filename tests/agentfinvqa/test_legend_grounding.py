"""Tests for the legend grounding pipeline stage.

Covers:
- _format_legend_grounding_block formatting
- build_vision_task_description legend injection and prepend_instruction
- MEPLegendGrounding schema fields and serialisation
- LegendGrounderTool prompt construction, error fallback, pop_traces
- Gate logic (chart type + legend length)
- Compliance check (label mentioned / not mentioned in explanation)
"""

import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agentfinvqa.agents.vision_agent import (
    _format_legend_grounding_block,
    build_vision_task_description,
)
from agentfinvqa.datasets.perceived_sample import PerceivedSample, QuestionType
from agentfinvqa.mep.schema import MEP, MEPLegendGrounding
from agentfinvqa.runner.run_generate_meps import _LEGEND_GROUNDING_CHART_TYPES
from agentfinvqa.tools.legend_grounder_tool import _LEGEND_GROUNDER_PROMPT_TEMPLATE, LegendGrounderTool


try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _tiny_png_bytes(width: int = 2, height: int = 2) -> bytes:
    if PILImage is None:
        return _MINIMAL_PNG
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color="white").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(image_path: str = "/tmp/chart.png") -> PerceivedSample:
    """Create a dummy PerceivedSample for testing."""
    return PerceivedSample(
        sample_id="test_001",
        question="Which player scored highest?",
        question_type=QuestionType.STANDARD,
        expected_output="Clayton Oliver",
        image_path=image_path,
        choices=[],
        context=[],
        metadata={},
    )


def _make_plan() -> dict:
    """Create a sample plan dict for testing."""
    return {"steps": ["Read legend", "Find max value"]}


_SAMPLE_LEGEND_MAP = [
    {
        "label": "Clayton Oliver",
        "color_description": "orange",
        "rgb_approximate": "(255, 165, 0)",
        "line_style": "solid",
        "confidence": 0.92,
    },
    {
        "label": "Marcus Bontempelli",
        "color_description": "blue",
        "rgb_approximate": "(70, 130, 180)",
        "line_style": "solid",
        "confidence": 0.88,
    },
    {
        "label": "Top-10 Player",
        "color_description": "grey",
        "rgb_approximate": "(128, 128, 128)",
        "line_style": "dashed",
        "confidence": 0.91,
    },
]


# ---------------------------------------------------------------------------
# _format_legend_grounding_block
# ---------------------------------------------------------------------------


class TestFormatLegendGroundingBlock:
    """Tests for ``_format_legend_grounding_block``."""

    def test_empty_returns_empty_string(self):
        """Test returns empty string for an empty legend map."""
        assert _format_legend_grounding_block([]) == ""

    def test_none_returns_empty_string(self):
        """Test returns empty string for legend map None."""
        assert _format_legend_grounding_block(None) == ""

    def test_header_present(self):
        """Test the output contains the proper header for a valid legend map."""
        result = _format_legend_grounding_block(_SAMPLE_LEGEND_MAP)
        assert "Pre-mapped legend (treat as ground truth" in result

    def test_each_label_present(self):
        """Test all legend labels are present in the formatted block."""
        result = _format_legend_grounding_block(_SAMPLE_LEGEND_MAP)
        assert "Clayton Oliver" in result
        assert "Marcus Bontempelli" in result
        assert "Top-10 Player" in result

    def test_color_and_style_present(self):
        """Test that color descriptions and line styles are included in the output."""
        result = _format_legend_grounding_block(_SAMPLE_LEGEND_MAP)
        assert "orange" in result
        assert "solid" in result
        assert "dashed" in result

    def test_confidence_formatted(self):
        """Test that confidence values are formatted correctly."""
        result = _format_legend_grounding_block(_SAMPLE_LEGEND_MAP)
        assert "0.92" in result
        assert "0.88" in result

    def test_instruction_block_present(self):
        """Test that the instruction block is appended in the legend ground output."""
        result = _format_legend_grounding_block(_SAMPLE_LEGEND_MAP)
        assert "INSTRUCTION" in result
        assert "pre-mapped colors as ground truth" in result

    def test_missing_fields_do_not_crash(self):
        """Sparse entries format without error; missing fields use ``unknown``."""
        sparse_map = [{"label": "SeriesA"}]
        result = _format_legend_grounding_block(sparse_map)
        assert "SeriesA" in result
        assert "unknown" in result  # default color/style fallback


# ---------------------------------------------------------------------------
# build_vision_task_description — legend_map and prepend_instruction
# ---------------------------------------------------------------------------


class TestBuildVisionTaskDescription:
    """Tests for ``build_vision_task_description`` (legend map and prefix)."""

    def test_legend_grounding_block_injected_when_legend_map_given(self):
        """Include the legend block when ``legend_map`` is non-empty."""
        sample = _make_sample()
        plan = _make_plan()
        desc = build_vision_task_description(sample, plan, legend_map=_SAMPLE_LEGEND_MAP)
        assert "Pre-mapped legend" in desc
        assert "Clayton Oliver" in desc
        assert "INSTRUCTION" in desc

    def test_no_legend_block_when_legend_map_empty(self):
        """Test that the legend block is not included when legend_map is empty."""
        sample = _make_sample()
        plan = _make_plan()
        desc = build_vision_task_description(sample, plan, legend_map=[])
        assert "Pre-mapped legend" not in desc
        assert "INSTRUCTION: When extracting values" not in desc

    def test_no_legend_block_when_legend_map_none(self):
        """Test that no legend block appears when legend_map is None."""
        sample = _make_sample()
        plan = _make_plan()
        desc = build_vision_task_description(sample, plan, legend_map=None)
        assert "Pre-mapped legend" not in desc

    def test_prepend_instruction_appears_first(self):
        """``prepend_instruction`` is the first text in the description."""
        sample = _make_sample()
        plan = _make_plan()
        prefix = "IMPORTANT: Use the pre-mapped legend."
        desc = build_vision_task_description(sample, plan, prepend_instruction=prefix)
        assert desc.startswith(prefix)

    def test_prepend_instruction_none_no_effect(self):
        """``prepend_instruction=None`` matches omitting the argument."""
        sample = _make_sample()
        plan = _make_plan()
        desc_no_prefix = build_vision_task_description(sample, plan)
        desc_with_none = build_vision_task_description(sample, plan, prepend_instruction=None)
        assert desc_no_prefix == desc_with_none

    def test_ocr_block_still_present_alongside_legend_map(self):
        """Test that the OCR block is present even when using a legend_map."""
        sample = _make_sample()
        plan = _make_plan()
        ocr_result = {
            "chart_type": "line",
            "title": "Player Rankings",
            "x_axis": {"label": "Year", "ticks": ["2020", "2021"]},
            "y_axis": {"label": "Score", "ticks": ["0", "100"]},
            "legend": ["Clayton Oliver", "Marcus Bontempelli"],
            "data_labels": [],
            "annotations": [],
        }
        desc = build_vision_task_description(sample, plan, ocr_result=ocr_result, legend_map=_SAMPLE_LEGEND_MAP)
        assert "Pre-extracted text from chart" in desc
        assert "Pre-mapped legend" in desc


# ---------------------------------------------------------------------------
# MEPLegendGrounding schema
# ---------------------------------------------------------------------------


class TestMEPLegendGrounding:
    """Tests for ``MEPLegendGrounding`` defaults and serialization."""

    def test_default_values(self):
        """Test that defaults are set correctly for MEPLegendGrounding."""
        lg = MEPLegendGrounding()
        assert lg.triggered is False
        assert lg.legend_map == []
        assert lg.parse_error is False
        assert lg.compliance_retry_triggered is False
        assert lg.tool_trace == []

    def test_can_set_all_fields(self):
        """Test that all MEPLegendGrounding fields can be set and are stored."""
        lg = MEPLegendGrounding(
            triggered=True,
            legend_map=_SAMPLE_LEGEND_MAP,
            parse_error=False,
            compliance_retry_triggered=True,
            tool_trace=[{"tool": "legend_grounder_tool"}],
        )
        assert lg.triggered is True
        assert len(lg.legend_map) == 3
        assert lg.compliance_retry_triggered is True

    def test_mep_serialises_legend_grounding(self):
        """Test serializing legend_grounding inside MEP produces correct output."""
        mep = MEP(run_id="r1")
        mep.legend_grounding = MEPLegendGrounding(
            triggered=True,
            legend_map=_SAMPLE_LEGEND_MAP,
        )
        d = mep.to_dict()
        assert d["legend_grounding"]["triggered"] is True
        assert len(d["legend_grounding"]["legend_map"]) == 3

    def test_mep_legend_grounding_none_serialises(self):
        """Test serialization of a MEP with no legend_grounding yields None."""
        mep = MEP(run_id="r2")
        d = mep.to_dict()
        assert d["legend_grounding"] is None


# ---------------------------------------------------------------------------
# Gate logic: _LEGEND_GROUNDING_CHART_TYPES
# ---------------------------------------------------------------------------


class TestLegendGroundingGate:
    """Tests for legend-grounding gate (chart type and legend length)."""

    @pytest.mark.parametrize(
        "chart_type",
        [
            "line",
            "bar",
            "scatter",
            "area",
            "bar_grouped",
            "bar_stacked",
            "combination",
            "pie",
            "donut",
        ],
    )
    def test_multi_series_chart_types_in_gate(self, chart_type):
        """Chart types that use legend grounding are in the allowlist."""
        assert chart_type in _LEGEND_GROUNDING_CHART_TYPES

    @pytest.mark.parametrize("chart_type", ["table", "dashboard", "other", "unknown"])
    def test_single_series_types_not_in_gate(self, chart_type):
        """Chart types outside the legend-grounding allowlist are not gated in."""
        assert chart_type not in _LEGEND_GROUNDING_CHART_TYPES

    def test_gate_requires_more_than_one_legend_entry(self):
        """Simulate the gate check from process_sample."""
        ocr_legend = ["Only series"]
        ocr_chart_type = "line"
        grounder_mock = MagicMock()

        should_ground = (
            grounder_mock is not None and len(ocr_legend) > 1 and ocr_chart_type in _LEGEND_GROUNDING_CHART_TYPES
        )
        assert should_ground is False

    def test_gate_passes_with_two_entries_and_line_chart(self):
        """Two legend rows and a gated chart type enable grounding."""
        ocr_legend = ["Series A", "Series B"]
        ocr_chart_type = "line"
        grounder_mock = MagicMock()

        should_ground = (
            grounder_mock is not None and len(ocr_legend) > 1 and ocr_chart_type in _LEGEND_GROUNDING_CHART_TYPES
        )
        assert should_ground is True

    def test_gate_skips_when_grounder_is_none(self):
        """No grounding when the grounder tool is absent (None)."""
        ocr_legend = ["Series A", "Series B"]
        ocr_chart_type = "line"
        grounder = None
        should_ground = grounder is not None and len(ocr_legend) > 1 and ocr_chart_type in _LEGEND_GROUNDING_CHART_TYPES
        assert should_ground is False


# ---------------------------------------------------------------------------
# Compliance check logic
# ---------------------------------------------------------------------------


class TestComplianceCheck:
    """Test the label-in-explanation compliance logic in isolation."""

    def _compliance_passes(self, legend_map: list, explanation: str) -> bool:
        """Return True if any non-empty legend label occurs in the explanation."""
        expl = explanation.lower()
        return any(entry.get("label", "").lower() in expl for entry in legend_map if entry.get("label"))

    def test_passes_when_label_mentioned(self):
        """Test compliance passes when a label is mentioned in the explanation."""
        explanation = "Clayton Oliver shows the highest score in 2021."
        assert self._compliance_passes(_SAMPLE_LEGEND_MAP, explanation) is True

    def test_passes_when_any_label_mentioned(self):
        """Test compliance passes when any valid label is mentioned."""
        explanation = "The grey dashed line for Top-10 Player peaks in 2020."
        assert self._compliance_passes(_SAMPLE_LEGEND_MAP, explanation) is True

    def test_fails_when_no_label_mentioned(self):
        """Test compliance fails when no labels are mentioned in explanation."""
        explanation = "The orange line peaks at 95 in 2021."
        assert self._compliance_passes(_SAMPLE_LEGEND_MAP, explanation) is False

    def test_case_insensitive_match(self):
        """Test compliance check is case-insensitive."""
        explanation = "clayton oliver leads by a wide margin."
        assert self._compliance_passes(_SAMPLE_LEGEND_MAP, explanation) is True

    def test_empty_explanation_fails(self):
        """Test compliance fails for an empty explanation."""
        assert self._compliance_passes(_SAMPLE_LEGEND_MAP, "") is False

    def test_empty_legend_map_always_fails(self):
        """Test compliance passes never when the legend_map is empty."""
        assert self._compliance_passes([], "Clayton Oliver scored 90.") is False

    def test_entries_without_label_key_are_skipped(self):
        """Test entries in the legend_map without a 'label' key are skipped."""
        map_no_labels = [{"color_description": "orange"}]
        assert self._compliance_passes(map_no_labels, "orange line peaks.") is False


# ---------------------------------------------------------------------------
# LegendGrounderTool — unit tests with mocked API calls
# ---------------------------------------------------------------------------


class TestLegendGrounderTool:
    """Tests for ``LegendGrounderTool`` with mocked or invalid backends."""

    def _make_tool(self) -> LegendGrounderTool:
        """Instantiate a LegendGrounderTool for tests with test credentials."""
        return LegendGrounderTool(backend="gemini", model="gemini-2.5-flash-lite", api_key="fake")

    def test_pop_traces_returns_and_clears(self):
        """Test that pop_traces returns and clears trace data."""
        tool = self._make_tool()
        tool._traces.append({"tool": "legend_grounder_tool", "elapsed_ms": 42.0})
        traces = tool.pop_traces()
        assert len(traces) == 1
        assert traces[0]["tool"] == "legend_grounder_tool"
        assert tool.pop_traces() == []

    def test_unknown_backend_returns_error_json(self):
        """Unknown backend returns JSON with ``error``, not an exception."""
        img_bytes = _tiny_png_bytes()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        tool = LegendGrounderTool(backend="unknown", model="m", api_key="k")
        try:
            raw = tool._run(image_path=tmp_path, legend_list=["A", "B"])
        finally:
            os.unlink(tmp_path)

        parsed = json.loads(raw)
        assert parsed["legend_map"] == []
        assert "Unknown backend" in parsed.get("error", "")

    @patch("agentfinvqa.tools.legend_grounder_tool.genai")
    def test_gemini_backend_returns_raw_text_and_appends_trace(self, mock_genai):
        """Gemini path returns model text and records one tool trace."""
        # Arrange
        expected_json = json.dumps(
            {
                "legend_map": [
                    {"label": "Series A", "color_description": "red", "line_style": "solid", "confidence": 0.9}
                ]
            }
        )
        mock_response = MagicMock()
        mock_response.text = expected_json
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        mock_genai.types = MagicMock()

        tool = self._make_tool()

        img_bytes = _tiny_png_bytes(width=4, height=4)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        try:
            raw = tool._run(image_path=tmp_path, legend_list=["Series A"])
        finally:
            os.unlink(tmp_path)

        assert raw == expected_json
        traces = tool.pop_traces()
        assert len(traces) == 1
        assert traces[0]["tool"] == "legend_grounder_tool"
        assert traces[0]["backend"] == "gemini"

    @patch("agentfinvqa.tools.legend_grounder_tool.genai")
    def test_api_error_returns_fallback_json(self, mock_genai):
        """Test that errors during API calls are caught and return fallback JSON."""
        mock_genai.Client.side_effect = RuntimeError("API unavailable")

        img_bytes = _tiny_png_bytes(width=4, height=4)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        tool = self._make_tool()
        try:
            raw = tool._run(image_path=tmp_path, legend_list=["Series A"])
        finally:
            os.unlink(tmp_path)

        parsed = json.loads(raw)
        assert parsed["legend_map"] == []
        assert "error" in parsed

    def test_prompt_contains_all_legend_entries(self):
        """The prompt template lists every legend label and expected JSON keys."""
        entries = ["Clayton Oliver", "Marcus Bontempelli", "Top-10 Player"]
        legend_formatted = "\n".join(f"  - {e}" for e in entries)
        prompt = _LEGEND_GROUNDER_PROMPT_TEMPLATE.format(legend_list=legend_formatted)

        for entry in entries:
            assert entry in prompt
        assert "legend_map" in prompt
        assert "color_description" in prompt
        assert "line_style" in prompt
        assert "confidence" in prompt
