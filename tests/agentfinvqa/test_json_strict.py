"""Tests for strict JSON extraction from LLM outputs."""

from agentfinvqa.utils.json_strict import parse_strict


def test_parse_cot_preamble_with_final_json():
    raw = (
        'The tool returned answer "A".\n'
        '{"choice_analysis": {"A": {"evidence": "up", "confidence": 0.9}}, '
        '"answer": "A", "explanation": "trend up"}'
    )
    parsed, ok = parse_strict(raw, required_keys=["choice_analysis", "answer", "explanation"])
    assert ok
    assert parsed["answer"] == "A"


def test_parse_last_json_when_draft_precedes_final():
    raw = (
        "Draft:\n"
        '{"answer": "wrong", "explanation": "draft"}\n'
        "Final:\n"
        '{"choice_analysis": {}, "answer": "B", "explanation": "final"}'
    )
    parsed, ok = parse_strict(raw, required_keys=["choice_analysis", "answer", "explanation"])
    assert parsed["answer"] == "B"


def test_parse_verifier_with_markdown_fence():
    raw = (
        'Wrapper text\n{"verdict": "revised", "answer": "Investment expenditure", "reasoning": "source says 43%"}\n```'
    )
    parsed, ok = parse_strict(raw, required_keys=["verdict", "answer", "reasoning"])
    assert parsed["verdict"] == "revised"
    assert parsed["answer"] == "Investment expenditure"


def test_parse_qwen_thinking_block_stripped():
    raw = (
        "<think>Let me check the chart carefully. The bar for Q3 reaches 42%.</think>"
        '{"verdict": "confirmed", "answer": "42%", "reasoning": "bar reaches 42%"}'
    )
    parsed, ok = parse_strict(raw, required_keys=["verdict", "answer", "reasoning"])
    assert ok
    assert parsed["answer"] == "42%"


def test_parse_redacted_thinking_block_stripped():
    raw = (
        "<redacted_thinking>hidden reasoning</redacted_thinking>"
        '{"verdict": "revised", "answer": "15", "reasoning": "corrected value"}'
    )
    parsed, ok = parse_strict(raw, required_keys=["verdict", "answer", "reasoning"])
    assert ok
    assert parsed["verdict"] == "revised"


def test_repair_fallback_sets_parse_ok_false():
    raw = '{"answer": "A", "explanation": "trend up",}'  # trailing comma — invalid JSON
    parsed, ok = parse_strict(raw, required_keys=["answer", "explanation"])
    assert not ok
    assert parsed.get("answer") == "A"


def test_missing_required_keys_returns_empty():
    raw = '{"answer": "A"}'  # missing "explanation"
    parsed, ok = parse_strict(raw, required_keys=["answer", "explanation"])
    assert parsed == {}
    assert not ok


def test_no_required_keys_accepts_any_dict():
    raw = '{"foo": 1, "bar": "baz"}'
    parsed, ok = parse_strict(raw)
    assert ok
    assert parsed == {"foo": 1, "bar": "baz"}
