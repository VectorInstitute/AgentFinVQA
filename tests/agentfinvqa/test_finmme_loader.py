"""Unit tests for the FinMME dataset loader utilities."""

from PIL import Image

from agentfinvqa.datasets import finmme_loader
from agentfinvqa.datasets.perceived_sample import QuestionType


def test_parse_options_from_json():
    text = '["Option A", "Option B", ""]'
    assert finmme_loader._parse_options(text) == ["Option A", "Option B"]


def test_parse_options_from_labeled_text():
    text = "A. Growth\nB. Value\nC. Income"
    assert finmme_loader._parse_options(text) == ["Growth", "Value", "Income"]


def test_map_question_type_variants():
    assert finmme_loader._map_question_type("Multiple Choice") == QuestionType.MCQ
    assert finmme_loader._map_question_type("Conversational Q&A") == QuestionType.CONVERSATIONAL
    assert finmme_loader._map_question_type("Unanswerable") == QuestionType.UNANSWERABLE
    assert finmme_loader._map_question_type("Random Type") == QuestionType.STANDARD


def test_build_sample_writes_image(tmp_path):
    row = {
        "id": 7,
        "image": Image.new("RGB", (2, 2), color="white"),
        "question_text": "Select the correct metric.",
        "question_type": "multiple choice",
        "options": "A. Foo\nB. Bar",
        "answer": "A",
        "unit": "%",
        "tolerance": 0.1,
        "verified_caption": "A caption",
        "related_sentences": ["Foo is better"],
    }

    sample = finmme_loader._build_sample(0, row, tmp_path)

    assert sample.sample_id == "finmme_000007"
    assert sample.image_path.endswith("finmme_000007.png")
    assert sample.choices == ["Foo", "Bar"]
    assert sample.metadata["dataset"] == "FinMME"
    assert sample.expected_output == "Foo"
    assert sample.metadata["answer_label"] == "A"
    assert sample.metadata["choice_map"]["B"] == "Bar"


def test_build_sample_handles_multi_letter_answer(tmp_path):
    row = {
        "id": 8,
        "image": Image.new("RGB", (2, 2), color="white"),
        "question_text": "Pick the combo.",
        "question_type": "multiple choice",
        "options": ["Alpha", "Beta", "Gamma"],
        "answer": "AC",
    }

    sample = finmme_loader._build_sample(0, row, tmp_path)

    assert sample.expected_output == "Alpha + Gamma"
    assert sample.metadata["answer_label"] == "AC"
