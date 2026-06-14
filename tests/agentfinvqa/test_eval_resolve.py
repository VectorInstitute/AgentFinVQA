"""Tests for FinMME choice expansion used in rule-based eval."""

import math

from agentfinvqa.eval.eval_outputs import _to_number, resolve_eval_answers, score_answer_accuracy


def test_resolve_letter_to_full_choice_text():
    """Map a single letter to full choice text using ``choice_map``."""
    sample = {
        "expected_output": "Sell or avoid, as the stock shows a declining trend",
        "question_type": "mcq",
        "metadata": {
            "answer_label": "B",
            "choice_map": {
                "A": "Buy, as the stock price will increase",
                "B": "Sell or avoid, as the stock shows a declining trend",
                "C": "Hold, as there is potential for recovery",
                "D": "Diversify, as the stock is too volatile",
            },
        },
    }
    exp, pred = resolve_eval_answers(sample, "B")
    assert exp == pred == "Sell or avoid, as the stock shows a declining trend"


def test_resolve_multi_select_letters():
    """Join multi-select letters into full choice text; accuracy is 1.0 when exact."""
    sample = {
        "expected_output": "US + Germany",
        "question_type": "mcq",
        "metadata": {
            "answer_label": "AB",
            "choice_map": {
                "A": "US",
                "B": "Germany",
                "C": "Spain",
                "D": "Japan",
            },
        },
    }
    exp, pred = resolve_eval_answers(sample, "AB")
    assert exp == pred == "US + Germany"
    acc = score_answer_accuracy(exp, pred, "mcq")
    assert acc == 1.0


def test_no_spurious_numeric_match_on_product_names():
    """Do not treat product names with digits as matching numbers spuriously."""
    assert _to_number("jericho2") is None
    assert _to_number("jericho2e ramon") is None
    s = score_answer_accuracy("Jericho2e + Ramon", "Jericho2", "mcq")
    assert s < 1.0


def test_numeric_tolerance_still_works_for_plain_numbers():
    """Phrase-wrapped numbers still parse and can match plain numeric gold."""
    assert _to_number("approximately 115") is not None
    assert math.isclose(score_answer_accuracy("115", "approximately 115", "standard"), 1.0)


def test_resolve_predicted_full_text_unchanged():
    """Keep full-text predictions; still expand expected from labels when needed."""
    sample = {
        "expected_output": "US + Germany",
        "question_type": "mcq",
        "metadata": {
            "answer_label": "AB",
            "choice_map": {"A": "US", "B": "Germany", "C": "Spain", "D": "Japan"},
        },
    }
    exp, pred = resolve_eval_answers(sample, "US, Germany")
    assert exp == "US + Germany"
    assert "US" in pred and "Germany" in pred
