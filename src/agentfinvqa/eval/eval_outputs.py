r"""Pass 1: output-based evaluation — rule-based accuracy + LLM judge rubric.

Usage:
    uv run --env-file .env -m agentfinvqa.eval.eval_outputs \\
        --mep_dir meps/openai_gemini/chartqapro/test \\
        --out metrics.jsonl
"""

import argparse
import contextlib
import json
import math
import os
import re
from typing import Optional

from dotenv import load_dotenv

from ..langfuse_integration.client import get_client
from ..mep.writer import iter_meps
from .judge import judge_mep


load_dotenv()


# ---------------------------------------------------------------------------
# Rule-based scorers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = text.strip().lower()
    # Strip leading choice-letter prefix from some models (e.g. "B: sell or avoid…").
    text = re.sub(r"^[a-z]:\s+", "", text)
    text = re.sub(r"[^\w\s\-\.]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_number(text: str) -> Optional[float]:
    """Extract a scalar for numeric tolerance.

    Returns None for product-style tokens (e.g. Jericho2).
    """
    text = text.replace(",", "").replace("%", "").strip()
    # Skip digit-after-letter patterns (Jericho2, 3Q20); avoids false matches like
    # 1.0 vs Jericho2e + Ramon.
    if re.search(r"[a-z][0-9]", text.lower()):
        return None
    m = re.search(r"-?\d+\.?\d*", text)
    try:
        return float(m.group()) if m else None
    except ValueError:
        return None


def score_answer_accuracy(expected: str, predicted: str, question_type: str) -> float:
    """Exact-match with numeric tolerance and MCQ partial credit."""
    exp = _normalize(expected)
    pred = _normalize(predicted)

    if exp == pred:
        return 1.0

    # Numeric tolerance check (5% relative, 0.5 absolute for small values)
    exp_num = _to_number(exp)
    pred_num = _to_number(pred)
    if (
        exp_num is not None and pred_num is not None and math.isclose(exp_num, pred_num, rel_tol=0.001, abs_tol=0.5)
    ):  # tighter tolerance for chart QA, can adjust as needed
        return 1.0

    # MCQ substring check
    if question_type == "mcq" and (exp in pred or pred in exp):
        return 0.5

    return 0.0


def score_unanswerable(expected: str, predicted: str) -> Optional[float]:
    """Score binary classification for unanswerable samples.

    Returns None if expected is NOT UNANSWERABLE (metric not applicable).
    """
    exp_ua = expected.strip().upper() == "UNANSWERABLE"
    pred_ua = predicted.strip().upper() == "UNANSWERABLE"
    if exp_ua:
        return 1.0 if pred_ua else 0.0
    return None


# ---------------------------------------------------------------------------
# Choice mapping helpers (for datasets like FinMME)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_choice_labels(answer: str, choice_map: dict[str, str]) -> list[str]:
    """Extract ordered choice labels (e.g. ['A','C']) from an answer string."""
    if not answer or not choice_map:
        return []
    answer = answer.strip()
    # Direct match against full choice text
    normalized_map = {_normalize_text(text): label for label, text in choice_map.items()}
    normalized_answer = _normalize_text(answer)
    if normalized_answer in normalized_map:
        return [normalized_map[normalized_answer]]

    # Letter combo detection (e.g. "ACD")
    letters_only = re.sub(r"[^A-Za-z]", "", answer).upper()
    if letters_only and all(ch in choice_map for ch in letters_only):
        labels: list[str] = []
        for ch in letters_only:
            if ch not in labels:
                labels.append(ch)
        return labels

    # Direct substring match against choice text
    labels = []
    for norm_text, label in normalized_map.items():
        if norm_text and norm_text in normalized_answer and label not in labels:
            labels.append(label)
    if labels:
        return labels

    # Split via connectors (+, commas, "and", etc.)
    parts = re.split(r"\s*(?:\+|,|/|;|&|\band\b|\|)\s*", answer, flags=re.IGNORECASE)
    for part in parts:
        norm = _normalize_text(part)
        if not norm:
            continue
        maybe_label = normalized_map.get(norm)
        if maybe_label and maybe_label not in labels:
            labels.append(maybe_label)
    return labels


def _labels_to_text(labels: list[str], choice_map: dict[str, str]) -> str:
    if not labels or not choice_map:
        return ""
    return " + ".join(choice_map.get(label, label) for label in labels)


def resolve_eval_answers(sample: dict, raw_predicted: str) -> tuple[str, str]:
    """Expand FinMME ``choice_map`` / ``answer_label`` into comparable strings.

    Maps MCQ letter answers (e.g. ``\"B\"``, ``\"AB\"``) to the same canonical
    phrasing as gold labels so rule-based accuracy matches ``evaluate_mep``.
    """
    expected = sample.get("expected_output", "")
    metadata = sample.get("metadata") or {}
    choice_map = metadata.get("choice_map") or {}
    expected_labels = _parse_choice_labels(metadata.get("answer_label", ""), choice_map)
    predicted_labels = _parse_choice_labels(raw_predicted, choice_map)
    if expected_labels:
        expected = _labels_to_text(expected_labels, choice_map)
    predicted = (raw_predicted or "").strip()
    if predicted_labels:
        predicted = _labels_to_text(predicted_labels, choice_map)
    return expected, predicted


# ---------------------------------------------------------------------------
# Per-MEP evaluation
# ---------------------------------------------------------------------------


def evaluate_mep(
    mep: dict,
    use_judge: bool = True,
    judge_backend: str = "gemini",
    judge_model: str = "gemini-2.5-flash-lite",
) -> dict:
    """Evaluate a single MEP and return a metrics dict."""
    sample = mep.get("sample", {})
    plan = mep.get("plan", {})
    vision = mep.get("vision", {})
    timestamps = mep.get("timestamps", {})
    config = mep.get("config", {})

    vision_parsed = vision.get("parsed", {})
    verifier = mep.get("verifier") or {}
    verifier_parsed = verifier.get("parsed") or {}
    verifier_verdict = verifier.get("verdict", "skipped")

    # Final answer: use verifier output when it ran, otherwise fall back to vision
    raw_predicted = verifier_parsed.get("answer") or vision_parsed.get("answer", "")
    expected, predicted = resolve_eval_answers(sample, raw_predicted)
    question_type = sample.get("question_type", "standard")

    planner_ms = timestamps.get("planner_ms") or 0
    vision_ms = timestamps.get("vision_ms") or 0
    verifier_ms = timestamps.get("verifier_ms") or 0

    metrics: dict = {
        "sample_id": sample.get("sample_id", ""),
        "question_type": question_type,
        "config_name": config.get("config_name", ""),
        "expected": expected,
        "predicted": predicted,
        "vision_answer": vision_parsed.get("answer", ""),  # raw vision answer pre-verification
        "verifier_verdict": verifier_verdict,
        "planner_parse_ok": not plan.get("parse_error", True),
        "vision_parse_ok": not vision.get("parse_error", True),
        "json_parse_ok": (not plan.get("parse_error", True)) and (not vision.get("parse_error", True)),
        "answer_accuracy": score_answer_accuracy(expected, predicted, question_type),
        "latency_sec": (planner_ms + vision_ms + verifier_ms) / 1000.0,
        "tool_call_count": len(vision.get("tool_trace", [])),
        "has_errors": len(mep.get("errors", [])) > 0,
    }

    ua = score_unanswerable(expected, predicted)
    if ua is not None:
        metrics["unanswerable_accuracy"] = ua

    if use_judge:
        judge_scores = judge_mep(mep, backend=judge_backend, model=judge_model)
        for k, v in judge_scores.items():
            metrics[f"judge_{k}"] = v

    # Log all scores back to the Langfuse trace if one was recorded in the MEP
    lf_trace_id = mep.get("lf_trace_id")
    if lf_trace_id:
        client = get_client()
        if client:
            score_keys = ["answer_accuracy", "latency_sec"] + (
                [f"judge_{k}" for k in judge_scores] if use_judge else []
            )
            scores = {k: metrics[k] for k in score_keys if isinstance(metrics.get(k), (int, float))}
            for k, v in scores.items():
                with contextlib.suppress(Exception):
                    client.create_score(trace_id=lf_trace_id, name=k, value=float(v))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Evaluate MEPs and write output-based metrics to JSONL."""
    parser = argparse.ArgumentParser(description="Evaluate MEPs — output-based metrics")
    parser.add_argument("--mep_dir", required=True, help="Directory containing MEP JSON files")
    parser.add_argument("--out", default="metrics.jsonl", help="Output JSONL file")
    parser.add_argument("--no_judge", action="store_true", help="Skip LLM judge (faster)")
    parser.add_argument("--judge_backend", default="gemini", choices=["openai", "gemini"])
    parser.add_argument("--judge_model", default="gemini-2.5-flash-lite")
    args = parser.parse_args()

    use_judge = not args.no_judge

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f_out:
        count = 0
        for mep in iter_meps(args.mep_dir):
            try:
                metrics = evaluate_mep(
                    mep,
                    use_judge=use_judge,
                    judge_backend=args.judge_backend,
                    judge_model=args.judge_model,
                )
                f_out.write(json.dumps(metrics) + "\n")
                count += 1
                if count % 10 == 0:
                    print(f"  evaluated {count} samples …")
            except Exception as exc:
                print(f"  Error evaluating MEP: {exc}")

    print(f"Done. {count} metrics written to {args.out}")


if __name__ == "__main__":
    main()
