"""Fix zero-shot metrics JSONL in-place without re-running API calls.

Repairs two issues:
  1. MCQ scoring: "c) 42%" was scored 0 against expected "C" — extract leading letter.
  2. Truncated JSON: tries to recover answer from cut-off raw_response by closing the
     JSON.

Usage:
    .venv/bin/python baselines/fix_zeroshot_scores.py \
        output/baselines/metrics_chartqapro_test_zeroshot_gemini_gemini_2_5_flash.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from agentfinvqa.eval.eval_outputs import score_answer_accuracy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_raw(raw: str) -> str | None:
    """Try to extract an answer from a (possibly truncated) raw JSON string."""
    raw = raw.strip()
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()

    # Happy path
    with contextlib.suppress(Exception):
        return json.loads(raw).get("answer", "")

    # Truncated mid-value: close the string and object
    for suffix in ['"}', '"}}', '"}}}']:
        with contextlib.suppress(Exception):
            return json.loads(raw + suffix).get("answer", "")

    # Extract answer field with regex as last resort
    m = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.DOTALL)
    if m:
        with contextlib.suppress(Exception):
            return json.loads(f'"{m.group(1)}"')
        return m.group(1)

    return None


def _normalize_mcq(predicted: str, expected: str) -> str:
    """If expected is a single MCQ letter, strip trailing text from predicted."""
    if not re.fullmatch(r"[A-Da-d]", expected.strip()):
        return predicted
    m = re.match(r"^\s*([A-Da-d])\s*[).:]?\s*", predicted.strip())
    return m.group(1) if m else predicted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fix_file(path: Path) -> None:
    """Fix scoring errors in a metrics JSONL file in-place."""
    rows = [json.loads(line) for line in path.open() if line.strip()]

    reparsed = fixed_score = 0
    out_rows = []

    for row in rows:
        raw = row.get("raw_response", "")
        expected = row.get("expected", "")
        old_predicted = row.get("predicted", "")
        old_accuracy = row.get("answer_accuracy", 0.0)
        old_parse_ok = row.get("parse_ok", False)
        qt = row.get("question_type", "standard")

        # Step 1: re-parse if truncated
        new_predicted = old_predicted
        new_parse_ok = old_parse_ok
        if not old_parse_ok and raw:
            recovered = _try_parse_raw(raw)
            if recovered is not None:
                new_predicted = recovered
                new_parse_ok = True
                reparsed += 1

        # Step 2: normalize MCQ letter
        scored_predicted = _normalize_mcq(new_predicted, expected)

        # Step 3: re-score (cast to str in case of numeric values in the data)
        new_accuracy = score_answer_accuracy(str(expected), str(scored_predicted), qt)

        if new_accuracy != old_accuracy or new_predicted != old_predicted:
            fixed_score += 1

        row["predicted"] = new_predicted
        row["parse_ok"] = new_parse_ok
        row["answer_accuracy"] = new_accuracy
        out_rows.append(row)

    # Backup original
    backup = path.with_suffix(".jsonl.bak")
    shutil.copy(path, backup)
    print(f"Backup  → {backup}")

    # Write fixed file
    with open(path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    total = len(out_rows)
    acc = sum(r["answer_accuracy"] for r in out_rows) / total if total else 0
    print(f"Rows    : {total}")
    print(f"Reparsed truncated : {reparsed}")
    print(f"Score changed      : {fixed_score}")
    print(f"New accuracy       : {acc:.4f}")
    print(f"Fixed  → {path}")


def main() -> None:
    """CLI entry point: fix scoring errors in one or more metrics JSONL files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSONL file(s) to fix")
    args = parser.parse_args()
    for f in args.files:
        print(f"\n=== {f} ===")
        fix_file(Path(f))


if __name__ == "__main__":
    main()
