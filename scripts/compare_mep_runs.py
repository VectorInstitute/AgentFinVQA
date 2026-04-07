#!/usr/bin/env python3
"""Compare two MEP directories using the same rule-based scoring as evaluate_mep.

Unlike naive string comparison, this applies FinMME choice_map expansion so
letter answers (e.g. B, AB) match gold phrasing.

Example:
  uv run python scripts/compare_mep_runs.py \\
    meps/gemini_gemini/finmme/no_legend_grounding/train \\
    meps/gemini_gemini/finmme/fixes_v1/train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentfinvqa.eval.eval_outputs import evaluate_mep
from agentfinvqa.mep.writer import read_mep


def _repo_root() -> Path:
    """
    Return the root directory of the repository (one parent up from this script).

    Returns
    -------
    Path
        The absolute path to the repository's root.
    """
    return Path(__file__).resolve().parents[1]


def _ensure_src() -> None:
    """Add repo ``src`` to ``sys.path`` so imports work from any cwd."""
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_index(mep_dir: Path) -> dict[str, dict]:
    """
    Load a mapping of sample IDs to MEP output dicts from a directory.

    Parameters
    ----------
    mep_dir : Path
        Directory containing MEP .json result files.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping sample ids (file stems) to parsed MEP dicts.
    """
    out: dict[str, dict] = {}
    for p in sorted(mep_dir.glob("*.json")):
        out[p.stem] = read_mep(str(p))
    return out


def main() -> None:  # noqa: PLR0912, PLR0915
    """Compare two MEP run dirs with FinMME-aware scoring and optional JSON output.

    Parse CLI args, validate dirs, evaluate overlapping samples, print a summary,
    and write per-sample metrics when ``--out-json`` is set.
    """
    _ensure_src()

    parser = argparse.ArgumentParser(description="Compare two MEP run directories (production-aligned scoring).")
    parser.add_argument("baseline_dir", type=Path, help="First run (e.g. no_legend_grounding/train)")
    parser.add_argument("candidate_dir", type=Path, help="Second run (e.g. fixes_v1/train)")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM judge per sample (slow; needs API keys). Default is rule-based accuracy only.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional JSON path: per-sample metrics for baseline and candidate",
    )
    args = parser.parse_args()

    base_dir = args.baseline_dir.resolve()
    cand_dir = args.candidate_dir.resolve()
    if not base_dir.is_dir():
        print(f"Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)
    if not cand_dir.is_dir():
        print(f"Not a directory: {cand_dir}", file=sys.stderr)
        sys.exit(1)

    baseline = _load_index(base_dir)
    candidate = _load_index(cand_dir)
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        print("No overlapping sample IDs between the two directories.", file=sys.stderr)
        sys.exit(1)

    use_judge = args.judge
    rows: list[dict] = []
    n_base = n_cand = 0
    flip_wrong_to_right: list[str] = []
    flip_right_to_wrong: list[str] = []
    same_right = same_wrong = 0

    for sid in ids:
        mb = baseline[sid]
        mc = candidate[sid]
        try:
            rb = evaluate_mep(mb, use_judge=use_judge)
            rc = evaluate_mep(mc, use_judge=use_judge)
        except Exception as exc:
            print(f"[{sid}] evaluate_mep failed: {exc}", file=sys.stderr)
            continue

        ab = float(rb.get("answer_accuracy", 0.0))
        ac = float(rc.get("answer_accuracy", 0.0))
        ok_b = ab >= 0.999
        ok_c = ac >= 0.999
        if ok_b:
            n_base += 1
        if ok_c:
            n_cand += 1

        if ok_b and ok_c:
            same_right += 1
        elif not ok_b and not ok_c:
            same_wrong += 1
        elif not ok_b and ok_c:
            flip_wrong_to_right.append(sid)
        else:
            flip_right_to_wrong.append(sid)

        rows.append(
            {
                "sample_id": sid,
                "baseline_accuracy": ab,
                "candidate_accuracy": ac,
                "baseline_expected": rb.get("expected"),
                "baseline_predicted": rb.get("predicted"),
                "candidate_expected": rc.get("expected"),
                "candidate_predicted": rc.get("predicted"),
            }
        )

    n = len(rows)
    print(f"Compared {n} samples (intersection of both dirs).")
    print(f"  Baseline accuracy : {n_base / n:.1%}  ({n_base}/{n})")
    print(f"  Candidate accuracy: {n_cand / n:.1%}  ({n_cand}/{n})")
    print(f"  Same correct      : {same_right}")
    print(f"  Same wrong        : {same_wrong}")
    print(f"  Wrong → correct   : {len(flip_wrong_to_right)}")
    print(f"  Correct → wrong   : {len(flip_right_to_wrong)}")
    if flip_wrong_to_right:
        print(f"  Wrong→correct ids : {', '.join(flip_wrong_to_right)}")
    if flip_right_to_wrong:
        print(f"  Correct→wrong ids: {', '.join(flip_right_to_wrong)}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(
                {
                    "baseline_dir": str(base_dir),
                    "candidate_dir": str(cand_dir),
                    "n_compared": n,
                    "summary": {
                        "baseline_accuracy": n_base / n if n else 0.0,
                        "candidate_accuracy": n_cand / n if n else 0.0,
                        "flip_wrong_to_right": flip_wrong_to_right,
                        "flip_right_to_wrong": flip_right_to_wrong,
                    },
                    "per_sample": rows,
                },
                f,
                indent=2,
            )
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
