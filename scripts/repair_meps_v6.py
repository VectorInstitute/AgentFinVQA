"""
Repair fixes_v6_conf_gate MEPs: re-parse verifier.raw_text using the corrected
required-keys list (confidence is optional, not required).

Run:
    python scripts/repair_meps_v6.py
"""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentfinvqa.utils.json_strict import parse_strict


VERIFIER_REQUIRED_KEYS = ["verdict", "answer", "reasoning"]  # confidence is optional
VALID_VERDICTS = {"confirmed", "revised"}

MEP_DIR = Path("meps/gemini_gemini/finmme/fixes_v6_conf_gate/train")


def repair_mep(path: Path) -> str:
    """Return one of: 'fixed', 'already_ok', 'no_raw', 'unparsable'."""
    data = json.loads(path.read_text())
    verifier = data.get("verifier") or {}
    verdict = verifier.get("verdict", "skipped")

    if verdict in VALID_VERDICTS and verifier.get("parse_error") is False:
        return "already_ok"

    raw_text = (verifier.get("raw_text") or "").strip()
    if not raw_text:
        return "no_raw"

    parsed, parse_ok = parse_strict(raw_text, required_keys=VERIFIER_REQUIRED_KEYS)
    if not parsed:
        return "unparsable"

    # Normalise verdict
    raw_verdict = str(parsed.get("verdict", "")).lower()
    if raw_verdict not in VALID_VERDICTS:
        raw_verdict = "confirmed"
    parsed["verdict"] = raw_verdict

    # Clamp optional confidence
    try:
        conf = float(parsed.get("confidence", 1.0))
        parsed["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        parsed["confidence"] = 1.0 if raw_verdict == "confirmed" else 0.5

    # Apply confidence gate: downgrade low-confidence revisions
    gated = False
    if raw_verdict == "revised":
        if parsed["confidence"] < 0.75:
            old_reasoning = parsed.get("reasoning", "")[:80]
            parsed["verdict"] = "confirmed"
            parsed["answer"] = (data.get("vision") or {}).get("parsed", {}).get("answer", "")
            parsed["reasoning"] = (
                f"[Confidence gate: revision confidence {parsed['confidence']:.2f} < 0.75 — "
                f"keeping vision answer. Original reasoning: {old_reasoning}]"
            )
            gated = True

    # Update verifier block
    verifier["parsed"] = parsed
    verifier["verdict"] = parsed["verdict"]
    verifier["parse_error"] = not parse_ok

    # Clean up errors list: remove verifier_invalid_output entries
    errors = [e for e in (data.get("errors") or []) if "verifier_invalid_output" not in str(e)]
    if gated:
        errors.append(f"verifier_revision_gated: confidence={parsed['confidence']:.2f}")
    data["errors"] = errors

    path.write_text(json.dumps(data, indent=2))
    return "gated" if gated else "fixed"


def main():
    results = {"fixed": 0, "gated": 0, "already_ok": 0, "no_raw": 0, "unparsable": 0}
    paths = sorted(MEP_DIR.glob("*.json"))
    print(f"Repairing {len(paths)} MEPs in {MEP_DIR} ...")
    for i, p in enumerate(paths, 1):
        outcome = repair_mep(p)
        results[outcome] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(paths)} processed ...")

    print("\nDone.")
    for k, v in results.items():
        print(f"  {k:<14}: {v}")


if __name__ == "__main__":
    main()
