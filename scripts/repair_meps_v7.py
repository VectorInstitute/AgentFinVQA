"""
Repair fixes_v7_g3flash_conf_gate MEPs: the confidence gate incorrectly fired
for all verifier revisions because the model never returned a 'confidence' field,
causing the default of 0.0 to trigger the < 0.75 gate threshold.

Fix: un-gate revisions where 'confidence' is NOT present in raw_text.
Only keep gated if the model explicitly returned confidence < 0.75.

Run:
    python scripts/repair_meps_v7.py
"""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentfinvqa.utils.json_strict import parse_strict


VERIFIER_REQUIRED_KEYS = ["verdict", "answer", "reasoning"]
VALID_VERDICTS = {"confirmed", "revised"}
CONFIDENCE_GATE_THRESHOLD = 0.75

MEP_DIR = Path("meps/gemini_gemini/finmme/fixes_v7_g3flash_conf_gate/train")


def repair_mep(path: Path) -> str:
    """Return one of: 'ungated', 'gate_kept', 'already_ok', 'no_raw', 'unparsable'."""
    data = json.loads(path.read_text())
    errors = data.get("errors") or []
    verifier = data.get("verifier") or {}

    was_gated = any("verifier_revision_gated" in str(e) for e in errors)
    if not was_gated:
        return "already_ok"

    raw_text = (verifier.get("raw_text") or "").strip()
    if not raw_text:
        return "no_raw"

    parsed, parse_ok = parse_strict(raw_text, required_keys=VERIFIER_REQUIRED_KEYS)
    if not parsed:
        return "unparsable"

    raw_verdict = str(parsed.get("verdict", "")).lower()
    if raw_verdict not in VALID_VERDICTS:
        raw_verdict = "confirmed"

    # Check if the model explicitly returned a confidence value
    confidence_in_raw = "confidence" in parsed
    if confidence_in_raw:
        try:
            explicit_conf = float(parsed["confidence"])
            explicit_conf = max(0.0, min(1.0, explicit_conf))
        except (TypeError, ValueError):
            explicit_conf = CONFIDENCE_GATE_THRESHOLD  # neutral if unparsable
    else:
        explicit_conf = CONFIDENCE_GATE_THRESHOLD  # model didn't return it → neutral, don't gate

    # Only keep gated if confidence was explicit AND below threshold
    should_gate = (raw_verdict == "revised") and confidence_in_raw and (explicit_conf < CONFIDENCE_GATE_THRESHOLD)

    if should_gate:
        # Gate is legitimately correct — leave as-is
        return "gate_kept"

    # Un-gate: restore the verifier's actual verdict/answer
    parsed["verdict"] = raw_verdict
    parsed["confidence"] = explicit_conf

    verifier["parsed"] = parsed
    verifier["verdict"] = raw_verdict
    verifier["parse_error"] = not parse_ok

    # Remove the incorrect gate error
    errors = [e for e in errors if "verifier_revision_gated" not in str(e)]
    data["errors"] = errors

    path.write_text(json.dumps(data, indent=2))
    return "ungated"


def main():
    results = {"ungated": 0, "gate_kept": 0, "already_ok": 0, "no_raw": 0, "unparsable": 0}
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
