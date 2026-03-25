r"""Aggregate metrics.jsonl → summary.csv, grouped by config_name and question_type.

Usage:
    uv run --env-file .env -m agentfinvqa.eval.summarize \\
        --metrics metrics.jsonl \\
        --out summary.csv
"""

import argparse
import contextlib
import csv
import json
from collections import defaultdict
from typing import List


def load_metrics(path: str) -> List[dict]:
    """Load a metrics JSONL file and return a list of record dicts."""
    records = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    records.append(json.loads(line))
    return records


def _numeric_keys(records: List[dict]) -> List[str]:
    keys = []
    seen: set = set()
    for rec in records:
        for k, v in rec.items():
            if k not in seen and isinstance(v, (int, float)) and k != "sample_id":
                keys.append(k)
                seen.add(k)
    return keys


def aggregate(records: List[dict]) -> dict:
    """Compute mean and count for all numeric keys across a list of records."""
    if not records:
        return {}
    num_keys = _numeric_keys(records)
    result: dict = {"count": len(records)}
    for key in num_keys:
        vals = [r[key] for r in records if key in r and isinstance(r[key], (int, float))]
        if vals:
            result[f"{key}_mean"] = round(sum(vals) / len(vals), 4)
            result[f"{key}_n"] = len(vals)
    return result


def summarize(metrics: List[dict]) -> List[dict]:
    """Group metrics by config and question type, returning aggregated rows."""
    by_config: dict = defaultdict(list)
    by_config_qtype: dict = defaultdict(list)

    for m in metrics:
        cfg = m.get("config_name", "unknown")
        qt = m.get("question_type", "unknown")
        by_config[cfg].append(m)
        by_config_qtype[(cfg, qt)].append(m)

    rows = []

    # Overall per config
    for cfg, recs in sorted(by_config.items()):
        agg = aggregate(recs)
        rows.append({"config_name": cfg, "question_type": "ALL", **agg})

    # Per config × question_type
    for (cfg, qt), recs in sorted(by_config_qtype.items()):
        agg = aggregate(recs)
        rows.append({"config_name": cfg, "question_type": qt, **agg})

    return rows


def write_csv(rows: List[dict], path: str) -> None:
    """Write a list of dicts to a CSV file at the given path."""
    if not rows:
        print("No rows to write.")
        return
    # Collect all keys preserving order
    all_keys: list = []
    seen: set = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[dict]) -> None:
    print("\n--- Summary (ALL rows) ---")
    for row in rows:
        if row.get("question_type") != "ALL":
            continue
        cfg = row["config_name"]
        n = row.get("count", 0)
        acc = row.get("answer_accuracy_mean")
        lat = row.get("latency_sec_mean")
        ua = row.get("unanswerable_accuracy_mean")
        parts = [f"  {cfg}: n={n}"]
        if acc is not None:
            parts.append(f"accuracy={acc:.3f}")
        if ua is not None:
            parts.append(f"unanswerable_acc={ua:.3f}")
        if lat is not None:
            parts.append(f"latency={lat:.1f}s")
        print(", ".join(parts))


def main() -> None:
    """Parse CLI arguments and write an aggregated summary CSV."""
    parser = argparse.ArgumentParser(description="Aggregate metrics into summary CSV")
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl")
    parser.add_argument("--out", default="summary.csv", help="Output CSV path")
    args = parser.parse_args()

    metrics = load_metrics(args.metrics)
    print(f"Loaded {len(metrics)} metric records from {args.metrics}")

    rows = summarize(metrics)
    write_csv(rows, args.out)
    print(f"Summary written to {args.out}  ({len(rows)} rows)")
    _print_summary(rows)


if __name__ == "__main__":
    main()
