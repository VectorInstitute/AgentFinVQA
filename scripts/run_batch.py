#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Run dataset batches with single-pass post-evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _ensure_src_on_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


_ensure_src_on_path()

import contextlib

from agentfinvqa.eval.error_taxonomy import classify_failure
from agentfinvqa.eval.eval_outputs import evaluate_mep
from agentfinvqa.eval.eval_traces import evaluate_trace
from agentfinvqa.eval.summarize import summarize, write_csv
from agentfinvqa.langfuse_integration.client import get_client
from agentfinvqa.mep.writer import iter_meps, mep_dataset_split_relpath
from agentfinvqa.runner import run_generate_meps

CONFIG_CHOICES = list(run_generate_meps.BACKEND_CONFIGS.keys())
DATASET_CHOICES = list(run_generate_meps.DATASET_CONFIGS.keys())


def _build_runner_args(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--n",
        str(args.n),
        "--config",
        args.config,
        "--workers",
        str(args.workers),
        "--out",
        args.out,
    ]

    def _append(flag: str, value: str | None) -> None:
        if value:
            cmd.extend([flag, value])

    _append("--image_dir", args.image_dir)
    _append("--cache_dir", args.cache_dir)
    _append("--planner_model", args.planner_model)
    _append("--vision_model", args.vision_model)
    _append("--verifier_model", args.verifier_model)
    _append("--ocr_model", args.ocr_model)

    if args.no_verifier:
        cmd.append("--no_verifier")
    if args.no_ocr:
        cmd.append("--no_ocr")
    if args.no_legend_grounding:
        cmd.append("--no_legend_grounding")
    _append("--run_tag", args.run_tag)
    if not args.langfuse:
        cmd.append("--no_langfuse")
    if args.resume:
        cmd.append("--resume")

    return cmd


def _eval_mep(
    mep: dict,
    config: dict,
    use_judge: bool,
    judge_model: str,
    lf_client: Any,
) -> tuple[dict, dict, dict | None]:
    """Evaluate a single MEP. Returns (metrics_row, trace_row, taxonomy_row | None)."""
    sid = mep.get("sample", {}).get("sample_id", "?")

    row = evaluate_mep(
        mep,
        use_judge=use_judge,
        judge_backend=config.get("judge_backend", config["planner_backend"]),
        judge_model=judge_model,
    )

    if lf_client:
        lf_trace_id = mep.get("lf_trace_id")
        if lf_trace_id:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    with contextlib.suppress(Exception):
                        lf_client.create_score(trace_id=lf_trace_id, name=k, value=float(v))

    trace_row = evaluate_trace(mep)

    tax_row = None
    acc_val = row.get("answer_accuracy", 0.0)
    if acc_val < 0.999:
        result = classify_failure(
            mep,
            answer_accuracy=acc_val,
            backend=config.get("vision_backend", "gemini"),
            model=judge_model,
        )
        tax_row = {"sample_id": sid, "answer_accuracy": acc_val, **result}

    return row, trace_row, tax_row


def run_post_evaluation(
    out_dir: Path,
    label: str,
    eval_dir: Path,
    config: dict,
    use_judge: bool,
    use_langfuse: bool = False,
    workers: int = 4,
    judge_model: str | None = None,
    mep_start: int = 0,
) -> None:
    """Compute metrics, traces, taxonomy, and summary using a thread pool."""
    if not out_dir.exists():
        print(f"[post-eval] Skipping: MEP dir {out_dir} not found")
        return

    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[post-eval] Writing artifacts to {eval_dir.resolve()} (label={label}, workers={workers})")

    metrics_path = eval_dir / f"metrics_{label}.jsonl"
    trace_path = eval_dir / f"trace_metrics_{label}.jsonl"
    taxonomy_path = eval_dir / f"taxonomy_{label}.jsonl"

    judge_model = judge_model or config.get("vision_model") or config.get("planner_model") or "gemini-2.5-flash-lite"
    lf_client = get_client() if use_langfuse else None

    # Accumulate all completed results in memory, then sort by sample_id and
    # write at the end. ThreadPoolExecutor yields futures in completion order,
    # which is non-deterministic across workers; sorting makes the on-disk
    # artifacts deterministic and diff-friendly between runs.
    results: list[tuple[str, dict, dict, dict | None]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        all_meps = list(iter_meps(str(out_dir)))
        if mep_start > 0:
            print(f"[post-eval] Skipping first {mep_start} MEPs (mep_start={mep_start}, total={len(all_meps)})")
            all_meps = all_meps[mep_start:]
        futures = {pool.submit(_eval_mep, mep, config, use_judge, judge_model, lf_client): mep for mep in all_meps}
        for future in as_completed(futures):
            mep = futures[future]
            sid = mep.get("sample", {}).get("sample_id", "?")
            try:
                row, trace_row, tax_row = future.result()
            except Exception as exc:
                print(f"  [eval] {sid} ERROR: {exc}")
                continue
            results.append((sid, row, trace_row, tax_row))

    if not results:
        print("[post-eval] No metrics produced.")
        return

    results.sort(key=lambda t: t[0])
    metrics_list: list[dict] = [row for _, row, _, _ in results]
    taxonomy_rows = sum(1 for _, _, _, tax in results if tax)

    with (
        open(metrics_path, "w") as f_met,
        open(trace_path, "w") as f_trace,
        open(taxonomy_path, "w") as f_tax,
    ):
        for _, row, trace_row, tax_row in results:
            f_met.write(json.dumps(row) + "\n")
            f_trace.write(json.dumps(trace_row) + "\n")
            if tax_row:
                f_tax.write(json.dumps(tax_row) + "\n")

    acc = sum(m.get("answer_accuracy", 0.0) for m in metrics_list) / len(metrics_list)
    print(f"[post-eval] Accuracy {acc:.1%} (n={len(metrics_list)})")
    qt_groups: dict[str, list[float]] = defaultdict(list)
    for row in metrics_list:
        qt_groups[row.get("question_type", "unknown")].append(row.get("answer_accuracy", 0.0))
    for qt, vals in sorted(qt_groups.items()):
        print(f"  {qt:<18} {sum(vals) / max(len(vals), 1):.1%} (n={len(vals)})")

    print(f"[post-eval] Trace metrics  -> {trace_path}")
    print(f"[post-eval] Taxonomy rows: {taxonomy_rows} -> {taxonomy_path}")

    summary_path = eval_dir / f"summary_{label}.csv"
    write_csv(summarize(metrics_list), str(summary_path))
    print(f"[post-eval] Summary -> {summary_path}")

    print("[post-eval] Artifacts ready:")
    for path in [metrics_path, trace_path, taxonomy_path, summary_path]:
        print(f"  - {path}")


def main() -> None:
    """Run dataset generation and optional post-evaluation in one command."""
    parser = argparse.ArgumentParser(description="Run AgentFinVQA on a dataset at scale")
    parser.add_argument("--dataset", default="finmme", choices=DATASET_CHOICES, help="Dataset to run")
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Max samples (0 or negative = entire split after --split slice)",
    )
    parser.add_argument("--split", default="train", help="Dataset split slice (default train).")
    parser.add_argument("--config", default="gemini_gemini", choices=CONFIG_CHOICES, help="Backend pairing")
    parser.add_argument("--workers", type=int, default=4, help="Threaded workers for generation")
    parser.add_argument("--out", default="meps/", help="Base output directory for MEPs")
    parser.add_argument("--image_dir", default=None, help="Override image cache directory")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace datasets cache override")
    parser.add_argument("--planner_model", default=None, help="Planner model override")
    parser.add_argument("--vision_model", default=None, help="Vision model override")
    parser.add_argument("--verifier_model", default=None, help="Verifier model override")
    parser.add_argument("--ocr_model", default=None, help="OCR model override")
    parser.add_argument("--no_verifier", action="store_true", help="Disable verifier")
    parser.add_argument("--no_ocr", action="store_true", help="Disable OCR preread")
    parser.add_argument("--no_legend_grounding", action="store_true", help="Disable legend grounding stage")
    parser.add_argument("--run_tag", default=None, help="Subfolder tag within dataset dir (e.g. planner_v2)")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse tracing")
    parser.add_argument("--resume", action="store_true", help="Skip samples whose MEPs already exist")
    parser.add_argument("--use_judge", action="store_true", help="Run LLM judge during post-eval")
    parser.add_argument(
        "--judge_model", default="gemini-2.5-flash-lite", help="Judge model (default: gemini-2.5-flash-lite)"
    )
    parser.add_argument("--mep_start", type=int, default=0, help="Skip first N MEPs (default: 0 = process all)")
    parser.add_argument("--post_eval", action="store_true", help="Run post-eval after generation")
    parser.add_argument("--eval_only", action="store_true", help="Skip generation, run post-eval on existing MEPs")
    parser.add_argument("--eval_dir", default="output", help="Directory for eval outputs")
    parser.add_argument("--eval_label", default=None, help="Label for eval files (defaults to config_n)")
    parser.add_argument("--env_file", default=".env", help=".env file to load before running")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path, override=True)

    if not args.eval_only:
        sys.argv = ["run_generate_meps"] + _build_runner_args(args)
        run_generate_meps.main()

    if args.post_eval or args.eval_only:
        config = run_generate_meps.BACKEND_CONFIGS[args.config]
        out_dir = (
            Path(args.out)
            / f"{config['planner_backend']}_{config['vision_backend']}"
            / mep_dataset_split_relpath(
                args.dataset, args.split, no_verifier=args.no_verifier, no_ocr=args.no_ocr, run_tag=args.run_tag
            )
        )
        label = args.eval_label or (f"{args.config}_all" if args.n <= 0 else f"{args.config}_n{args.n}")
        run_post_evaluation(
            out_dir=out_dir,
            label=label,
            eval_dir=Path(args.eval_dir),
            config=config,
            use_judge=args.use_judge,
            use_langfuse=args.langfuse,
            workers=args.workers,
            judge_model=args.judge_model,
            mep_start=args.mep_start,
        )


if __name__ == "__main__":
    main()
