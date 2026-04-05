#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Convenience entrypoint for running FinMME batches outside notebooks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _ensure_src_on_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


_ensure_src_on_path()

from agentfinvqa.eval.error_taxonomy import classify_failure
from agentfinvqa.eval.eval_outputs import evaluate_mep
from agentfinvqa.eval.eval_traces import evaluate_trace
from agentfinvqa.eval.summarize import summarize, write_csv
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
    if not args.langfuse:
        cmd.append("--no_langfuse")
    if args.resume:
        cmd.append("--resume")

    return cmd


def _write_metrics(out_dir: Path, metrics_path: Path, config: dict, use_judge: bool) -> list[dict]:
    metrics_list: list[dict] = []
    judge_model = config.get("vision_model") or config.get("planner_model") or "gemini-2.5-flash-lite"
    with open(metrics_path, "w") as f_out:
        for mep in iter_meps(str(out_dir)):
            try:
                row = evaluate_mep(
                    mep,
                    use_judge=use_judge,
                    judge_backend=config.get("judge_backend", config["planner_backend"]),
                    judge_model=judge_model,
                )
                metrics_list.append(row)
                f_out.write(json.dumps(row) + "\n")
            except Exception as exc:
                sid = mep.get("sample", {}).get("sample_id", "?")
                print(f"  [metrics] {sid} ERROR: {exc}")
    return metrics_list


def _print_accuracy(metrics_list: list[dict]) -> None:
    acc = sum(m.get("answer_accuracy", 0.0) for m in metrics_list) / len(metrics_list)
    print(f"[post-eval] Accuracy {acc:.1%} (n={len(metrics_list)})")
    qt_groups: dict[str, list[float]] = defaultdict(list)
    for row in metrics_list:
        qt_groups[row.get("question_type", "unknown")].append(row.get("answer_accuracy", 0.0))
    for qt, vals in sorted(qt_groups.items()):
        mean_qt = sum(vals) / max(len(vals), 1)
        print(f"  {qt:<18} {mean_qt:.1%} (n={len(vals)})")


def _write_trace_metrics(out_dir: Path, trace_path: Path) -> None:
    with open(trace_path, "w") as f_out:
        for mep in iter_meps(str(out_dir)):
            row = evaluate_trace(mep)
            f_out.write(json.dumps(row) + "\n")
    print(f"[post-eval] Trace metrics -> {trace_path}")


def _write_taxonomy(out_dir: Path, taxonomy_path: Path, metrics_list: list[dict], config: dict) -> None:
    acc_map = {row["sample_id"]: row.get("answer_accuracy", 0.0) for row in metrics_list}
    taxonomy_rows = 0
    model = config.get("vision_model") or config.get("planner_model") or "gemini-2.5-flash-lite"
    with open(taxonomy_path, "w") as f_out:
        for mep in iter_meps(str(out_dir)):
            sid = mep.get("sample", {}).get("sample_id", "")
            acc_val = acc_map.get(sid, 0.0)
            if acc_val >= 0.999:
                continue
            try:
                result = classify_failure(
                    mep,
                    answer_accuracy=acc_val,
                    backend=config.get("vision_backend", "gemini"),
                    model=model,
                )
                row = {"sample_id": sid, "answer_accuracy": acc_val, **result}
                f_out.write(json.dumps(row) + "\n")
                taxonomy_rows += 1
            except Exception as exc:
                print(f"  [taxonomy] {sid} ERROR: {exc}")
    print(f"[post-eval] Taxonomy rows: {taxonomy_rows} -> {taxonomy_path}")


def _write_summary(metrics_list: list[dict], summary_path: Path) -> None:
    summary_rows = summarize(metrics_list)
    write_csv(summary_rows, str(summary_path))
    print(f"[post-eval] Summary -> {summary_path}")


def run_post_evaluation(
    out_dir: Path,
    label: str,
    eval_dir: Path,
    config: dict,
    use_judge: bool,
) -> None:
    """Compute metrics, traces, taxonomy, and summary artifacts for a run."""
    if not out_dir.exists():
        print(f"[post-eval] Skipping metrics: MEP dir {out_dir} not found")
        return

    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[post-eval] Writing artifacts to {eval_dir.resolve()} (label={label})")

    metrics_path = eval_dir / f"metrics_{label}.jsonl"
    metrics_list = _write_metrics(out_dir, metrics_path, config, use_judge)

    if not metrics_list:
        print("[post-eval] No metrics produced; aborting remaining steps.")
        return

    _print_accuracy(metrics_list)
    trace_path = eval_dir / f"trace_metrics_{label}.jsonl"
    _write_trace_metrics(out_dir, trace_path)
    taxonomy_path = eval_dir / f"taxonomy_{label}.jsonl"
    _write_taxonomy(out_dir, taxonomy_path, metrics_list, config)
    summary_path = eval_dir / f"summary_{label}.csv"
    _write_summary(metrics_list, summary_path)
    print("[post-eval] Artifacts ready:")
    for path in [metrics_path, trace_path, taxonomy_path, summary_path]:
        print(f"  - {path}")


def main() -> None:
    """Run FinMME generation and optional post-evaluation in one command."""
    parser = argparse.ArgumentParser(description="Run AgentFinVQA on a dataset at scale")
    parser.add_argument("--dataset", default="finmme", choices=DATASET_CHOICES, help="Dataset to run")
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Max samples (0 or negative = entire split after --split slice)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split slice (default train).",
    )
    parser.add_argument("--config", default="gemini_gemini", choices=CONFIG_CHOICES, help="Backend pairing")
    parser.add_argument("--workers", type=int, default=4, help="Threaded workers for generation")
    parser.add_argument("--out", default="meps/", help="Base output directory for MEPs")
    parser.add_argument("--image_dir", default=None, help="Override FinMME image cache directory")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace datasets cache override")
    parser.add_argument("--planner_model", default=None, help="Planner model override")
    parser.add_argument("--vision_model", default=None, help="Vision model override")
    parser.add_argument("--verifier_model", default=None, help="Verifier model override")
    parser.add_argument("--ocr_model", default=None, help="OCR model override")
    parser.add_argument("--no_verifier", action="store_true", help="Disable pass 2.5 verifier")
    parser.add_argument("--no_ocr", action="store_true", help="Disable OCR preread")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse registration/tracing")
    parser.add_argument("--resume", action="store_true", help="Skip samples whose MEPs already exist in the output dir")
    parser.add_argument("--use_judge", action="store_true", help="Run LLM judge during post-eval")
    parser.add_argument("--post_eval", action="store_true", help="Compute metrics/taxonomy/summary after generation")
    parser.add_argument("--eval_dir", default="output", help="Directory for metrics/taxonomy outputs")
    parser.add_argument("--eval_label", default=None, help="Custom label for eval files (defaults to config_n)")
    parser.add_argument("--env_file", default=".env", help=".env file to load before running")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path, override=True)

    sys.argv = ["run_generate_meps"] + _build_runner_args(args)
    run_generate_meps.main()

    if args.post_eval:
        config = run_generate_meps.BACKEND_CONFIGS[args.config]
        out_dir = (
            Path(args.out)
            / f"{config['planner_backend']}_{config['vision_backend']}"
            / mep_dataset_split_relpath(args.dataset, args.split, no_verifier=args.no_verifier)
        )
        label = args.eval_label or (f"{args.config}_all" if args.n <= 0 else f"{args.config}_n{args.n}")
        run_post_evaluation(
            out_dir=out_dir,
            label=label,
            eval_dir=Path(args.eval_dir),
            config=config,
            use_judge=args.use_judge,
        )


if __name__ == "__main__":
    main()
