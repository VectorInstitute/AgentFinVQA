r"""Async batch-mode MEP evaluation using the Gemini Batch API.

Three subcommands:

  submit    — upload prompts, create job, save state file, exit immediately
  status    — check job state without blocking
  retrieve  — download results and write metrics JSONL (run after job completes)

Usage:
    # 1. Submit (returns immediately, prints job name)
    uv run --env-file .env -m agentfinvqa.eval.eval_outputs_batch submit \
        --mep_dir meps/gemini_gemini/finmme/train[1500:3000] \
        --out output/metrics_1500_3000.jsonl \
        --judge_model gemini-2.5-flash-lite

    # 2. Check status any time
    uv run --env-file .env -m agentfinvqa.eval.eval_outputs_batch status \
        --state output/metrics_1500_3000.jsonl.batch_state.json

    # 3. Retrieve when done
    uv run --env-file .env -m agentfinvqa.eval.eval_outputs_batch retrieve \
        --state output/metrics_1500_3000.jsonl.batch_state.json
"""

import argparse
import contextlib
import json
import os

from dotenv import load_dotenv

from ..langfuse_integration.client import get_client
from ..mep.writer import iter_meps
from .eval_outputs import evaluate_mep
from .judge_batch import get_job_state, retrieve_batch_results, submit_batch_job


load_dotenv()


def _state_path(out: str) -> str:
    return out + ".batch_state.json"


def _save_state(state_file: str, job_name: str, mep_dir: str, out: str, model: str) -> None:
    with open(state_file, "w") as f:
        json.dump(
            {
                "job_name": job_name,
                "mep_dir": mep_dir,
                "out": out,
                "model": model,
                "commands": {
                    "status": f"uv run --env-file .env -m agentfinvqa.eval.eval_outputs_batch status --state {state_file}",
                    "retrieve": f"uv run --env-file .env -m agentfinvqa.eval.eval_outputs_batch retrieve --state {state_file}",
                },
            },
            f,
            indent=2,
        )
    print(f"State saved to {state_file}")


def _load_state(state_file: str) -> dict:
    with open(state_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit a batch job and write the local state file."""
    print(f"Loading MEPs from {args.mep_dir} …")
    meps = list(iter_meps(args.mep_dir))
    print(f"Loaded {len(meps)} MEPs")

    display_name = os.path.splitext(os.path.basename(args.out))[0]
    job_name = submit_batch_job(meps, model=args.judge_model, display_name=display_name)

    state_file = _state_path(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _save_state(state_file, job_name, args.mep_dir, args.out, args.judge_model)

    print(f"\nJob submitted: {job_name}")
    print(f"Check status : uv run -m agentfinvqa.eval.eval_outputs_batch status --state {state_file}")
    print(f"Retrieve     : uv run -m agentfinvqa.eval.eval_outputs_batch retrieve --state {state_file}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check the current job state for a submitted batch."""
    state = _load_state(args.state)
    job_name = state["job_name"]
    current = get_job_state(job_name)
    print(f"Job   : {job_name}")
    print(f"State : {current}")
    if current == "JOB_STATE_SUCCEEDED":
        print(f"Ready to retrieve → uv run -m agentfinvqa.eval.eval_outputs_batch retrieve --state {args.state}")
    elif current in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
        print("Job did not succeed — resubmit if needed.")


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Download batch results, merge with MEPs, and write metrics."""
    state = _load_state(args.state)
    job_name = state["job_name"]
    mep_dir = state["mep_dir"]
    out = state["out"]

    # Check state first to give a clear error if not done yet
    current = get_job_state(job_name)
    if current != "JOB_STATE_SUCCEEDED":
        print(f"Job not ready (state={current}). Run 'status' to check again.")
        return

    print(f"Downloading results for {job_name} …")
    judge_scores = retrieve_batch_results(job_name)

    print(f"Loading MEPs from {mep_dir} …")
    meps = list(iter_meps(mep_dir))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    count = 0
    with open(out, "w") as f_out:
        for mep in meps:
            sid = mep.get("sample", {}).get("sample_id", "")
            try:
                metrics = evaluate_mep(mep, use_judge=False)
                sample_judge_scores = judge_scores.get(sid, {})
                for k, v in sample_judge_scores.items():
                    metrics[f"judge_{k}"] = v
                lf_trace_id = mep.get("lf_trace_id")
                if lf_trace_id and sample_judge_scores:
                    client = get_client()
                    if client:
                        for k, v in sample_judge_scores.items():
                            if isinstance(v, (int, float)):
                                with contextlib.suppress(Exception):
                                    client.create_score(trace_id=lf_trace_id, name=f"judge_{k}", value=float(v))
                f_out.write(json.dumps(metrics) + "\n")
                count += 1
            except Exception as exc:
                print(f"  Error on {sid}: {exc}")

    print(f"Done. {count} metrics written to {out}")
    # Clean up state file
    os.unlink(args.state)
    print(f"State file removed: {args.state}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI for submit/status/retrieve batch evaluation workflows."""
    parser = argparse.ArgumentParser(
        description="Async batch MEP evaluation via Gemini Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = sub.add_parser("submit", help="Upload prompts and create batch job")
    p_submit.add_argument("--mep_dir", required=True)
    p_submit.add_argument("--out", required=True, help="Output metrics JSONL path (used to name state file)")
    p_submit.add_argument("--judge_model", default="gemini-2.5-flash-lite")

    # status
    p_status = sub.add_parser("status", help="Check batch job status")
    p_status.add_argument("--state", required=True, help="Path to .batch_state.json from submit")

    # retrieve
    p_retrieve = sub.add_parser("retrieve", help="Download results and write metrics JSONL")
    p_retrieve.add_argument("--state", required=True, help="Path to .batch_state.json from submit")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
