"""Gemini Batch API judge for large-scale MEP evaluation.

Submits all judge prompts as a single batch job (50% cost, async).
Use eval_outputs_batch.py as the CLI entry point.
"""

import contextlib
import json
import os
import tempfile
import time
from typing import Optional

from google import genai
from google.genai import types

from ..utils.json_strict import parse_strict
from .judge import _JUDGE_KEYS, _JUDGE_PROMPT, _default_scores


def _build_judge_prompt(mep: dict) -> str:
    """Build the judge prompt string for a single MEP."""
    sample = mep.get("sample", {})
    plan = mep.get("plan", {}).get("parsed", {})
    parsed_vision = mep.get("vision", {}).get("parsed", {})
    steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan.get("steps", []))) or "  (none)"
    return _JUDGE_PROMPT.format(
        question=sample.get("question", "") or "unknown",
        expected_answer=sample.get("expected_output", ""),
        agent_answer=parsed_vision.get("answer", ""),
        agent_explanation=parsed_vision.get("explanation", ""),
        plan_steps=steps_text,
    )


_COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _write_requests_jsonl(meps: list[dict], display_name: str) -> str:
    """Write judge requests to a JSONL temp file and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for mep in meps:
            sid = mep.get("sample", {}).get("sample_id", "")
            entry = {
                "key": sid,
                "request": {
                    "contents": [{"parts": [{"text": _build_judge_prompt(mep)}]}],
                    "generation_config": {"temperature": 0, "max_output_tokens": 512},
                },
            }
            tmp.write(json.dumps(entry) + "\n")
        tmp.flush()
        return tmp.name


def _extract_response_text(parsed: dict) -> str:
    text = ""
    with contextlib.suppress(KeyError, IndexError, TypeError):
        text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"]
    return text


def _parse_batch_content(content: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    errors = 0
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
            key = parsed.get("key", "")
            text = _extract_response_text(parsed)
            if text:
                scores, _ = parse_strict(text, required_keys=_JUDGE_KEYS)
                results[key] = scores if scores else _default_scores()
            else:
                results[key] = _default_scores()
                errors += 1
        except Exception:
            errors += 1
    print(f"  Parsed {len(results)} results ({errors} errors)")
    return results


def _require_job_state(job: types.BatchJob) -> str:
    if job.state is None:
        raise RuntimeError("Batch job state is missing.")
    return job.state.name


def _require_dest_file(job: types.BatchJob) -> str:
    if job.dest is None or job.dest.file_name is None:
        raise RuntimeError("Batch job output file is missing.")
    return job.dest.file_name


def batch_judge_meps(
    meps: list[dict],
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
    poll_interval: int = 30,
    display_name: str = "agentfinvqa-judge-batch",
) -> dict[str, dict]:
    """Submit all MEPs to the Gemini Batch API and return {sample_id: scores}.

    Uses the file-based approach (recommended for >20 requests).
    Blocks until the job completes or fails — run in a screen/tmux session
    or via SLURM for large batches.

    Parameters
    ----------
    meps : list of dict
        MEP dicts to judge.
    model : str
        Gemini model name (must support Batch API).
    api_key : str, optional
        Gemini API key; defaults to GEMINI_API_KEY env var.
    poll_interval : int
        Seconds between status polls.
    display_name : str
        Display name for the batch job.

    Returns
    -------
    dict[str, dict]
        Maps sample_id → judge score dict (same keys as judge_mep).
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))

    # Build JSONL — one request per MEP
    tmp_path = _write_requests_jsonl(meps, display_name)

    try:
        # Upload JSONL
        print(f"  Uploading {len(meps)} judge requests …")
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )
        if uploaded.name is None:
            raise RuntimeError("Upload succeeded without a file name.")
        print(f"  Uploaded: {uploaded.name}")

        # Create batch job
        job = client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": display_name},
        )
        if job.name is None:
            raise RuntimeError("Batch job creation returned no job name.")
        job_name = job.name
        print(f"  Batch job created: {job_name}")

        # Poll until done
        while True:
            job = client.batches.get(name=job_name)
            state = _require_job_state(job)
            if state in _COMPLETED_STATES:
                break
            print(f"  [{state}] waiting {poll_interval}s …")
            time.sleep(poll_interval)

        print(f"  Batch job finished: {state}")

        if state != "JOB_STATE_SUCCEEDED":
            print(f"  Error: {getattr(job, 'error', 'unknown')}")
            return {}

        # Download and parse results
        file_name = _require_dest_file(job)
        content = client.files.download(file=file_name).decode("utf-8")
        return _parse_batch_content(content)

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Split submit / retrieve for async (fire-and-forget) workflows
# ---------------------------------------------------------------------------


def submit_batch_job(
    meps: list[dict],
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
    display_name: str = "agentfinvqa-judge-batch",
) -> str:
    """Upload judge prompts and create a Gemini Batch job. Returns the job name.

    Does NOT poll — call retrieve_batch_results later with the returned job name.
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))

    tmp_path = _write_requests_jsonl(meps, display_name)

    try:
        print(f"  Uploading {len(meps)} judge requests …")
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )
        if uploaded.name is None:
            raise RuntimeError("Upload succeeded without a file name.")
        job = client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": display_name},
        )
        if job.name is None:
            raise RuntimeError("Batch job creation returned no job name.")
        print(f"  Batch job created: {job.name}")
        return job.name
    finally:
        os.unlink(tmp_path)


def get_job_state(job_name: str, api_key: Optional[str] = None) -> str:
    """Return the current state string for a batch job."""
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
    job = client.batches.get(name=job_name)
    return _require_job_state(job)


def retrieve_batch_results(
    job_name: str,
    api_key: Optional[str] = None,
) -> dict[str, dict]:
    """Download and parse results for a completed batch job.

    Raises RuntimeError if the job has not succeeded yet.
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
    job = client.batches.get(name=job_name)
    state = _require_job_state(job)

    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Job {job_name} is not complete (state={state})")

    file_name = _require_dest_file(job)
    content = client.files.download(file=file_name).decode("utf-8")
    return _parse_batch_content(content)
