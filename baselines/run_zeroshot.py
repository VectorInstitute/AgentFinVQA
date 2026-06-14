r"""Zero-shot VLM baseline — single direct VLM call, no agent pipeline.

Usage:
    .venv/bin/python baselines/run_zeroshot.py \
        --dataset chartqapro \
        --split test \
        --n 200 \
        --backend gemini \
        --model gemini-2.5-flash \
        --workers 8 \
        --out output/baselines/

    # Weaker baseline (bare prompt, first line = answer):
    #   ... --prompt_style minimal

Outputs a metrics JSONL to --out with the same schema as the agent pipeline
so results are directly comparable with summarize.py.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Make src importable when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

load_dotenv(_REPO_ROOT / ".env")

# ruff: noqa: E402, I001
from agentfinvqa.datasets.chartqapro_loader import load_chartqapro  # noqa: E402
from agentfinvqa.datasets.finmme_loader import load_finmme  # noqa: E402
from agentfinvqa.datasets.perceived_sample import PerceivedSample  # noqa: E402
from agentfinvqa.eval.eval_outputs import score_answer_accuracy  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset registry (mirrors run_generate_meps.py)
# ---------------------------------------------------------------------------
DATASET_CONFIGS: dict = {
    "chartqapro": {
        "loader": load_chartqapro,
        "display_name": "ChartQAPro",
        "default_image_dir": str(_REPO_ROOT / "data/chartqapro_images"),
    },
    "finmme": {
        "loader": load_finmme,
        "display_name": "FinMME",
        "default_image_dir": str(_REPO_ROOT / "data/finmme_images"),
    },
}

# ---------------------------------------------------------------------------
# Zero-shot prompt
# ---------------------------------------------------------------------------

# Structured prompt: explicit rules + JSON output format (current default)
_PROMPT_STRUCTURED = """Look at the chart image and answer the following question.

{context_block}Question: {question}
{choices_block}
Rules:
- Read axis labels, legend, and scale carefully before answering
- Give a single exact value unless the question asks for multiple
- Only answer UNANSWERABLE if the data is genuinely absent from the chart

Output ONLY this JSON (no markdown, no extra text):
{{"answer": "...", "explanation": "..."}}
If the question cannot be answered from the chart:
{{"answer": "UNANSWERABLE", "explanation": "..."}}"""

# Minimal prompt: bare question, no rules, no structured output hint
_PROMPT_MINIMAL = """{context_block}Question: {question}
{choices_block}Answer:"""


def _build_prompt(sample: PerceivedSample, prompt_style: str = "structured") -> str:
    context_block = ""
    if sample.context:
        lines = ["Conversation so far:"]
        for turn in sample.context:
            lines.append(f"  {turn.get('role', 'user')}: {turn.get('content', '')}")
        context_block = "\n".join(lines) + "\n\n"

    choices_block = ""
    if sample.choices:
        choices_block = "Choices: " + ", ".join(sample.choices) + "\n"

    template = _PROMPT_MINIMAL if prompt_style == "minimal" else _PROMPT_STRUCTURED
    return template.format(
        context_block=context_block,
        question=sample.question,
        choices_block=choices_block,
    )


# ---------------------------------------------------------------------------
# VLM callers (minimal, no CrewAI dependency)
# ---------------------------------------------------------------------------


def _encode_image(image_path: str) -> tuple[str, str]:
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def _call_openai(model: str, prompt: str, image_path: str, api_base: str = "") -> tuple[str, float]:
    from agentfinvqa.utils.model_compat import openai_temperature  # noqa: PLC0415
    from agentfinvqa.utils.openai_compat import build_openai_client, qwen35_extra_body  # noqa: PLC0415

    client = build_openai_client(api_key=os.environ.get("OPENAI_API_KEY", ""), api_base=api_base)
    b64, mime = _encode_image(image_path)
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                ],
            }
        ],
        max_completion_tokens=1024,
        extra_body=qwen35_extra_body(model),  # disables thinking on Qwen3.x; no-op otherwise
        **openai_temperature(model),
    )
    latency = time.time() - t0
    return response.choices[0].message.content or "", latency


def _call_gemini(model: str, prompt: str, image_path: str) -> tuple[str, float]:
    from google import genai  # noqa: PLC0415

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    b64, mime = _encode_image(image_path)
    raw_bytes = base64.b64decode(b64)
    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=[  # type: ignore[arg-type]
            genai.types.Part.from_bytes(data=raw_bytes, mime_type=f"image/{mime}"),
            genai.types.Part.from_text(text=prompt),
        ],
        config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=1024),
    )
    latency = time.time() - t0
    return response.text or "", latency


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


def _strip_code_fences(raw: str) -> str:
    """Remove common markdown code-fence wrappers from model output."""
    text = raw.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_structured_answer(raw_text: str) -> tuple[str, bool]:
    """Best-effort extraction for structured JSON answers."""
    cleaned = _strip_code_fences(raw_text)
    if not cleaned:
        return "", False

    # Happy path: full JSON object
    with contextlib.suppress(Exception):
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            ans = str(parsed.get("answer", "")).strip()
            return ans, bool(ans)

    # Common truncated JSON suffixes
    for suffix in ['"}', '"}}', '"}}}', '"\n}', '"\n}}']:
        with contextlib.suppress(Exception):
            parsed = json.loads(cleaned + suffix)
            if isinstance(parsed, dict):
                ans = str(parsed.get("answer", "")).strip()
                if ans:
                    return ans, True

    # Extract answer field directly even when explanation is truncated
    m = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, flags=re.DOTALL)
    if m:
        with contextlib.suppress(Exception):
            ans = json.loads(f'"{m.group(1)}"').strip()
            return ans, bool(ans)

    # Last-resort fallback for non-JSON responses
    m = re.search(r"(?im)^\s*answer\s*[:\-]\s*(.+?)\s*$", cleaned)
    if m:
        ans = m.group(1).strip().strip('"').strip()
        return ans, bool(ans)

    return "", False


def _run_sample(
    sample: PerceivedSample,
    backend: str,
    model: str,
    config_name: str,
    prompt_style: str = "structured",
    api_base: str = "",
) -> dict:
    prompt = _build_prompt(sample, prompt_style)
    raw_text = ""
    latency = 0.0
    error = None

    try:
        if backend == "openai":
            raw_text, latency = _call_openai(model, prompt, sample.image_path, api_base=api_base)
        else:
            raw_text, latency = _call_gemini(model, prompt, sample.image_path)
    except Exception as exc:
        error = str(exc)
        raw_text = json.dumps({"answer": "ERROR", "explanation": error})

    # Parse answer
    predicted = ""
    parse_ok = False
    if prompt_style == "minimal":
        # Raw text IS the answer — just clean it up
        predicted = raw_text.strip().splitlines()[0].strip()
        parse_ok = bool(predicted)
    else:
        predicted, parse_ok = _extract_structured_answer(raw_text)

    # Score — mirror evaluate_mep MCQ label-to-text expansion
    expected = sample.expected_output
    # If expected is a single letter (MCQ), strip trailing text from predicted
    # so "c) 42%" → "c" scores correctly against expected "C"
    scored_predicted = predicted
    if re.fullmatch(r"[A-Da-d]", expected.strip()):
        m = re.match(r"^\s*([A-Da-d])\s*[).:]?\s*", predicted.strip())
        if m:
            scored_predicted = m.group(1)
    accuracy = score_answer_accuracy(expected, scored_predicted, sample.question_type.value)

    row: dict = {
        "sample_id": sample.sample_id,
        "question_type": sample.question_type.value,
        "config_name": config_name,
        "expected": expected,
        "predicted": predicted,
        "raw_response": raw_text,
        "answer_accuracy": accuracy,
        "latency_sec": round(latency, 4),
        "parse_ok": parse_ok,
        "has_errors": error is not None,
    }
    if error:
        row["error"] = error
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    """CLI entry point for zero-shot VLM baseline evaluation."""
    parser = argparse.ArgumentParser(description="Zero-shot VLM baseline")
    parser.add_argument("--dataset", default="chartqapro", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=100, help="Number of samples (0 = all)")
    parser.add_argument("--backend", default="gemini", choices=["openai", "gemini"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="output/baselines/", help="Output directory")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-completed samples")
    parser.add_argument(
        "--prompt_style",
        default="structured",
        choices=["structured", "minimal"],
        help="structured: rules + JSON output (default); minimal: bare question only",
    )
    parser.add_argument(
        "--api_base",
        default="",
        help=(
            "OpenAI-compatible base URL (e.g. local vLLM endpoint). When set with "
            "--backend openai, routes calls to this server instead of api.openai.com. "
            "Falls back to OPENAI_BASE_URL env var when blank."
        ),
    )
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]
    image_dir = args.image_dir or ds_cfg["default_image_dir"]
    # Slugify the model name for filenames — replace separators that can't appear
    # in filenames (`/` is the killer for HF org/name pairs like Qwen/Qwen3.6-...).
    model_slug = args.model.replace("-", "_").replace(".", "_").replace("/", "_")
    config_name = f"zeroshot_{args.prompt_style}_{args.backend}_{model_slug}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"metrics_{args.dataset}_{args.split}_{config_name}.jsonl"

    print(f"Dataset  : {ds_cfg['display_name']} / {args.split}")
    print(f"Model    : {args.backend} / {args.model}")
    print(f"Prompt   : {args.prompt_style}")
    print(f"Workers  : {args.workers}")
    print(f"Output   : {out_file}")

    # Load samples
    samples: list[PerceivedSample] = ds_cfg["loader"](
        split=args.split,
        image_dir=image_dir,
        n=args.n if args.n > 0 else None,
    )
    print(f"Samples  : {len(samples)}")

    # Resume: skip already-written sample IDs
    done_ids: set[str] = set()
    if args.resume and out_file.exists():
        with open(out_file) as f:
            for line in f:
                with contextlib.suppress(Exception):
                    done_ids.add(json.loads(line)["sample_id"])
        print(f"Resume   : skipping {len(done_ids)} already done")

    todo = [s for s in samples if s.sample_id not in done_ids]

    # Run in parallel
    completed = 0
    errors = 0
    mode = "a" if args.resume else "w"
    with open(out_file, mode) as f_out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_sample, s, args.backend, args.model, config_name, args.prompt_style, args.api_base): s
            for s in todo
        }
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                s = futures[future]
                print(f"  ERROR {s.sample_id}: {exc}")
                errors += 1
                continue
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()
            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(todo)} done ...")

    print(f"\nDone: {completed} samples, {errors} errors → {out_file}")

    # Quick accuracy summary
    with open(out_file) as _f:
        rows = [json.loads(line) for line in _f if line.strip()]
    if rows:
        acc = statistics.mean(r["answer_accuracy"] for r in rows)
        lat = statistics.mean(r["latency_sec"] for r in rows)
        print(f"Accuracy : {acc:.4f}   Latency: {lat:.2f}s")


if __name__ == "__main__":
    main()
