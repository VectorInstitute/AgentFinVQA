r"""Runner: generate Model Evaluation Packets (MEPs) for ChartQAPro.

Usage:
    uv run --env-file .env -m agentfinvqa.runner.run_generate_meps \\
        --dataset chartqapro \\
        --split test \\
        --n 200 \\
        --config openai_gemini \\
        --workers 8 \\
        --out meps/
"""

import argparse
import contextlib
import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from dotenv import load_dotenv

from ..agents.planner_agent import PlannerAgent
from ..agents.verifier_agent import VerifierAgent
from ..agents.vision_agent import VisionAgent
from ..datasets.chartqapro_loader import load_chartqapro
from ..datasets.finmme_loader import load_finmme
from ..datasets.perceived_sample import PerceivedSample
from ..langfuse_integration.client import get_client
from ..langfuse_integration.dataset import register_dataset
from ..langfuse_integration.prompts import push_prompts
from ..langfuse_integration.tracing import (
    log_trace_scores,
    sample_trace,
)
from ..mep.schema import (
    MEP,
    ImageRef,
    MEPConfig,
    MEPLegendGrounding,
    MEPOcr,
    MEPPlan,
    MEPSample,
    MEPTimestamps,
    MEPVerifier,
    MEPVision,
)
from ..mep.writer import mep_dataset_split_relpath, write_mep
from ..tools.legend_grounder_tool import LegendGrounderTool
from ..tools.ocr_reader_tool import OcrReaderTool
from ..utils.hashing import sha256_file
from ..utils.json_strict import parse_strict
from ..utils.timing import iso_now, timed


load_dotenv()

# ---------------------------------------------------------------------------
# Backend configuration presets
# ---------------------------------------------------------------------------

BACKEND_CONFIGS: dict = {
    "openai_openai": {
        "planner_backend": "openai",
        "planner_model": "gpt-4o",
        "vision_backend": "openai",
        "vision_model": "gpt-4o",
        "judge_backend": "openai",
    },
    "gemini_gemini": {
        "planner_backend": "gemini",
        "planner_model": "gemini-2.5-flash-lite",
        "vision_backend": "gemini",
        "vision_model": "gemini-2.5-flash-lite",
        "judge_backend": "gemini",
    },
    # Full Flash (non-lite) — higher capability, higher cost
    "gemini_gemini_flash": {
        "planner_backend": "gemini",
        "planner_model": "gemini-2.5-flash",
        "vision_backend": "gemini",
        "vision_model": "gemini-2.5-flash",
        "judge_backend": "gemini",
    },
    # Gemini 2.5 Flash Preview — latest preview tier
    "gemini_gemini_flash_preview": {
        "planner_backend": "gemini",
        "planner_model": "gemini-2.5-flash-preview-04-17",
        "vision_backend": "gemini",
        "vision_model": "gemini-2.5-flash-preview-04-17",
        "judge_backend": "gemini",
    },
    "openai_gemini": {
        "planner_backend": "openai",
        "planner_model": "gpt-4o",
        "vision_backend": "gemini",
        "vision_model": "gemini-2.5-flash-lite",
        "judge_backend": "openai",
    },
    "gemini_openai": {
        "planner_backend": "gemini",
        "planner_model": "gemini-2.5-flash-lite",
        "vision_backend": "openai",
        "vision_model": "gpt-4o",
        "judge_backend": "gemini",
    },
}

DatasetLoader = Callable[..., list[PerceivedSample]]


class DatasetConfig(TypedDict):
    """Loader metadata for supported evaluation datasets."""

    loader: DatasetLoader
    display_name: str
    default_image_dir: str


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "chartqapro": {
        "loader": load_chartqapro,
        "display_name": "ChartQAPro",
        "default_image_dir": "data/chartqapro_images",
    },
    "finmme": {
        "loader": load_finmme,
        "display_name": "FinMME",
        "default_image_dir": "data/finmme_images",
    },
}

# Chart types where legend grounding is worth the extra VLM call
_LEGEND_GROUNDING_CHART_TYPES = {
    "line",
    "bar",
    "scatter",
    "area",
    "bar_grouped",
    "bar_stacked",
    "combination",
    "pie",
    "donut",
}

# Fallback plan used when the planner fails entirely
_FALLBACK_PLAN = {
    "steps": [
        "Identify the chart type and read the title",
        "Locate axes labels, legend entries, and series names",
        "Extract the data values relevant to the question",
        "Check whether the question is answerable from the chart",
    ],
    "expected_answer_type": "string",
    "question_type": "standard",
    "answerability_check": "uncertain",
    "hints": [],
}


def _run_verifier_stage(
    verifier_agent: Optional[VerifierAgent],
    sample: PerceivedSample,
    plan_parsed: dict,
    vision_parsed: dict,
    lf_trace: Any,
) -> tuple[str, dict, bool, str, float, str, list, list[str]]:
    """Run the optional verifier agent and normalize its outputs."""
    if verifier_agent is None:
        return "", {}, False, "", 0.0, "skipped", [], []

    errors: list[str] = []
    verifier_prompt = ""
    verifier_parsed: dict = {}
    verifier_parse_error = False
    verifier_raw = ""
    verifier_traces: list = []
    verifier_ms = 0.0
    verifier_verdict = "skipped"
    valid_verifier_verdicts = {"confirmed", "revised"}

    try:
        with timed() as vrt:
            (
                verifier_prompt,
                verifier_parsed,
                verifier_parse_error,
                verifier_raw,
                verifier_traces,
            ) = verifier_agent.run(sample, plan_parsed, vision_parsed, lf_trace=lf_trace)
        verifier_ms = vrt.elapsed_ms
        raw_verdict_value = verifier_parsed.get("verdict")
        raw_verdict = raw_verdict_value if isinstance(raw_verdict_value, str) else None
        invalid_output = verifier_parse_error or raw_verdict not in valid_verifier_verdicts
        if invalid_output:
            if not verifier_parse_error:
                verifier_parse_error = True
            errors.append(f"verifier_invalid_output: {raw_verdict or 'missing verdict'}")
            verifier_verdict = "error"
            if not verifier_parsed:
                verifier_parsed = {}
            verifier_parsed.setdefault("verdict", "error")
            verifier_parsed.setdefault("answer", vision_parsed.get("answer", ""))
            verifier_parsed.setdefault(
                "reasoning",
                "Verifier output malformed or unavailable; defaulting to error.",
            )
        else:
            assert raw_verdict is not None
            verifier_verdict = raw_verdict

            # ── Confidence gate: downgrade low-confidence revisions to confirmed ──
            # The verifier self-reports confidence (0–1). When it wants to revise
            # but isn't confident enough, we keep the vision answer instead.
            if verifier_verdict == "revised":
                try:
                    ver_confidence = float(verifier_parsed.get("confidence", 0.75))
                except (TypeError, ValueError):
                    ver_confidence = 0.75
                if ver_confidence < 0.75:
                    verifier_verdict = "confirmed"
                    verifier_parsed["verdict"] = "confirmed"
                    verifier_parsed["answer"] = vision_parsed.get("answer", "")
                    verifier_parsed["reasoning"] = (
                        f"[Confidence gate: revision confidence {ver_confidence:.2f} < 0.75 — "
                        f"keeping vision answer. Original reasoning: {verifier_parsed.get('reasoning', '')[:80]}]"
                    )
                    errors.append(f"verifier_revision_gated: confidence={ver_confidence:.2f}")
    except Exception as exc:  # pragma: no cover - verifier optional
        errors.append(f"verifier_error: {exc}")
        verifier_parse_error = True
        verifier_parsed = {
            "verdict": "error",
            "answer": vision_parsed.get("answer", ""),
            "reasoning": f"Verifier crashed: {exc}",
        }
        verifier_verdict = "error"
        traceback.print_exc()

    return (
        verifier_prompt,
        verifier_parsed,
        verifier_parse_error,
        verifier_raw,
        verifier_ms,
        verifier_verdict,
        verifier_traces,
        errors,
    )


def process_sample(  # noqa: PLR0912, PLR0915
    sample: PerceivedSample,
    planner: PlannerAgent,
    vision_agent: VisionAgent,
    config: dict,
    run_id: str,
    out_dir: str,
    dataset_name: str = "ChartQAPro",
    lf_client: Any = None,
    verifier_agent: Optional[VerifierAgent] = None,
    ocr_tool: Optional[OcrReaderTool] = None,
    legend_grounder: Optional[LegendGrounderTool] = None,
) -> str:
    """
    Execute the multi-stage evaluation pipeline for a single sample.

    Coordinates the planner, optional OCR tool, vision agent, and
    optional verifier to produce a Model Evaluation Packet (MEP).

    Parameters
    ----------
    sample : PerceivedSample
        The data sample containing the question and chart image.
    planner : PlannerAgent
        The agent responsible for generating the inspection plan.
    vision_agent : VisionAgent
        The agent that performs visual analysis.
    config : dict
        Configuration dictionary for backends and models.
    run_id : str
        Unique identifier for the current evaluation run.
    out_dir : str
        Directory where the resulting MEP JSON should be saved.
    dataset_name : str, default 'ChartQAPro'
        Human-readable dataset label for logging and artifacts.
    langfuse_client : object, optional
        The Langfuse client for tracing and observability.
    verifier_agent : VerifierAgent, optional
        The agent for pass 2.5 verification.
    ocr_tool : OcrReaderTool, optional
        Tool for pre-extracting text from the chart.

    Returns
    -------
    str
        The absolute path to the written MEP file.
    """
    config_name = f"{config['planner_backend']}_{config['vision_backend']}"
    dataset_slug = dataset_name.lower().replace(" ", "_")
    run_start = iso_now()
    errors: list = []

    with sample_trace(
        lf_client,
        sample_id=sample.sample_id,
        question=sample.question,
        expected_output=sample.expected_output,
        question_type=sample.question_type.value,
        config_name=config_name,
        run_id=run_id,
        dataset_slug=dataset_slug,
    ) as lf_trace:
        lf_trace_id = getattr(lf_trace, "id", None)

        # ---- Planner ----
        plan_prompt = ""
        plan_parsed: dict = {}
        plan_parse_error = True
        plan_raw = ""
        plan_ms = 0.0

        try:
            with timed() as pt:
                plan_prompt, plan_parsed, plan_parse_error, plan_raw = planner.run(sample, lf_trace=lf_trace)

            plan_ms = pt.elapsed_ms
        except Exception as exc:
            errors.append(f"planner_error: {exc}")
            plan_parsed = dict(_FALLBACK_PLAN)
            plan_parsed["question_type"] = sample.question_type.value
            plan_parse_error = True
            traceback.print_exc()

        # ---- OCR pre-read (optional) ----
        ocr_parsed: dict = {}
        ocr_raw = ""
        ocr_parse_error = False
        ocr_traces: list = []
        ocr_ms = 0.0

        if ocr_tool is not None:
            try:
                ocr_tool.lf_trace = lf_trace
                with timed() as t:
                    ocr_raw = ocr_tool._run(sample.image_path)
                ocr_ms = t.elapsed_ms
                ocr_traces = ocr_tool.pop_traces()
                ocr_parsed, ocr_ok = parse_strict(ocr_raw, required_keys=["chart_type", "title"])
                ocr_parse_error = not ocr_ok
                if not ocr_parsed:
                    ocr_parsed = {}
            except Exception as exc:
                errors.append(f"ocr_error: {exc}")
                ocr_parse_error = True
                traceback.print_exc()

        # ---- Legend grounding (optional, between OCR and vision) ----
        legend_grounding_triggered = False
        legend_map: list = []
        legend_grounding_parse_error = False
        legend_grounding_traces: list = []
        compliance_retry_triggered = False

        ocr_legend = ocr_parsed.get("legend", []) if ocr_parsed else []
        ocr_chart_type = (ocr_parsed.get("chart_type") or "").strip().lower() if ocr_parsed else ""
        should_ground = (
            legend_grounder is not None and len(ocr_legend) > 1 and ocr_chart_type in _LEGEND_GROUNDING_CHART_TYPES
        )

        if should_ground:
            legend_grounding_triggered = True
            try:
                legend_grounder.lf_trace = lf_trace  # type: ignore[union-attr]
                lg_raw = legend_grounder._run(  # type: ignore[union-attr]
                    image_path=sample.image_path,
                    legend_list=ocr_legend,
                )
                legend_grounding_traces = legend_grounder.pop_traces()  # type: ignore[union-attr]
                lg_parsed, lg_ok = parse_strict(lg_raw, required_keys=["legend_map"])
                legend_grounding_parse_error = not lg_ok
                if lg_ok and isinstance(lg_parsed.get("legend_map"), list):
                    legend_map = lg_parsed["legend_map"]
            except Exception as exc:
                errors.append(f"legend_grounding_error: {exc}")
                legend_grounding_parse_error = True
                traceback.print_exc()

        # ---- Vision ----
        vision_prompt = ""
        vision_parsed: dict = {}
        vision_parse_error = True
        vision_raw = ""
        vision_traces: list = []
        vision_ms = 0.0

        _vision_legend_map = legend_map if legend_grounding_triggered and not legend_grounding_parse_error else None

        try:
            with timed() as vt:
                (
                    vision_prompt,
                    vision_parsed,
                    vision_parse_error,
                    vision_raw,
                    vision_traces,
                ) = vision_agent.run(
                    sample,
                    plan_parsed,
                    lf_trace=lf_trace,
                    ocr_result=ocr_parsed if ocr_parsed else None,
                    legend_map=_vision_legend_map,
                )
            vision_ms = vt.elapsed_ms
        except Exception as exc:
            errors.append(f"vision_error: {exc}")
            vision_parsed = {"answer": "ERROR", "explanation": str(exc)}
            vision_parse_error = True
            traceback.print_exc()

        # ---- Legend compliance check (only when legend grounding was active) ----
        if legend_grounding_triggered and not legend_grounding_parse_error and legend_map and not vision_parse_error:
            explanation = vision_parsed.get("explanation", "") or ""
            label_mentioned = any(
                entry.get("label", "").lower() in explanation.lower() for entry in legend_map if entry.get("label")
            )
            if not label_mentioned:
                compliance_retry_triggered = True
                retry_instruction = (
                    "IMPORTANT: Your previous response did not reference the pre-mapped "
                    "legend entries by name. You must begin your analysis by explicitly "
                    "stating which color you are reading for each series mentioned in "
                    "your answer, using the pre-mapped legend above."
                )
                try:
                    (
                        vision_prompt,
                        vision_parsed,
                        vision_parse_error,
                        vision_raw,
                        vision_traces,
                    ) = vision_agent.run(
                        sample,
                        plan_parsed,
                        lf_trace=lf_trace,
                        ocr_result=ocr_parsed if ocr_parsed else None,
                        legend_map=_vision_legend_map,
                        prepend_instruction=retry_instruction,
                    )
                except Exception as exc:
                    errors.append(f"vision_compliance_retry_error: {exc}")
                    traceback.print_exc()

        # ---- Forced-choice retry (UNANSWERABLE + MCQ) ----
        # If vision refused to answer but choices exist, force a selection before
        # handing off to the verifier — the verifier's override often fires too late.
        _vision_ua = vision_parsed.get("answer", "").strip().upper() == "UNANSWERABLE"
        _sample_choices: list = []
        _meta = getattr(sample, "metadata", {}) or {}
        _labeled = _meta.get("choices_labeled") or []
        if _labeled:
            _sample_choices = [f"{item.get('label')}: {item.get('text')}" for item in _labeled]
        elif getattr(sample, "choices", None):
            _sample_choices = list(sample.choices)

        if _vision_ua and _sample_choices and not vision_parse_error:
            _choices_text = "\n".join(f"  - {c}" for c in _sample_choices)
            _forced_instruction = (
                "FORCED CHOICE — your previous response returned UNANSWERABLE, which is not "
                "permitted when MCQ choices are provided.\n"
                "You MUST select one of the following options based on whatever visual evidence "
                "is available in the chart. Even if the chart is partially occluded or labels "
                "are small, pick the most visually plausible option and note your uncertainty "
                "in the explanation.\n\n"
                f"Available choices:\n{_choices_text}\n\n"
                "Do NOT output UNANSWERABLE. Output the letter or exact text of the best choice."
            )
            try:
                (
                    vision_prompt,
                    vision_parsed,
                    vision_parse_error,
                    vision_raw,
                    vision_traces,
                ) = vision_agent.run(
                    sample,
                    plan_parsed,
                    lf_trace=lf_trace,
                    ocr_result=ocr_parsed if ocr_parsed else None,
                    legend_map=_vision_legend_map,
                    prepend_instruction=_forced_instruction,
                )
            except Exception as exc:
                errors.append(f"vision_forced_choice_retry_error: {exc}")
                traceback.print_exc()

        # ---- Verifier (Pass 2.5) ----
        (
            verifier_prompt,
            verifier_parsed,
            verifier_parse_error,
            verifier_raw,
            verifier_ms,
            verifier_verdict,
            verifier_traces,
            verifier_errors,
        ) = _run_verifier_stage(verifier_agent, sample, plan_parsed, vision_parsed, lf_trace)
        errors.extend(verifier_errors)

        run_end = iso_now()

        # ---- Image ref ----
        image_sha = ""
        if sample.image_path and Path(sample.image_path).exists():
            with contextlib.suppress(Exception):
                image_sha = sha256_file(sample.image_path)

        # ---- Assemble MEP ----
        mep = MEP(
            run_id=run_id,
            config=MEPConfig(
                planner_backend=config["planner_backend"],
                vision_backend=config["vision_backend"],
                judge_backend=config.get("judge_backend", config["planner_backend"]),
                config_name=config_name,
                planner_model=config["planner_model"],
                vision_model=config["vision_model"],
            ),
            sample=MEPSample(
                dataset=dataset_name,
                sample_id=sample.sample_id,
                question=sample.question,
                question_type=sample.question_type.value,
                expected_output=sample.expected_output,
                image_ref=ImageRef(path=sample.image_path, sha256=image_sha),
                metadata=sample.metadata,
            ),
            plan=MEPPlan(
                prompt=plan_prompt,
                raw_text=plan_raw,
                parsed=plan_parsed,
                parse_error=plan_parse_error,
            ),
            ocr=MEPOcr(
                raw_text=ocr_raw,
                parsed=ocr_parsed,
                parse_error=ocr_parse_error,
                tool_trace=ocr_traces,
            )
            if ocr_tool is not None
            else None,
            legend_grounding=MEPLegendGrounding(
                triggered=legend_grounding_triggered,
                legend_map=legend_map,
                parse_error=legend_grounding_parse_error,
                compliance_retry_triggered=compliance_retry_triggered,
                tool_trace=legend_grounding_traces,
            )
            if legend_grounding_triggered
            else None,
            vision=MEPVision(
                prompt=vision_prompt,
                raw_text=vision_raw,
                parsed=vision_parsed,
                parse_error=vision_parse_error,
                tool_trace=vision_traces,
            ),
            verifier=MEPVerifier(
                prompt=verifier_prompt,
                raw_text=verifier_raw,
                parsed=verifier_parsed,
                parse_error=verifier_parse_error,
                verdict=verifier_verdict,
                tool_trace=verifier_traces,
            )
            if verifier_agent is not None
            else None,
            timestamps=MEPTimestamps(
                start=run_start,
                end=run_end,
                planner_ms=plan_ms,
                ocr_ms=ocr_ms,
                vision_ms=vision_ms,
                verifier_ms=verifier_ms,
            ),
            errors=errors,
            lf_trace_id=lf_trace_id,
        )

        # ---- Immediately log available scores to Langfuse ----
        scores: dict = {
            "planner_parse_ok": float(not plan_parse_error),
            "vision_parse_ok": float(not vision_parse_error),
            "has_errors": float(bool(errors)),
        }
        if legend_grounding_triggered:
            scores["legend_compliance"] = float(not compliance_retry_triggered)
        log_trace_scores(lf_trace, scores)
        if lf_trace:
            lf_trace.update(output=vision_parsed if vision_parsed else None)

    return write_mep(mep, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915
    """
    Parse CLI arguments and run the MEP generation pipeline.

    Configures agents, loads the dataset, and manages parallel execution
    of the evaluation pipeline.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Generate MEPs for supported financial VQA datasets")
    parser.add_argument(
        "--dataset",
        default="chartqapro",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset name",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Max samples to process (0 or negative = entire split after slice)",
    )
    parser.add_argument(
        "--config",
        default="gemini_gemini",
        choices=sorted(BACKEND_CONFIGS.keys()),
        help="Backend config preset",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = sequential)")
    parser.add_argument("--out", default="meps/", help="Output directory for MEPs")
    parser.add_argument("--image_dir", default=None, help="Directory to save/load chart images (defaults per dataset)")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace datasets cache dir")
    parser.add_argument(
        "--planner_model",
        default=None,
        help="Override planner model name (e.g. gpt-4o, o3)",
    )
    parser.add_argument(
        "--vision_model",
        default=None,
        help="Override vision model name (e.g. gpt-4o, gemini-1.5-pro)",
    )
    parser.add_argument(
        "--verifier_model",
        default=None,
        help="Override verifier model (defaults to vision_model)",
    )
    parser.add_argument("--no_verifier", action="store_true", help="Skip Pass 2.5 verifier agent")
    parser.add_argument("--no_ocr", action="store_true", help="Skip OCR pre-read step")
    parser.add_argument("--no_legend_grounding", action="store_true", help="Skip legend grounding stage")
    parser.add_argument("--run_tag", default=None, help="Subfolder tag within dataset dir (e.g. planner_v2)")
    parser.add_argument("--no_langfuse", action="store_true", help="Disable Langfuse dataset/prompt registration")
    parser.add_argument(
        "--resume", action="store_true", help="Skip samples that already have MEP JSONs in the output dir"
    )
    parser.add_argument(
        "--ocr_model",
        default=None,
        help="Override OCR model (defaults to vision_model)",
    )
    args = parser.parse_args()

    config = dict(BACKEND_CONFIGS[args.config])  # copy so we don't mutate the preset
    if args.planner_model:
        config["planner_model"] = args.planner_model
    if args.vision_model:
        config["vision_model"] = args.vision_model
    run_id = str(uuid.uuid4())
    dataset_key = args.dataset.lower()
    ds_cfg = DATASET_CONFIGS[dataset_key]
    dataset_name: str = ds_cfg["display_name"]
    dataset_slug = dataset_key

    image_dir = args.image_dir or ds_cfg["default_image_dir"]

    out_dir = str(
        Path(args.out)
        / f"{config['planner_backend']}_{config['vision_backend']}"
        / mep_dataset_split_relpath(
            dataset_slug, args.split, no_verifier=args.no_verifier, no_ocr=args.no_ocr, run_tag=args.run_tag
        )
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    n_limit = None if args.n <= 0 else args.n
    n_disp = "all" if n_limit is None else str(n_limit)
    print(f"Loading dataset  : {dataset_name} ({dataset_slug}) split={args.split} n={n_disp}")
    samples: list[PerceivedSample] = ds_cfg["loader"](
        split=args.split,
        n=n_limit,
        image_dir=image_dir,
        cache_dir=args.cache_dir,
    )
    if args.resume:
        existing_ids: set[str] = set()
        for p in Path(out_dir).glob("*.json"):
            try:
                data = json.loads(p.read_text())
                sid = data.get("sample", {}).get("sample_id") or p.stem
            except Exception:
                sid = p.stem
            existing_ids.add(sid)
        before = len(samples)
        samples = [s for s in samples if s.sample_id not in existing_ids]
        skipped = before - len(samples)
        print(f"Resume enabled   : skipped {skipped} existing samples")
        if not samples:
            print("All requested samples already processed; exiting early.")
            return
    print(f"Samples loaded   : {len(samples)}")
    print(f"Config           : {args.config}  run_id={run_id}")
    print(f"Output dir       : {out_dir}")
    print(f"Workers          : {args.workers}")

    # Langfuse: register dataset + version prompts at run start (no-ops if unavailable)
    lf_client = None
    if args.no_langfuse:
        print("Langfuse         : disabled (--no_langfuse)")
    else:
        lf_client = get_client()
        if lf_client:
            print("Langfuse         : enabled")
            register_dataset(samples, dataset_name=dataset_name, split=args.split)
            push_prompts()
        else:
            print("Langfuse         : not configured (set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable)")

    # Build agents once — run() creates fresh Crew/Tool per call so this is thread-safe
    print("Initialising agents …")
    planner = PlannerAgent(backend=config["planner_backend"], model=config["planner_model"])
    vision_agent = VisionAgent(
        agent_backend=config["planner_backend"],
        agent_model=config["planner_model"],
        vision_backend=config["vision_backend"],
        vision_model=config["vision_model"],
    )
    verifier: Optional[VerifierAgent] = None
    if not args.no_verifier:
        verifier_model = args.verifier_model or config["vision_model"]
        verifier = VerifierAgent(backend=config["vision_backend"], model=verifier_model)
        print(f"Verifier         : enabled ({config['vision_backend']} / {verifier_model})")
    else:
        print("Verifier         : disabled (--no_verifier)")

    ocr: Optional[OcrReaderTool] = None
    if not args.no_ocr:
        ocr_model = args.ocr_model or config["vision_model"]
        ocr = OcrReaderTool(backend=config["vision_backend"], model=ocr_model)
        print(f"OCR pre-reader   : enabled ({config['vision_backend']} / {ocr_model})")
    else:
        print("OCR pre-reader   : disabled (--no_ocr)")

    legend_grounder: Optional[LegendGrounderTool] = None
    if not args.no_ocr and not args.no_legend_grounding:
        lg_model = args.ocr_model or config["vision_model"]
        legend_grounder = LegendGrounderTool(backend=config["vision_backend"], model=lg_model)
        print(f"Legend grounder  : enabled ({config['vision_backend']} / {lg_model})")
    else:
        reason = "--no_ocr" if args.no_ocr else "--no_legend_grounding"
        print(f"Legend grounder  : disabled ({reason})")
    print()

    if args.workers <= 1:
        for i, sample in enumerate(samples, 1):
            print(f"[{i}/{len(samples)}] {sample.sample_id} …", end=" ", flush=True)
            try:
                path = process_sample(
                    sample,
                    planner,
                    vision_agent,
                    config,
                    run_id,
                    out_dir,
                    dataset_name,
                    lf_client,
                    verifier,
                    ocr,
                    legend_grounder,
                )
                print(f"OK → {path}")
            except Exception as exc:
                print(f"ERROR: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_sample = {
                pool.submit(
                    process_sample,
                    s,
                    planner,
                    vision_agent,
                    config,
                    run_id,
                    out_dir,
                    dataset_name,
                    lf_client,
                    verifier,
                    ocr,
                    legend_grounder,
                ): s
                for s in samples
            }
            for done, future in enumerate(as_completed(future_to_sample), 1):
                s = future_to_sample[future]
                try:
                    path = future.result()
                    print(f"[{done}/{len(samples)}] {s.sample_id} → {path}")
                except Exception as exc:
                    print(f"[{done}/{len(samples)}] {s.sample_id} ERROR: {exc}")

    print(f"\nDone. MEPs written to: {out_dir}")


if __name__ == "__main__":
    main()
