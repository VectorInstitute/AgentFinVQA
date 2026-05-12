"""VerifierAgent — Pass 2.5: critically reviews the VisionAgent's draft answer.

The verifier sees the chart image AND the draft answer/explanation and decides
whether to CONFIRM or REVISE the answer.

Like VisionAgent, this agent uses CrewAI + a dedicated tool (VerifierTool) so
that the VLM call is observable, traceable, and consistent with the rest of the
pipeline architecture.
"""

import os
from typing import Any, Optional, Tuple

from crewai import LLM, Agent, Crew, Task

from ..agents.vision_agent import _is_multi_select
from ..langfuse_integration.tracing import close_span, open_llm_span
from ..tools.verifier_tool import VerifierTool, format_source_sentences_block
from ..utils.json_strict import parse_strict
from ..utils.model_compat import openai_temperature


VERIFIER_REQUIRED_KEYS = ["verdict", "answer", "reasoning"]


def _build_llm(backend: str, model: str, api_key: Optional[str]) -> LLM:
    if backend == "openai":
        return LLM(
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            **openai_temperature(model),
        )
    if backend == "gemini":
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            temperature=0,
        )
    raise ValueError(f"Unknown verifier backend: {backend!r}")


class VerifierAgent:
    """
    A validation agent that critiques draft answers against visual evidence.

    Uses CrewAI + VerifierTool to make a single auditing VLM call, consistent
    with how VisionAgent uses VisionQATool.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.model = model
        self.api_key = api_key

    def _build_tool(self, lf_trace: Any = None) -> VerifierTool:
        key = self.api_key or (
            os.environ.get("OPENAI_API_KEY", "") if self.backend == "openai" else os.environ.get("GEMINI_API_KEY", "")
        )
        return VerifierTool(
            backend=self.backend,
            model=self.model,
            api_key=key,
            lf_trace=lf_trace,
        )

    def run(
        self,
        sample,  # PerceivedSample
        plan: dict,
        vision_parsed: dict,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str, list]:
        """
        Critically audit a draft answer using a CrewAI agent + VerifierTool.

        Returns
        -------
        task_description : str
        parsed : dict  — keys: verdict, answer, reasoning
        parse_error : bool
        raw_text : str
        tool_traces : list of dict
        """
        plan_steps = plan.get("steps", [])
        draft_answer = vision_parsed.get("answer", "(none)")
        draft_explanation = vision_parsed.get("explanation", "(none)")
        question_type = getattr(
            getattr(sample, "question_type", None),
            "value",
            str(getattr(sample, "question_type", "standard")),
        )
        image_path = getattr(sample, "image_path", "") or ""

        # Extract MCQ choices: prefer labeled choices, else plain list.
        choices: list = []
        meta = getattr(sample, "metadata", {}) or {}
        labeled = meta.get("choices_labeled") or []
        if labeled:
            choices = [f"{item.get('label')}: {item.get('text')}" for item in labeled]
        elif getattr(sample, "choices", None):
            choices = list(sample.choices)

        caption = (meta.get("verified_caption") or "").strip()
        # When the sample's metadata carries analyst-written source sentences in
        # ``related_sentences``, pass them through to the verifier prompt so it can
        # cross-check numeric / categorical claims in the draft answer against the
        # original text grounding. Field is optional — datasets without this signal
        # leave the block empty.
        related_sentences = meta.get("related_sentences")
        is_ms = _is_multi_select(sample)

        # Compute min choice_analysis confidence from vision output
        ca = vision_parsed.get("choice_analysis") or {}
        ca_confs = [v.get("confidence", 1.0) for v in ca.values() if isinstance(v, dict)]
        vision_min_conf = min(ca_confs) if ca_confs else 1.0

        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"
        choices_text = ("\nMCQ Choices:\n" + "\n".join(f"  - {c}" for c in choices) + "\n") if choices else ""
        caption_text = (
            "\nChart context (analyst caption — cross-check the draft answer against this text; "
            "REVISE if a value or category here clearly contradicts the draft, but do NOT revise "
            f"on phrasing differences alone):\n  {caption}\n"
            if caption
            else ""
        )
        source_sentences_text = format_source_sentences_block(related_sentences, leading_newline=True)
        # Reluctance hint when vision was highly confident (single expression to keep run() compact).
        reluctance_note = (
            "\n⚠ HIGH-CONFIDENCE VISION: the vision agent reported high confidence (≥0.95) on all choices. "
            "Only REVISE if you can identify a specific, clear factual error with direct visual evidence. "
            "If merely uncertain, set confidence < 0.75 and CONFIRM.\n"
            if vision_min_conf >= 0.95
            else "\n⚠ The vision agent was reasonably confident. REVISE only for clear, specific errors.\n"
            if vision_min_conf >= 0.85
            else ""
        )

        multi_select_note = (
            "\n⚠ MULTI-SELECT: verify EACH choice independently. "
            "Include ALL correct letters; never reduce to a single letter unless only one is correct.\n"
            if is_ms
            else ""
        )
        task_description = (
            f"You are auditing a vision agent's answer to a chart question.\n\n"
            f"Question         : {sample.question}\n"
            f"Question Type    : {question_type}\n"
            f"Image path       : {image_path}\n"
            f"{choices_text}"
            f"{caption_text}"
            f"{source_sentences_text}"
            f"{reluctance_note}"
            f"{multi_select_note}\n"
            f"Inspection plan the agent followed:\n{steps_text}\n\n"
            f"Vision agent's draft answer     : {draft_answer}\n"
            f"Vision agent's draft explanation: {draft_explanation}\n\n"
            f"Call verifier_tool ONCE with all these details and return its JSON output exactly."
        )

        tool = self._build_tool(lf_trace=lf_trace)
        llm = _build_llm(self.backend, self.model, self.api_key)

        verifier_span = open_llm_span(
            lf_trace,
            name="verifier_agent",
            input_data={"task_description": task_description, "draft_answer": draft_answer},
            model=self.model,
            metadata={"backend": self.backend},
        )

        agent = Agent(
            role="Chart QA Verifier",
            goal=(
                "Audit the vision agent's draft answer by calling verifier_tool exactly once "
                "and returning the JSON result with 'verdict', 'answer', and 'reasoning'."
            ),
            backstory=(
                "You are a precise chart QA auditor. You use verifier_tool to inspect the chart "
                "image alongside the draft answer, identify any errors, and decide to CONFIRM or REVISE."
            ),
            llm=llm,
            tools=[tool],
            verbose=False,
            allow_delegation=False,
            max_iter=2,
        )

        task = Task(
            description=task_description,
            expected_output=(
                'JSON object: {"verdict": "confirmed" | "revised", "answer": "...", "reasoning": "..."}\n'
                "When calling verifier_tool, pass:\n"
                "  - `choices`: MCQ choices listed above (empty list if none)\n"
                "  - `caption`: analyst caption if shown above (empty string if none)\n"
                "  - `related_sentences`: list of source sentences from the 'Source sentences' block above "
                "(empty list if no such block was shown)\n"
                + (
                    "  - `multi_select`: True — this is a multi-select question; answer must contain ALL correct letters.\n"
                    if is_ms
                    else "  - `multi_select`: False\n"
                )
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, parse_ok = parse_strict(raw_text, required_keys=VERIFIER_REQUIRED_KEYS)

        tool_traces = tool.pop_traces()

        if not parsed:
            parsed = {
                "verdict": "confirmed",
                "answer": draft_answer,
                "reasoning": f"Parse error — defaulting to confirm. Raw: {raw_text[:120]}",
            }
            parse_ok = False

        if parsed.get("verdict", "").lower() not in ("confirmed", "revised"):
            parsed["verdict"] = "confirmed"

        # Confidence in [0, 1]. Default 0.75 if missing (passes gate). Gate revisions
        # only when the model reports low confidence (< 0.75).
        try:
            parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.75))))
        except (TypeError, ValueError):
            parsed["confidence"] = 0.75

        close_span(verifier_span, output=parsed)
        return task_description, parsed, not parse_ok, raw_text, tool_traces
