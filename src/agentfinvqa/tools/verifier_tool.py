"""verifier_tool — CrewAI tool wrapping the verifier VLM call.

Accepts the draft answer/explanation alongside the chart image and question,
runs the verifier prompt against a VLM backend, and returns a JSON string
with 'verdict', 'answer', and 'reasoning' fields.
"""

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr

from ..langfuse_integration.tracing import close_span, open_llm_span
from ..utils.model_compat import openai_temperature


def format_source_sentences_block(
    related_sentences: Optional[Any],
    *,
    leading_newline: bool = False,
) -> str:
    """Render the analyst source-sentences block for a verifier prompt.

    Accepts ``None``, a string, or a list of strings. Empty / blank-only inputs
    collapse to ``""`` so the surrounding format string drops the block cleanly.
    The block text is dataset-agnostic — any sample that carries
    ``related_sentences`` in metadata gets the same treatment.
    """
    if not related_sentences:
        return ""
    if isinstance(related_sentences, str):
        rs_items = [related_sentences.strip()] if related_sentences.strip() else []
    else:
        rs_items = [str(s).strip() for s in related_sentences if str(s).strip()]
    if not rs_items:
        return ""
    rs_lines = "\n".join(f"  - {s}" for s in rs_items)
    body = (
        "Source sentences from the chart's accompanying analyst note "
        "(may contain the specific values being asked about — cross-check "
        "numeric and categorical facts against this text before confirming):\n"
        f"{rs_lines}\n"
    )
    return f"\n{body}" if leading_newline else body


_VERIFIER_PROMPT_SINGLE = """\
You are a critical chart QA verifier. A vision agent has already attempted to answer
the question below. Your job: look at the chart image carefully and audit the work.

Question         : {question}
Question Type    : {question_type}
{choices_block}{caption_block}{source_sentences_block}
Inspection plan the agent was supposed to follow:
{plan_steps}

Vision Agent's Draft Answer     : {draft_answer}
Vision Agent's Draft Explanation: {draft_explanation}

Examine the chart image. Then decide:
  CONFIRM — the draft answer is correct (output the same answer unchanged)
  REVISE  — you can see a clear, specific error; output the corrected answer

Rules:
- Only REVISE when you are confident you can point to a concrete error in the chart OR a direct contradiction with the caption / source sentences above
- If uncertain, CONFIRM — do not second-guess without visual or textual evidence
- Stylistic / phrasing differences are NOT a reason to revise — only revise when a value or category is wrong
- For MCQ questions: the answer must be one of the stated choices above
- For MCQ questions: return exactly one of the provided choice texts or the precise letter/letter-combo (e.g., "AC")
- UNANSWERABLE override: if the draft answer is "UNANSWERABLE" and MCQ choices are listed above,
  you MUST systematically check EACH choice against the chart AND the source sentences before confirming:
    1. For each choice: is there visual evidence (bar height, line position, label, tick mark) OR caption / source-sentence text that
       supports OR rules it out?
    2. If ANY choice can be supported or ruled out, REVISE to the best-supported choice.
    3. Only confirm UNANSWERABLE when ALL choices are genuinely undeterminable from chart and text.
- Keep answers concise — numbers, short phrases, or single words where appropriate
- Always cite at least one numeric value or labeled element from the chart, caption, or source sentences in your reasoning
- For MCQ questions, compare the draft answer against at least one alternative choice before confirming
- Limit the final answer to a short phrase (≤10 words) and keep the reasoning ≤2 sentences.
- Add a `confidence` field (0.0–1.0): your confidence that your verdict and answer are correct based on what you can see in the chart and any caption / source-sentence text.
  Use ≥0.8 only when you can cite specific visual evidence (tick value, bar height, label) or a direct textual match. Use <0.6 when both chart and text are ambiguous.

Output ONLY JSON, no markdown, no extra text:
{{"verdict": "confirmed" | "revised", "answer": "<final answer>", "reasoning": "<one sentence grounded in what you see in the chart or read in the caption/source sentences>", "confidence": 0.0}}"""

_VERIFIER_PROMPT_MULTI = """\
You are a critical chart QA verifier. This is a MULTI-SELECT question — multiple choices may be correct.

Question         : {question}
Question Type    : {question_type}
{choices_block}{caption_block}{source_sentences_block}
Inspection plan the agent was supposed to follow:
{plan_steps}

Vision Agent's Draft Answer     : {draft_answer}
Vision Agent's Draft Explanation: {draft_explanation}

YOUR TASK — evaluate the draft answer for a multi-select question:
  CONFIRM — all selected letters are correct AND no correct letters are missing
  REVISE  — some letters are wrong, or correct letters were omitted

Step-by-step verification (do this for EVERY choice listed above):
  1. For each letter in the draft answer: is there clear visual evidence in the chart, OR a supporting statement in the caption / source sentences? If NOT supported by either → it must be removed.
  2. For each letter NOT in the draft answer: is there clear visual evidence in the chart, OR a supporting statement in the caption / source sentences that it should be included? If YES → it must be added.
  3. Your final answer = all and only the letters with chart or text evidence.

Rules:
- NEVER collapse a multi-select answer to a single letter unless truly only one is correct
- NEVER add letters you cannot visually confirm or find textual support for in the caption / source sentences
- Stylistic / phrasing differences in the caption are NOT a reason to add or remove a letter — only true factual support matters
- If the draft already selected the right set, CONFIRM — do not change it
- Return letters concatenated with no separator (e.g. "ABC")
- Reasoning must cite specific chart evidence (values, labels, bar heights) and/or caption-or-source-sentence quotes for each included/excluded choice
- Keep reasoning ≤3 sentences.
- Add a `confidence` field (0.0–1.0): your confidence that your final letter set is correct.
  Use ≥0.8 only when you have clear visual evidence or direct textual support for each included/excluded choice. Use <0.6 when both chart and text are ambiguous.

Output ONLY JSON, no markdown, no extra text:
{{"verdict": "confirmed" | "revised", "answer": "<all correct letters concatenated, e.g. ABC>", "reasoning": "<evidence-grounded sentence(s)>", "confidence": 0.0}}"""


class VerifierInput(BaseModel):
    """Input schema for VerifierTool."""

    image_path: str = Field(description="Absolute or relative path to the chart image file")
    question: str = Field(description="The question being answered")
    question_type: str = Field(description="The type of question (e.g. MCQ, standard)")
    plan_steps: List[str] = Field(description="Ordered inspection steps from the planner")
    draft_answer: str = Field(description="The draft answer produced by the vision agent")
    draft_explanation: str = Field(description="The draft explanation produced by the vision agent")
    choices: Optional[List[str]] = Field(default=None, description="MCQ answer choices if applicable")
    caption: Optional[str] = Field(default=None, description="Analyst-written chart caption for context")
    related_sentences: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional analyst-written source sentences attached to the sample. "
            "May contain the specific numeric or categorical values being asked about — "
            "cross-check the draft answer's facts against this text before confirming."
        ),
    )
    multi_select: bool = Field(default=False, description="True when multiple choices can be correct simultaneously")


class VerifierTool(BaseTool):
    """Calls a VLM backend to critically audit a chart QA draft answer."""

    name: str = "verifier_tool"
    description: str = (
        "Critically audit a vision agent's draft answer against the chart image. "
        "Provide the image path, question, question type, plan steps, draft answer, "
        "and draft explanation. Returns a JSON string with 'verdict', 'answer', and 'reasoning'."
    )
    args_schema: Type[BaseModel] = VerifierInput

    backend: str = "gemini"
    model: str = "gemini-2.5-flash-lite"
    api_key: str = ""
    lf_trace: Optional[Any] = None

    _traces: list = PrivateAttr(default_factory=list)

    def pop_traces(self) -> list:
        """Return buffered tool traces and clear the buffer."""
        traces = list(self._traces)
        self._traces.clear()
        return traces

    def _run(
        self,
        image_path: str,
        question: str,
        question_type: str,
        plan_steps: List[str],
        draft_answer: str,
        draft_explanation: str,
        choices: Optional[List[str]] = None,
        caption: Optional[str] = None,
        related_sentences: Optional[List[str]] = None,
        multi_select: bool = False,
    ) -> str:
        start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        lf_span = open_llm_span(
            self.lf_trace,
            name="verifier_tool",
            input_data={
                "image_path": image_path,
                "question": question,
                "draft_answer": draft_answer,
            },
            model=self.model,
            metadata={"backend": self.backend},
        )

        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"
        choices_block = "MCQ Choices:\n" + "\n".join(f"  - {c}" for c in choices) + "\n" if choices else ""
        if caption and caption.strip():
            caption_block = (
                "Chart context (analyst caption — cross-check the draft answer against this text; "
                "REVISE if a value or category here clearly contradicts the draft, but do NOT revise "
                f"on phrasing differences alone):\n  {caption.strip()}\n"
            )
        else:
            caption_block = ""
        source_sentences_block = format_source_sentences_block(related_sentences)
        template = _VERIFIER_PROMPT_MULTI if multi_select else _VERIFIER_PROMPT_SINGLE
        prompt = template.format(
            question=question,
            question_type=question_type,
            choices_block=choices_block,
            caption_block=caption_block,
            source_sentences_block=source_sentences_block,
            plan_steps=steps_text,
            draft_answer=draft_answer,
            draft_explanation=draft_explanation,
        )

        provider_meta: dict = {}
        error_str: Optional[str] = None
        has_image = image_path and Path(image_path).exists()

        try:
            if not has_image:
                raw_text = json.dumps(
                    {
                        "verdict": "confirmed",
                        "answer": draft_answer,
                        "reasoning": "Image unavailable; cannot verify visually.",
                    }
                )
                provider_meta = {"skipped": "no image"}
            elif self.backend == "openai":
                raw_text, provider_meta = self._call_openai(prompt, image_path)
            elif self.backend == "gemini":
                raw_text, provider_meta = self._call_gemini(prompt, image_path)
            else:
                raise ValueError(f"Unknown backend: {self.backend!r}")
        except Exception as exc:
            raw_text = json.dumps(
                {
                    "verdict": "confirmed",
                    "answer": draft_answer,
                    "reasoning": f"Tool error: {exc}",
                }
            )
            provider_meta = {"error": str(exc)}
            error_str = str(exc)

        end_ts = datetime.now(timezone.utc).isoformat()
        elapsed_ms = (time.time() - t0) * 1000.0

        model_used = provider_meta.pop("model", self.model)
        usage = provider_meta.get("usage", {})

        close_span(lf_span, output={"raw_text": raw_text}, usage=usage if usage else None, error=error_str)

        self._traces.append(
            {
                "tool": "verifier_tool",
                "backend": self.backend,
                "model": model_used,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "elapsed_ms": elapsed_ms,
                "provider_metadata": provider_meta,
            }
        )

        return raw_text

    def _encode_image(self, image_path: str) -> tuple:
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, mime

    def _call_openai(self, prompt: str, image_path: str) -> tuple:
        client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""))
        b64, mime = self._encode_image(image_path)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                    ],
                }
            ],
            max_completion_tokens=2048,
            **openai_temperature(self.model),
        )
        raw_text = response.choices[0].message.content or ""
        provider_meta = {
            "model": response.model,
            "request_id": response.id,
            "usage": response.usage.model_dump() if response.usage else {},
        }
        return raw_text, provider_meta

    def _call_gemini(self, prompt: str, image_path: str) -> tuple:
        client = genai.Client(api_key=self.api_key or os.environ.get("GEMINI_API_KEY", ""))
        b64, mime = self._encode_image(image_path)
        response = client.models.generate_content(
            model=self.model,
            contents=[genai.types.Part.from_bytes(data=base64.b64decode(b64), mime_type=f"image/{mime}"), prompt],
            config=genai.types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2048,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=512),
            ),
        )
        raw_text = response.text or ""
        finish = str(response.candidates[0].finish_reason) if response.candidates else "unknown"
        provider_meta = {"model": self.model, "finish_reason": finish}
        return raw_text, provider_meta
