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


_VERIFIER_PROMPT = """\
You are a critical chart QA verifier. A vision agent has already attempted to answer
the question below. Your job: look at the chart image carefully and audit the work.

Question         : {question}
Question Type    : {question_type}

Inspection plan the agent was supposed to follow:
{plan_steps}

Vision Agent's Draft Answer     : {draft_answer}
Vision Agent's Draft Explanation: {draft_explanation}

Examine the chart image. Then decide:
  CONFIRM — the draft answer is correct (output the same answer unchanged)
  REVISE  — you can see a clear, specific error; output the corrected answer

Rules:
- Only REVISE when you are confident you can point to a concrete error in the chart
- If uncertain, CONFIRM — do not second-guess without visual evidence
- For MCQ questions: the answer must be one of the stated choices
- For MCQ questions: return exactly one of the provided choice texts or the precise letter/letter-combo (e.g., "AC")
- If the answer is truly unanswerable from the chart, say exactly "UNANSWERABLE"
- Keep answers concise — numbers, short phrases, or single words where appropriate
- Always cite at least one numeric value or labeled element from the chart in your reasoning; if you cannot cite evidence, REVISE
- For MCQ questions, compare the draft answer against at least one alternative choice before confirming
- Limit the final answer to a short phrase (≤10 words) and keep the reasoning ≤2 sentences.

Output ONLY JSON, no markdown, no extra text:
{{"verdict": "confirmed" | "revised", "answer": "<final answer>", "reasoning": "<one sentence grounded in what you see in the chart>"}}"""


class VerifierInput(BaseModel):
    """Input schema for VerifierTool."""

    image_path: str = Field(description="Absolute or relative path to the chart image file")
    question: str = Field(description="The question being answered")
    question_type: str = Field(description="The type of question (e.g. MCQ, standard)")
    plan_steps: List[str] = Field(description="Ordered inspection steps from the planner")
    draft_answer: str = Field(description="The draft answer produced by the vision agent")
    draft_explanation: str = Field(description="The draft explanation produced by the vision agent")


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
        prompt = _VERIFIER_PROMPT.format(
            question=question,
            question_type=question_type,
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
            max_completion_tokens=256,
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
            config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=768),
        )
        raw_text = response.text or ""
        finish = str(response.candidates[0].finish_reason) if response.candidates else "unknown"
        provider_meta = {"model": self.model, "finish_reason": finish}
        return raw_text, provider_meta
