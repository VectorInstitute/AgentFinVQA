"""legend_grounder_tool — VLM call that maps legend entries to visual properties.

Runs a single structured extraction call to bind each legend label to its
color, line style, and confidence score. The output is injected into the
vision prompt as locked ground truth, preventing the vision agent from
visually re-identifying series (which is the root cause of legend confusion).
"""

import base64
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, List, Optional, Type


try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool  # type: ignore[assignment]
from google import genai
from openai import OpenAI  # noqa: F401 — re-exported for backwards-compat / type hints
from pydantic import BaseModel, Field, PrivateAttr

from ..langfuse_integration.tracing import close_span, open_llm_span
from ..utils.model_compat import openai_temperature
from ..utils.openai_compat import build_openai_client, qwen35_extra_body


_LEGEND_GROUNDER_PROMPT_TEMPLATE = """\
You are analyzing a chart image. The following legend entries have been \
extracted from this chart:

{legend_list}

For each legend entry, identify its visual representation in the chart.
Output ONLY a JSON object with this exact structure — no markdown, \
no explanation:

{{
  "legend_map": [
    {{
      "label": "<exact legend text>",
      "color_description": "<e.g. dark blue, orange, dashed grey>",
      "rgb_approximate": "<e.g. (70, 130, 180) — your best estimate>",
      "line_style": "<solid|dashed|dotted|bar|area|point>",
      "confidence": <0.0 to 1.0>
    }}
  ]
}}

Rules:
- Include every legend entry, even if confidence is low
- color_description must be a human-readable color name
- If two entries look visually similar, note that in color_description
- Do not invent entries not in the legend list
- Output JSON only
"""


class LegendGrounderInput(BaseModel):
    """Input schema for LegendGrounderTool."""

    image_path: str = Field(description="Absolute or relative path to the chart image file")
    legend_list: List[str] = Field(description="Legend entry labels extracted by OCR")


class LegendGrounderTool(BaseTool):
    """Map OCR legend labels to visual properties (color, line style)."""

    name: str = "legend_grounder_tool"
    description: str = (
        "Given a chart image and a list of legend entry labels, identify the color, "
        "line style, and confidence score for each entry. Returns structured JSON "
        "to ground the vision agent's series identification."
    )
    args_schema: Type[BaseModel] = LegendGrounderInput

    backend: str = "gemini"
    model: str = "gemini-2.5-flash-lite"
    api_key: str = ""
    # Optional OpenAI-compatible endpoint (e.g. local vLLM serving Qwen2.5-VL).
    # When set together with backend="openai", the OpenAI client is pointed here
    # instead of api.openai.com. Falls back to OPENAI_BASE_URL env var.
    api_base: str = ""
    lf_trace: Optional[Any] = None

    _traces: list = PrivateAttr(default_factory=list)

    def pop_traces(self) -> list:
        """Flush and return the tool's execution traces."""
        traces = list(self._traces)
        self._traces.clear()
        return traces

    # ------------------------------------------------------------------
    # CrewAI entry point
    # ------------------------------------------------------------------

    def _run(self, image_path: str, legend_list: List[str]) -> str:
        """
        Make a single VLM call to map each legend entry to its visual properties.

        Parameters
        ----------
        image_path : str
            Path to the chart image.
        legend_list : list of str
            Legend entry labels from OCR.

        Returns
        -------
        str
            JSON string with a ``legend_map`` list.
        """
        start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        legend_formatted = "\n".join(f"  - {entry}" for entry in legend_list)
        prompt = _LEGEND_GROUNDER_PROMPT_TEMPLATE.format(legend_list=legend_formatted)

        lf_span = open_llm_span(
            self.lf_trace,
            name="legend_grounder_tool",
            input_data={"image_path": image_path, "legend_list": legend_list},
            model=self.model,
            metadata={"backend": self.backend},
        )

        provider_meta: dict = {}
        error_str: Optional[str] = None
        try:
            if self.backend == "openai":
                raw_text, provider_meta = self._call_openai(image_path, prompt)
            elif self.backend == "gemini":
                raw_text, provider_meta = self._call_gemini(image_path, prompt)
            else:
                raise ValueError(f"Unknown backend: {self.backend!r}")
        except Exception as exc:
            raw_text = json.dumps({"legend_map": [], "error": str(exc)})
            provider_meta = {"error": str(exc)}
            error_str = str(exc)

        end_ts = datetime.now(timezone.utc).isoformat()
        elapsed_ms = (time.time() - t0) * 1000.0

        model_used = provider_meta.pop("model", self.model)
        usage = provider_meta.get("usage", {})

        close_span(
            lf_span,
            output={"raw_text": raw_text},
            usage=usage if usage else None,
            error=error_str,
        )

        self._traces.append(
            {
                "tool": "legend_grounder_tool",
                "backend": self.backend,
                "model": model_used,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "elapsed_ms": elapsed_ms,
                "provider_metadata": provider_meta,
            }
        )

        return raw_text

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _encode_image(self, image_path: str) -> tuple:
        with open(image_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        mime = "png" if image_path.lower().endswith(".png") else "jpeg"
        return b64, mime

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def _call_openai(self, image_path: str, prompt: str) -> tuple:
        client = build_openai_client(self.api_key, self.api_base)
        b64, mime = self._encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=512,
            extra_body=qwen35_extra_body(self.model),
            **openai_temperature(self.model),
        )

        raw_text = response.choices[0].message.content or ""
        provider_meta = {
            "model": response.model,
            "request_id": response.id,
            "usage": response.usage.model_dump() if response.usage else {},
        }
        return raw_text, provider_meta

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    def _call_gemini(self, image_path: str, prompt: str) -> tuple:
        client = genai.Client(api_key=self.api_key or os.environ.get("GEMINI_API_KEY", ""))
        b64, mime = self._encode_image(image_path)
        response = client.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Part.from_bytes(data=base64.b64decode(b64), mime_type=f"image/{mime}"),
                prompt,
            ],
            config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=512),
        )

        raw_text = response.text or ""
        finish = str(response.candidates[0].finish_reason) if response.candidates else "unknown"
        provider_meta = {
            "model": self.model,
            "finish_reason": finish,
        }
        return raw_text, provider_meta
