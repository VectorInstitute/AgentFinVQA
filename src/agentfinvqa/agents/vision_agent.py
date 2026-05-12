"""VisionAgent — tool-using CrewAI agent that calls vision_qa_tool once.

The agent orchestrates the VLM call via vision_qa_tool and returns strict JSON
{answer, explanation}.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from crewai import LLM, Agent, Crew, Task

from ..datasets.perceived_sample import PerceivedSample
from ..langfuse_integration.tracing import close_span, open_llm_span
from ..mep.schema import MEPColorArea
from ..tools.color_area_tool import format_color_area_block_for_vision
from ..tools.vision_qa_tool import VisionQATool
from ..utils.json_strict import parse_strict
from ..utils.model_compat import openai_temperature


VISION_PROMPT_PATH = Path(__file__).parent / "prompts" / "vision.txt"

VISION_REQUIRED_KEYS = ["choice_analysis", "answer", "explanation"]


def _load_template() -> str:
    """
    Retrieve the prompt template string from the vision configuration file.

    Returns
    -------
    str
        The raw text of the vision prompt template.
    """
    return VISION_PROMPT_PATH.read_text()


def _format_choice_blocks(sample: PerceivedSample) -> tuple[str, str]:
    labeled_choices = sample.metadata.get("choices_labeled") or []
    if labeled_choices:
        label_lines = ["Choice labels (letter → text):"]
        for item in labeled_choices:
            label_lines.append(f"  {item.get('label')}: {item.get('text')}")
        return "", "\n".join(label_lines)
    if sample.choices:
        return f"Choices: {', '.join(sample.choices)}", ""
    return "", ""


def _format_context_block(sample: PerceivedSample) -> str:
    if not sample.context:
        return ""
    lines = ["Conversation context:"]
    for turn in sample.context:
        lines.append(f"  {turn.get('role', 'user')}: {turn.get('content', '')}")
    return "\n".join(lines)


def _format_plan_steps(plan: dict) -> str:
    steps = plan.get("steps") or []
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))


def _format_ocr_block(ocr_result: Optional[dict]) -> str:
    if not ocr_result:
        return ""
    lines = ["Pre-extracted text from chart (use as ground truth for visible text):"]
    chart_type = (ocr_result.get("chart_type") or "").strip()
    title = (ocr_result.get("title") or "").strip()
    if chart_type:
        lines.append(f"  Chart type : {chart_type}")
    if title:
        lines.append(f"  Title      : {title}")
    x_axis = ocr_result.get("x_axis", {})
    x_label = (x_axis.get("label") or "").strip()
    x_ticks = x_axis.get("ticks", [])
    if x_label or x_ticks:
        lines.append(f"  X-axis     : label={x_label!r}  ticks={x_ticks}")
    y_axis = ocr_result.get("y_axis", {})
    y_label = (y_axis.get("label") or "").strip()
    y_ticks = y_axis.get("ticks", [])
    if y_label or y_ticks:
        lines.append(f"  Y-axis     : label={y_label!r}  ticks={y_ticks}")
    if ocr_result.get("legend"):
        lines.append(f"  Legend     : {ocr_result['legend']}")
    if ocr_result.get("data_labels"):
        lines.append(f"  Data labels: {ocr_result['data_labels']}")
    if ocr_result.get("annotations"):
        lines.append(f"  Annotations: {ocr_result['annotations']}")
    return "\n".join(lines)


_MULTI_SELECT_KEYWORDS = (
    "select all",
    "all that apply",
    "which of the following are",
    "all correct",
    "all applicable",
    "all that are",
)


def _is_multi_select(sample: PerceivedSample) -> bool:
    """Return True for multi-label MCQ (select-all-that-apply style)."""
    if (sample.metadata or {}).get("question_type_raw") == "multiple_choice":
        return True
    q = sample.question.lower()
    return any(kw in q for kw in _MULTI_SELECT_KEYWORDS)


def _format_multi_select_block(sample: PerceivedSample) -> str:
    """Return a prominent multi-select instruction block, or empty string."""
    if not _is_multi_select(sample):
        return ""
    labeled = (sample.metadata or {}).get("choices_labeled") or []
    letters = ", ".join(item.get("label", "") for item in labeled if item.get("label"))
    letters_hint = f" ({letters})" if letters else ""
    return (
        "⚠ MULTI-SELECT MCQ — select ALL correct options, not just one.\n"
        f"  Evaluate EACH choice{letters_hint} independently against the chart.\n"
        "  Your `answer` must contain ALL correct letters concatenated (e.g. 'ACD').\n"
        "  Do NOT stop at the first match. Include every letter that is supported by visual evidence.\n\n"
    )


def _format_caption_block(sample: PerceivedSample) -> str:
    """Format the verified caption from sample metadata as a context hint.

    The ``verified_caption`` field in FinMME contains analyst-written descriptions
    that name the chart's subject — information the visual model cannot always infer
    from the image alone. Injecting it prevents UNANSWERABLE over-refusals on charts
    that lack visible titles.

    Returns empty string when no caption is available.
    """
    caption = (sample.metadata.get("verified_caption") or "").strip()
    if not caption:
        return ""
    return f"Chart context (analyst caption — use as background, not as the answer):\n  {caption}"


def _format_legend_grounding_block(legend_map: Optional[List[dict]]) -> str:
    """Format the structured legend map as readable lines for the vision prompt.

    Parameters
    ----------
    legend_map : list of dict, optional
        Each entry has ``label``, ``color_description``, ``line_style``,
        and ``confidence`` keys as returned by LegendGrounderTool.

    Returns
    -------
    str
        A formatted block to inject into the vision prompt, or empty string
        if ``legend_map`` is empty or None.
    """
    if not legend_map:
        return ""
    lines = [
        "Pre-mapped legend (treat as ground truth — do not reassign colors):",
    ]
    for entry in legend_map:
        label = entry.get("label", "")
        color = entry.get("color_description", "unknown")
        style = entry.get("line_style", "unknown")
        conf = entry.get("confidence", 0.0)
        lines.append(f"  {label:<30} → {color} {style} (confidence: {conf:.2f})")
    lines.append(
        "\nINSTRUCTION: When extracting values for any series, you MUST use "
        "the color mapping above. Do not visually re-identify which line "
        "belongs to which series — use the pre-mapped colors as ground truth. "
        "If a color in the legend map contradicts what you see, trust the "
        "legend map and note the discrepancy in your explanation."
    )
    return "\n".join(lines)


def build_vision_task_description(
    sample: PerceivedSample,
    plan: dict,
    ocr_result: Optional[dict] = None,
    legend_map: Optional[List[dict]] = None,
    color_area: Optional[MEPColorArea] = None,
    prepend_instruction: Optional[str] = None,
) -> str:
    """
    Compose the complete task instruction for the vision agent.

    Integrates the sample details, the inspection plan, optional OCR data,
    and an optional pre-mapped legend block into a single prompt for the model.

    Parameters
    ----------
    sample : PerceivedSample
        The data sample containing the question and image path.
    plan : dict
        The inspection plan generated by the PlannerAgent.
    ocr_result : dict, optional
        Pre-extracted text data from the chart image.
    legend_map : list of dict, optional
        Structured color-to-series mapping from LegendGrounderTool.
    color_area : MEPColorArea, optional
        Optional pixel-area breakdown from the color-area tool stage.
    prepend_instruction : str, optional
        Extra instruction to prepend to the prompt (used for compliance retry).

    Returns
    -------
    str
        The rendered prompt ready for the vision-capable agent.
    """
    template = _load_template()
    choices_block, choice_labels_block = _format_choice_blocks(sample)
    context_block = _format_context_block(sample)
    plan_steps_block = _format_plan_steps(plan)
    ocr_block = _format_ocr_block(ocr_result)
    legend_grounding_block = _format_legend_grounding_block(legend_map)
    color_area_block = format_color_area_block_for_vision(color_area)
    caption_block = _format_caption_block(sample)
    multi_select_block = _format_multi_select_block(sample)

    rendered = template.format(
        image_path=sample.image_path,
        question=sample.question,
        choices_block=choices_block,
        choice_labels_block=choice_labels_block,
        context_block=context_block,
        ocr_block=ocr_block,
        legend_grounding_block=legend_grounding_block,
        color_area_block=color_area_block,
        caption_block=caption_block,
        multi_select_block=multi_select_block,
        plan_steps_block=plan_steps_block,
    )

    if prepend_instruction:
        rendered = prepend_instruction.rstrip("\n") + "\n\n" + rendered

    return rendered


def _build_llm(backend: str, model: str, api_key: Optional[str]) -> LLM:
    """
    Configure a CrewAI LLM instance based on the chosen provider.

    Parameters
    ----------
    backend : {'openai', 'gemini'}
        The model provider.
    model : str
        The specific model identifier (e.g., 'gpt-4o').
    api_key : str, optional
        The API key to use. Defaults to provider-specific env variables.

    Returns
    -------
    LLM
        The initialized model interface.

    Raises
    ------
    ValueError
        If an unsupported `backend` is specified.
    """
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
    raise ValueError(f"Unknown vision agent backend: {backend!r}")


class VisionAgent:
    """
    An agent that performs visual analysis using a tool-based architecture.

    The `VisionAgent` acts as an orchestrator that uses text-only reasoning
    to decide on a tool-based inspection strategy. It leverages the
    `VisionQATool` to actually interact with the multimodal model and
    extract visual data.
    """

    def __init__(
        self,
        agent_backend: str = "gemini",
        agent_model: str = "gemini-2.5-flash-lite",
        vision_backend: str = "gemini",
        vision_model: str = "gemini-2.5-flash-lite",
        agent_api_key: Optional[str] = None,
        vision_api_key: Optional[str] = None,
    ):
        """
        Set up the vision agent with its orchestration and vision backends.

        Parameters
        ----------
        agent_backend : str, default 'gemini'
            The backend for the orchestration logic.
        agent_model : str, default 'gemini-2.5-flash-lite'
            The model used for reasoning.
        vision_backend : str, default 'gemini'
            The backend for the actual image inspection tool.
        vision_model : str, default 'gemini-2.5-flash-lite'
            The model used for visual perception.
        agent_api_key : str, optional
            API key for the orchestrator.
        vision_api_key : str, optional
            API key for the vision tool.
        """
        self.agent_backend = agent_backend
        self.agent_model = agent_model
        self.vision_backend = vision_backend
        self.vision_model = vision_model
        self.agent_api_key = agent_api_key
        self.vision_api_key = vision_api_key

    def _build_tool(self, lf_trace: Any = None) -> VisionQATool:
        """
        Instantiate the vision tool with the configured vision model.

        Parameters
        ----------
        langfuse_trace : Any, optional
            A tracing object for observability.

        Returns
        -------
        VisionQATool
            The tool used by the agent to see images.
        """
        key = self.vision_api_key or (
            os.environ.get("OPENAI_API_KEY", "")
            if self.vision_backend == "openai"
            else os.environ.get("GEMINI_API_KEY", "")
        )
        return VisionQATool(
            backend=self.vision_backend,
            model=self.vision_model,
            api_key=key,
            lf_trace=lf_trace,
        )

    def run(
        self,
        sample: PerceivedSample,
        plan: dict,
        lf_trace: Any = None,
        ocr_result: Optional[dict] = None,
        legend_map: Optional[List[dict]] = None,
        color_area: Optional[MEPColorArea] = None,
        prepend_instruction: Optional[str] = None,
    ) -> Tuple[str, dict, bool, str, List[dict]]:
        """
        Execute the vision analysis process for a single chart question.

        Coordinates the CrewAI agent to follow the plan, call the vision tool,
        and return a structured response.

        Parameters
        ----------
        sample : PerceivedSample
            The question and image to analyze.
        plan : dict
            The inspection procedure to follow.
        langfuse_trace : Any, optional
            Trace object for execution tracking.
        ocr_result : dict, optional
            Ground-truth OCR data for grounding.
        legend_map : list of dict, optional
            Structured color-to-series mapping from LegendGrounderTool.
        color_area : MEPColorArea, optional
            Pixel-area breakdown for size-comparison questions.
        prepend_instruction : str, optional
            Extra instruction to prepend (used for compliance retry).

        Returns
        -------
        task_description : str
            The prompt used for the task.
        parsed : dict
            The extracted 'answer' and 'explanation'.
        parse_error : bool
            True if the model output was not valid JSON.
        raw_text : str
            The full string returned by the model.
        tool_traces : list of dict
            A log of tool interactions during the run.
        """
        tool = self._build_tool(lf_trace=lf_trace)
        llm = _build_llm(self.agent_backend, self.agent_model, self.agent_api_key)
        task_description = build_vision_task_description(
            sample,
            plan,
            ocr_result=ocr_result,
            legend_map=legend_map,
            color_area=color_area,
            prepend_instruction=prepend_instruction,
        )

        vision_span = open_llm_span(
            lf_trace,
            name="vision_agent",
            input_data={"task_description": task_description},
            model=self.agent_model,
            metadata={"backend": self.agent_backend},
        )

        agent = Agent(
            role="Chart Reading Vision Agent",
            goal=(
                "Answer chart questions by calling vision_qa_tool exactly once, "
                "then output strict JSON with 'choice_analysis', 'answer', and 'explanation'."
            ),
            backstory=(
                "You are a precise chart analysis agent. You use vision_qa_tool to "
                "inspect chart images, score each option in 'choice_analysis', and produce grounded answers. "
                "You follow inspection plans step by step and never hallucinate."
            ),
            llm=llm,
            tools=[tool],
            verbose=False,
            allow_delegation=False,
            max_iter=3,  # limit iterations to prevent runaway tool calls
        )

        is_ms = _is_multi_select(sample)
        task = Task(
            description=task_description,
            expected_output=(
                'JSON object: {"choice_analysis": {...}, "answer": "...", "explanation": "..."}\n'
                "MULTI-SELECT: answer must contain ALL correct letters concatenated (e.g. 'ACD'), not just one."
                if is_ms
                else 'JSON object: {"choice_analysis": {...}, "answer": "...", "explanation": "..."}'
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, parse_ok = parse_strict(raw_text, required_keys=VISION_REQUIRED_KEYS)

        tool_traces = tool.pop_traces()

        close_span(vision_span, output=parsed if parsed else {"parse_error": True})

        return task_description, parsed, not parse_ok, raw_text, tool_traces
