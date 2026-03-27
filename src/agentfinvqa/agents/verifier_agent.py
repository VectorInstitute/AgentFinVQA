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

from ..langfuse_integration.tracing import close_span, open_llm_span
from ..tools.verifier_tool import VerifierTool
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

        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"
        task_description = (
            f"You are auditing a vision agent's answer to a chart question.\n\n"
            f"Question         : {sample.question}\n"
            f"Question Type    : {question_type}\n"
            f"Image path       : {image_path}\n\n"
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
            expected_output='JSON object: {"verdict": "confirmed" | "revised", "answer": "...", "reasoning": "..."}',
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

        close_span(verifier_span, output=parsed)
        return task_description, parsed, not parse_ok, raw_text, tool_traces
