# AgentFinVQA

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/AgentFinVQA/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/AgentFinVQA)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/AgentFinVQA)

A multi-agent evaluation framework for Visual Question Answering on financial charts, built on the [ChartQAPro](https://huggingface.co/datasets/ahmed-masry/ChartQAPro) dataset. The framework decomposes chart QA into an explicit **Plan → Inspect → Explain** loop, producing fully traceable evaluation artifacts for each sample.

## Overview

Unlike single-pass VLM approaches, AgentFinVQA coordinates multiple specialized agents:

- **PlannerAgent** — text-only LLM that generates a structured inspection plan without seeing the image
- **OcrReaderTool** — focused VLM call that transcribes all visible text from the chart
- **VisionAgent** — CrewAI-orchestrated agent that executes the plan and produces an answer
- **VerifierAgent** — second VLM that critiques the draft answer and confirms or revises it

Each run produces a **Model Evaluation Packet (MEP)** — a portable JSON artifact capturing the full trace: inspection plan, vision reasoning, verifier critique, tool call logs, timestamps, and errors. MEPs enable reproducible evaluation, post-hoc explainability analysis, and model comparison across VLM backends.

### Architecture

```
Input Sample (question, chart image, expected answer)
           │
           ▼
    PlannerAgent (text-only LLM)
    • Produces a structured JSON inspection plan
    • Does NOT see the image
           │ plan.steps
           ▼
    OcrReaderTool (optional)
    • Single VLM call focused on text transcription
    • Produces structured JSON of all visible text
           │ ocr_text
           ▼
    VisionAgent (CrewAI + tools)
    • Executes the plan step-by-step
    • Produces answer + explanation
           │ draft_answer
           ▼
    VerifierAgent (single VLM call)
    • Reviews draft answer against chart image
    • Verdict: CONFIRM or REVISE
           │
           ▼
    MEP (Model Evaluation Packet)
    • JSON artifact stored to disk
    • Optionally traced in Langfuse
```

## Installation

The development environment is managed with [uv](https://github.com/astral-sh/uv).

Install core dependencies:

```bash
uv sync
source .venv/bin/activate
```

Install the agentic pipeline dependencies (CrewAI, Google GenAI, Streamlit dashboard):

```bash
uv sync --group agentic-xai-eval
source .venv/bin/activate
```

Install all dependencies including docs:

```bash
uv sync --all-groups
source .venv/bin/activate
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (for planner / verifier backends) |
| `GEMINI_API_KEY` | Google Gemini API key (for vision backend) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (optional — enables tracing) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (optional) |
| `LANGFUSE_HOST` | Langfuse host URL (optional — defaults to cloud) |

## Running the Pipeline

Generate MEPs for a subset of ChartQAPro:

```bash
uv run --env-file .env -m agentfinvqa.runner.run_generate_meps \
    --dataset chartqapro \
    --split test \
    --n 200 \
    --config openai_gemini \
    --workers 8 \
    --out meps/
```

See `notebooks/run_pipeline.ipynb` for an interactive walkthrough.

## Evaluation

Evaluate generated MEPs and explore results:

```bash
# Summarize accuracy across MEPs
uv run -m agentfinvqa.eval.summarize --mep-dir meps/

# Launch the Streamlit dashboard
uv run streamlit run src/agentfinvqa/eval/dashboard.py -- --mep-dir meps/
```

See `notebooks/analysis.ipynb` for detailed analysis examples.

## Project Structure

```
src/agentfinvqa/
├── agents/             # PlannerAgent, VisionAgent, VerifierAgent
│   └── prompts/        # System prompt templates
├── datasets/           # ChartQAPro dataset loader
├── eval/               # Evaluation utilities, metrics, Streamlit dashboard
├── langfuse_integration/  # Observability: tracing, client, dataset registration
├── mep/                # Model Evaluation Packet schema and I/O
├── runner/             # End-to-end pipeline runner
├── tools/              # OcrReaderTool, VisionQATool (CrewAI tools)
└── utils/              # Hashing, strict JSON parsing, timing
notebooks/
├── run_pipeline.ipynb  # Interactive pipeline walkthrough
└── analysis.ipynb      # MEP analysis and visualization
```

## Developing

### Installing dev dependencies

```bash
uv sync --dev
source .venv/bin/activate
```

### Running pre-commit hooks

```bash
uv run pre-commit run --all-files
```

> **Note for Vector Institute HPC users:** The Compute Canada pip configuration
> (set via `PIP_CONFIG_FILE`) interferes with pre-commit's environment setup,
> causing source builds of Rust-based tools (ruff, typos) instead of downloading
> pre-built wheels. To avoid this, either run with:
>
> ```bash
> PIP_CONFIG_FILE=/dev/null uv run pre-commit run --all-files
> ```
>
> Or add `export PIP_CONFIG_FILE=/dev/null` to your `~/.bashrc`.

### Running tests

```bash
uv run pytest
```
