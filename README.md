# AgentFinVQA

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/AgentFinVQA/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/AgentFinVQA)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/AgentFinVQA)

A multi-agent evaluation framework for Visual Question Answering on financial charts, supporting both [ChartQAPro](https://huggingface.co/datasets/ahmed-masry/ChartQAPro) and [FinMME](https://huggingface.co/datasets/luojunyu/FinMME). The framework decomposes chart QA into an explicit **Plan → Inspect → Explain** loop, producing fully traceable evaluation artifacts for each sample.

## Supported datasets

| Dataset | HF handle | Notes |
| --- | --- | --- |
| ChartQAPro | `ahmed-masry/ChartQAPro` | Multi-turn chart QA with factoid/mcq/unanswerable tasks. Images cached under `data/chartqapro_images/` by default. |
| FinMME | `luojunyu/FinMME` | Financial multi-modal evaluation benchmark (only a `train` split on HF; use slicing like `train[:1000]`). Images cached under `data/finmme_images/` by default. |

Select the dataset at runtime with `--dataset {chartqapro|finmme}`; all downstream tooling (Langfuse registration, output directories) key off the same slug.

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

To target FinMME, switch `--dataset finmme`. The loader automatically writes FinMME charts to `data/finmme_images/` unless you override `--image_dir`. Note: Hugging Face only exposes a `train` split for `luojunyu/FinMME`. Any request for `test` is remapped to `train` internally, so use slicing (e.g. `--split train[:200]`) to simulate held-out subsets.

### Batch helpers

For headless or batch jobs, use the helper scripts:

- Python entrypoint: `python scripts/run_finmme_batch.py --n 500 --split train[:500] --config gemini_gemini --workers 8`
- Bash wrapper (convenient for schedulers): `scripts/run_finmme_batch.sh --n 500 --split train[:500]`

Both commands default to loading `.env` from the repo root; override by setting `ENV_FILE=/path/to/.env` before calling the bash script or by passing `--env_file` to the Python script.

### SLURM batch job template

If you need to submit a single FinMME run to SLURM (MEP generation + post-eval in one job), use the monolithic template:

```bash
sbatch scripts/slurm_run_finmme_batch.slrm
```

This script sets a time limit (`--time=0-04:00:00`), runs the existing bash helper, and logs output to `logs/slurm_finmme_<jobid>.out/err`. Edit the SBATCH directives and the `run_finmme_batch.sh` arguments inside the file to match your workload.

### SLURM two-stage pipeline (recommended for large runs)

For large runs, use the chained pipeline that separates MEP generation from LLM judge evaluation. This uses the [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch) for judge scoring (50% cost reduction, async):

```bash
scripts/submit_pipeline.sh \
    --split "train[3000:5000]" \
    --n 2000 \
    --workers 8 \
    --langfuse \
    --resume \
    --planner_model gemini-2.5-flash \
    --vision_model gemini-2.5-flash \
    --ocr_model gemini-2.5-flash-lite \
    --verifier_model gemini-2.5-flash \
    --judge_model gemini-2.5-flash-lite
```

This submits two SLURM jobs chained with `--dependency=afterok`:

| Job | Script | What it does |
|---|---|---|
| 1 | `slurm_generate_meps.slrm` | MEP generation only |
| 2 | `slurm_submit_judge_batch.slrm` | Uploads all judge prompts to Gemini Batch API and exits immediately |

Job 2 only runs if job 1 succeeds. When job 2 completes it prints the commands to check status and retrieve results:

```bash
# Check if Gemini batch job is done
python3 -m agentfinvqa.eval.eval_outputs_batch status \
    --state output/metrics_<label>.jsonl.batch_state.json

# Download results when ready
python3 -m agentfinvqa.eval.eval_outputs_batch retrieve \
    --state output/metrics_<label>.jsonl.batch_state.json
```

See `notebooks/run_pipeline.ipynb` for an interactive walkthrough.

## Evaluation

Evaluation is a four-step pipeline. MEPs must be generated first (see **Running the Pipeline** above).

### Step 1 — Generate metrics from MEPs

The batch helper (`run_finmme_batch.py`) can do this automatically with `--post_eval`:

```bash
uv run scripts/run_finmme_batch.sh \
    --n 500 --split train[:500] \
    --post_eval --use_judge \
    --workers 4
```

`--post_eval` writes `output/metrics_<label>.jsonl` and `output/trace_metrics_<label>.jsonl` at the end of the run.

To evaluate an existing MEP directory independently:

```bash
uv run -m agentfinvqa.eval.eval_outputs \
    --mep_dir meps/gemini_gemini/finmme/train[:500] \
    --out output/metrics.jsonl \
    --judge_model gemini-2.5-flash-lite   # omit --no_judge to include LLM rubric scores
```

### Step 2 — Failure taxonomy (optional)

Classifies each incorrect answer into a failure category (legend confusion, extraction error, etc.):

```bash
uv run -m agentfinvqa.eval.error_taxonomy \
    --mep_dir meps/gemini_gemini/finmme/train[:500] \
    --metrics_file output/metrics.jsonl \
    --out output/taxonomy.jsonl \
    --model gemini-2.5-flash-lite
```

### Step 3 — Summarize accuracy

Aggregates metrics by config and question type:

```bash
uv run -m agentfinvqa.eval.summarize \
    --metrics output/metrics.jsonl \
    --out output/summary.csv
```

### Step 4 — Explore in the dashboard

```bash
uv run streamlit run src/agentfinvqa/eval/dashboard.py
```

The dashboard auto-discovers MEP directories under `meps/` and metric files under `output/`. Use the sidebar to select paths and filters.

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
