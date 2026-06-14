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
    • MCQ-aware: checks/eliminates each choice; multi-select guidance
    • Does NOT see the image
           │ plan.steps
           ▼
    OcrReaderTool
    • Single VLM call focused on text transcription
    • Produces structured chart metadata (axes, legend, data labels)
           │ ocr_text + chart_type
           ▼
    LegendGrounderTool (conditional)
    • Triggered for line/bar/scatter/area/pie/donut charts
    • Maps legend labels → color descriptions + RGB + line style
    • Compliance check: re-runs if legend entries are missing
           │ legend_map
           ▼
    VisionAgent (CrewAI + VisionQATool)
    • Executes the plan using OCR text and legend map as ground truth
    • Single-select MCQ / multi-select MCQ / open-ended answer paths
    • Produces answer + explanation + per-choice confidence analysis
           │ draft_answer
           ▼
    Forced-Choice Retry (conditional)
    • If vision returns UNANSWERABLE and MCQ choices exist:
      re-runs vision with explicit "FORCED CHOICE" instruction
           │ draft_answer (revised if retry triggered)
           ▼
    VerifierAgent (CrewAI + VerifierTool)
    • Reviews draft answer against chart image
    • Adds reluctance hint when vision confidence is high (≥ 0.85)
    • Verdict: CONFIRM or REVISE + self-reported confidence
    • Confidence gate: downgrades low-confidence revisions (< 0.75)
           │
           ▼
    MEP (Model Evaluation Packet)
    • JSON artifact stored to disk
    • Optionally traced in Langfuse
```

## Results

### FinMME (250-sample train slice; v8 / v9 / v10 scale-up to 1,250)

| Run | Accuracy | Δ vs baseline | Key change |
|-----|----------|---------------|------------|
| `no_legend_grounding` | 48.0% | — | Baseline |
| `fixes_v1` | 50.4% | +2.4 pp | Legend grounding, caption injection, token limits |
| `fixes_v2` | 51.6% | +3.6 pp | Disable thinking tokens, MCQ choices to verifier |
| `fixes_v3` | 51.6% | +3.6 pp | Thinking budget = 512 |
| `fixes_v4_g3flash` | 56.0% | +8.0 pp | Gemini 3 Flash, forced-choice retry, MCQ-aware planner |
| `fixes_v5_multiselect` | **69.4%** | +21.4 pp | Full multi-select MCQ support |
| `fixes_v7_g3flash_conf_gate` | **69.6%** | +21.6 pp | Confidence gate fix, fresh g3flash run |
| `fixes_v8_g3flash_color_area` | **71.2%** *(n = 1,250)* | +23.2 pp | Color-area OpenCV pre-hint; see `results.md` §8b |
| `fixes_v9_g3flash_related_sents` | **71.3%** *(n = 1,250)* | +23.3 pp | Verifier + `related_sentences` + caption cross-check; **~2.4× tighter latency tail** vs v8 (p95 87 s vs 209 s) |
| `fixes_v10_g3flash_choice_conflict` | **71.1%** *(n = 1,250)* | +23.1 pp | v9 + high-confidence **choice-conflict** flag for verifier |

**vs. FinMME paper (Table 3, Gemini Flash 2.0 = 51.85%):** our best **250-ID** ladder run achieves **+17.8 pp** (v7 mean `answer_accuracy` vs paper headline — metric families differ).

**Fair same-model baseline (Gemini-3 Flash Preview structured zero-shot vs agent):**

- **Primary (matched n = 1,250 train IDs):** zero-shot mean `answer_accuracy` **63.56%** vs agents — **v8** **71.24%** (**+7.68 pp**, exact **+8.72 pp**, McNemar χ² = 68.21, p ≈ 1.1×10⁻¹⁶); **v9** **71.28%** (**+7.72 pp**, exact **+8.16 pp**, χ² = 61.45, p ≈ 4.5×10⁻¹⁵); **v10** **71.08%** (**+7.52 pp**, exact **+7.84 pp**, χ² = 57.37, p ≈ 3.6×10⁻¹⁴). All three crush zero-shot; pairwise between agents nothing is significant (v9 vs v8 p = 0.56; v10 vs v8 p = 0.34; v9 vs v10 p = 0.75). v9's distinctive contribution is **latency-tail tightening**, not extra accuracy. Full zero-shot train file: **11,099** rows — always join on `sample_id` before comparing.
- **Legacy 250-ID snapshot (strict exact, ablation era):** zero-shot **52.8%** vs agent v7 **62.8%** → **+10.0 pp** — useful historically; see `results.md` §8b for context.

> Note: the initial zero-shot Gemini-3 export had parser-related empty predictions; robust extraction + repair recovered many rows before the full 11k re-run.

Detailed per-run analysis, per-type breakdowns, and paper comparison are in [`notebooks/results_analysis.ipynb`](notebooks/results_analysis.ipynb).
For camera-ready citation numbers, see [`markdown/camera_ready_metrics.md`](markdown/camera_ready_metrics.md).

---

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

### Sample selection (`--split` and `--n`)

- **`--split`** — Hugging Face split and optional row slice (e.g. `test`, `test[1000:]`, `train[:500]`). This selects *which rows* of the dataset are loaded.
- **`--n`** — Maximum number of **perceived samples** to process after that slice. Use **`0` or a negative value for no cap** (process the entire loaded slice). Positive `n` stops early once enough samples are materialized.

So “run the whole `test` split” is typically:

```bash
--split test --n 0
```

A partial slice with no further cap:

```bash
--split 'test[1000:]' --n 0
```

### Batch helpers

The recommended entrypoints for all datasets are `scripts/run_batch.py` and its bash wrapper `scripts/run_batch.sh`. These are dataset-agnostic and run generation + post-evaluation in a single MEP pass (metrics, traces, failure taxonomy, and summary in one go):

```bash
scripts/run_batch.sh \
    --dataset chartqapro \
    --split test \
    --n 500 \
    --config gemini_gemini \
    --workers 8 \
    --post_eval \
    --use_judge \
    --langfuse \
    --resume \
    --eval_label chartqapro_test_n500
```

To skip generation and run post-eval on existing MEPs only:

```bash
scripts/run_batch.sh \
    --dataset chartqapro \
    --split test \
    --config gemini_gemini \
    --eval_only \
    --use_judge \
    --langfuse \
    --eval_label chartqapro_test_n500
```

Both commands default to loading `.env` from the repo root. `--langfuse` pushes all numeric eval scores (accuracy, judge rubric scores) back to the originating Langfuse traces.

**Verifier ablation:** pass `--no_verifier` to skip the VerifierAgent (Pass 2.5); the pipeline keeps the planner/vision draft without a revise step. Supported by `scripts/run_batch.py`, `scripts/run_finmme_batch.py`, and `scripts/submit_pipeline.sh` (see below).

### SLURM — single job (generation + eval)

Submit a complete run (generation and post-eval) as one SLURM job:

```bash
sbatch scripts/slurm_run_batch.slrm
```

Environment variables (`DATASET`, `SPLIT`, `N`, `CONFIG`, `WORKERS`, `LANGFUSE`, `RESUME`, `NO_VERIFIER`, and model overrides) are passed through from the environment or from `submit_pipeline.sh` via `--export`. Set `NO_VERIFIER=1` before `sbatch` if you call `slurm_run_batch.slrm` without the submit helper.

### SLURM — eval only

To run post-eval on MEPs that already exist:

```bash
scripts/submit_eval.sh \
    --dataset chartqapro \
    --split test \
    --use_judge \
    --langfuse \
    --out_label chartqapro_test_n500
```

This submits `slurm_eval_only.slrm` as a single SLURM job. You can chain it after a generation job:

```bash
scripts/submit_eval.sh \
    --dataset chartqapro \
    --split test \
    --use_judge \
    --langfuse \
    --after <JOB_ID>
```

### SLURM — two-stage pipeline (async judge, recommended for large runs)

For large runs, use the chained pipeline that separates MEP generation from LLM judge evaluation. This uses the [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch) for judge scoring (50% cost reduction, async):

```bash
scripts/submit_pipeline.sh \
    --dataset finmme \
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

**Defaults (you usually do not need to repeat model flags)** — `submit_pipeline.sh` already defaults to `--config gemini_gemini`, `--workers 8`, and the same planner/vision/OCR/verifier/judge models as above. Override only what you change. Add `--langfuse` and/or `--resume` when you want tracing or skip-existing MEPs. For a **verifier-off ablation**, add `--no_verifier`.

**Full split without counting rows** — use `--n 0` (see the *Sample selection* subsection above):

```bash
scripts/submit_pipeline.sh \
    --dataset chartqapro \
    --split test \
    --n 0 \
    --no_verifier \
    --resume
```

**Not the same as `run_batch.sh --post_eval`** — Job 1 in this chain runs **MEP generation only** (via `slurm_run_batch.slrm` → `run_batch.sh` **without** `--post_eval`). Job 2 submits prompts to the **Gemini Batch API** for async judge scoring. For **local / threaded** post-eval in one process (metrics, traces, taxonomy, summary written immediately), use `scripts/run_batch.sh` with `--post_eval` (and `--use_judge` if you want the LLM judge path during that step) instead of this two-stage pipeline.

This submits two SLURM jobs chained with `--dependency=afterok`:

| Job | Script | What it does |
|---|---|---|
| 1 | `slurm_run_batch.slrm` | MEP generation |
| 2 | `slurm_submit_judge_batch.slrm` | Uploads all judge prompts to Gemini Batch API and exits immediately |

**Where MEPs and batch metrics go**

- **MEP directory** (generation output): `meps/<CONFIG>/<dataset>/<split>/` when the verifier is on (default). With **`--no_verifier`**, MEPs go under **`meps/<CONFIG>/<dataset>/no_verifier/<split>/`** so verifier-on and verifier-off runs do not overwrite each other. Example: `meps/gemini_gemini/chartqapro/test/` vs `meps/gemini_gemini/chartqapro/no_verifier/test/`.
- **Batch judge file** (job 2): `output/metrics_<out_label>.jsonl` plus `output/metrics_<out_label>.jsonl.batch_state.json`. If you omit `--out_label`, the script sets `<out_label>` to `{dataset}_{sanitized_split}` and appends `_no_verifier` when `--no_verifier` is set (e.g. `chartqapro_test_no_verifier`), so different runs do not overwrite `metrics_test.jsonl`.

Job 2 only runs if job 1 succeeds. When job 2 completes it prints the commands to check status and retrieve results:

```bash
# Check if Gemini batch job is done
python3 -m agentfinvqa.eval.eval_outputs_batch status \
    --state output/metrics_<label>.jsonl.batch_state.json

# Download results when ready
python3 -m agentfinvqa.eval.eval_outputs_batch retrieve \
    --state output/metrics_<label>.jsonl.batch_state.json
```

The batch job display name in the Gemini console follows the metrics filename (from `--out_label` or the auto-generated label above).

See `notebooks/run_pipeline.ipynb` for an interactive walkthrough.

### Zero-shot baseline

For a single VLM call per sample (no agents), use `baselines/run_zeroshot.py`. Outputs metrics JSONL in the same schema as the agent pipeline for easy comparison.

- **Structured prompt (default)** — rules plus JSON `answer` / `explanation` format.
- **Minimal prompt** — bare `Question: … / Answer:` (first line of the model reply is scored); use `--prompt_style minimal`.

On SLURM:

```bash
baselines/submit_zeroshot.sh --dataset chartqapro --split test --prompt_style minimal
```

## Evaluation

The `run_batch.sh` scripts above handle the full eval pipeline automatically via `--post_eval` or `--eval_only`. All four artifacts are produced in a **single MEP pass**:

| Artifact | Path |
|---|---|
| Per-sample metrics | `output/metrics_<label>.jsonl` |
| Trace metrics | `output/trace_metrics_<label>.jsonl` |
| Failure taxonomy | `output/taxonomy_<label>.jsonl` |
| Summary CSV | `output/summary_<label>.csv` |

If `--langfuse` is set, all numeric scores are pushed back to the originating Langfuse traces.

To evaluate an existing MEP directory with the low-level CLI directly:

```bash
uv run -m agentfinvqa.eval.eval_outputs \
    --mep_dir meps/gemini_gemini/chartqapro/test \
    --out output/metrics.jsonl \
    --judge_model gemini-2.5-flash-lite
```

### Explore in the dashboard

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

Run the full test suite:

```bash
uv run pytest
```

Run a specific test file:

```bash
uv run pytest tests/agentfinvqa/test_legend_grounding.py
uv run pytest tests/agentfinvqa/test_finmme_loader.py
```

Run a specific test class or test:

```bash
uv run pytest tests/agentfinvqa/test_legend_grounding.py::TestComplianceCheck
uv run pytest tests/agentfinvqa/test_legend_grounding.py::TestLegendGrounderTool::test_api_error_returns_fallback_json
```

Run with coverage:

```bash
uv run pytest --cov=src/agentfinvqa --cov-report=term-missing
```

Run only integration tests (marked with `@pytest.mark.integration_test`):

```bash
uv run pytest -m integration_test
```

### Test coverage

| Test file | What it covers |
|---|---|
| `test_finmme_loader.py` | FinMME dataset loader — option parsing, question type mapping, sample construction, multi-letter answers |
| `test_legend_grounding.py` | Legend grounding pipeline stage — formatting, prompt injection, MEP schema, gate logic, compliance check, `LegendGrounderTool` with mocked API calls |
| `test_placeholder.py` | Placeholder (no-op) to keep pytest from exiting with "no tests collected" |

#### Legend grounding tests in detail

`test_legend_grounding.py` covers the full legend grounding feature added between the OCR and vision stages:

- **`TestFormatLegendGroundingBlock`** — prompt block formatting: empty/None inputs, header, per-entry label/color/style/confidence rendering, missing-field robustness
- **`TestBuildVisionTaskDescription`** — `legend_map` injected into vision prompt, absent when empty/None, `prepend_instruction` appears first, OCR block coexists with legend block
- **`TestMEPLegendGrounding`** — schema defaults, all fields settable, `MEP.to_dict()` serialisation, `None` case
- **`TestLegendGroundingGate`** — all 7 gated chart types (`line`, `bar`, `scatter`, `area`, `bar_grouped`, `bar_stacked`, `combination`) pass; `pie`, `table`, `dashboard` and others are blocked; single-legend and `grounder=None` are skipped
- **`TestComplianceCheck`** — label present/absent in explanation, any-label match, case-insensitive, empty explanation/map, entries without `label` key
- **`TestLegendGrounderTool`** — `pop_traces` flush-and-clear, unknown backend returns error JSON, Gemini success path with trace appended, API crash fallback, prompt template structure
