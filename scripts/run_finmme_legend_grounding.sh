#!/usr/bin/env bash
# Run the first 250 FinMME samples with legend grounding enabled.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

scripts/submit_pipeline.sh \
    --dataset        finmme \
    --split          train \
    --n              250 \
    --config         gemini_gemini \
    --planner_model  gemini-2.5-flash \
    --vision_model   gemini-2.5-flash \
    --ocr_model      gemini-2.5-flash-lite \
    --verifier_model gemini-2.5-flash \
    --judge_model    gemini-2.5-flash-lite \
    --workers        8 \
    --langfuse \
    --run_tag        legend_grounding \
    --job_name       finmme_legend_grounding \
    --out_label      finmme_train_legend_grounding
