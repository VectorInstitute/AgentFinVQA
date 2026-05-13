#!/usr/bin/env bash
# FinMME train (1,250 samples): Gemini 3 Flash Preview (planner / vision / verifier) + Flash-Lite judge.
# v10 — everything in v9, plus verifier **choice-conflict meta-signal**:
#   - related_sentences + caption cross-check (same as v9)
#   - When vision's ``choice_analysis`` rates ≥2 single-select options at confidence ≥0.95,
#     the verifier prompt gets a VISION AMBIGUITY FLAG (labels only — not the full dict).
# Same 1,250-ID slice as v8/v9 for paired comparison.
#
# Output dir (run_tag): meps/gemini_gemini/finmme/fixes_v10_g3flash_choice_conflict/train/
# Metrics dir         : output/final/finmme_fixes_v10_g3flash_choice_conflict/
#
# Usage: ./scripts/run_finmme_fixes_v10.sh
# Remove --resume below for a full re-run from scratch.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

scripts/submit_pipeline.sh \
    --dataset finmme \
    --split train \
    --n 1250 \
    --config gemini_gemini \
    --vision_model gemini-3-flash-preview \
    --planner_model gemini-3-flash-preview \
    --verifier_model gemini-3-flash-preview \
    --judge_model gemini-2.5-flash-lite \
    --run_tag fixes_v10_g3flash_choice_conflict \
    --out_label finmme_train_fixes_v10_g3flash_choice_conflict \
    --langfuse \
    --resume
