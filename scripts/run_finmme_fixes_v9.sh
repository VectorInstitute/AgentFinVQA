#!/usr/bin/env bash
# FinMME train (1,250 samples): Gemini 3 Flash Preview (planner / vision / verifier) + Flash-Lite judge.
# v9 — verifier now sees:
#   - related_sentences (analyst source text from sample metadata) — new in v9
#   - caption reframed from "background" to "cross-check the draft answer" — new in v9
# Same 1,250-ID slice as v8 for a paired comparison.
#
# Output dir (run_tag): meps/gemini_gemini/finmme/fixes_v9_g3flash_related_sents/train/
# Metrics dir         : output/final/finmme_fixes_v9_g3flash_related_sents/
#
# Usage: ./scripts/run_finmme_fixes_v9.sh
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
    --run_tag fixes_v9_g3flash_related_sents \
    --out_label finmme_train_fixes_v9_g3flash_related_sents \
    --langfuse \
    --resume
