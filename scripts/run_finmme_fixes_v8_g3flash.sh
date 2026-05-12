#!/usr/bin/env bash
# FinMME train (1250 samples = original 250 + 1000 new, indices 250..1249):
# Gemini 3 Flash preview (planner / vision / verifier) + Flash-Lite judge.
# v8 — includes color-area (OpenCV) stage; distinct run_tag / out_label vs v7 conf_gate runs.
#
# --n 1250 + --resume: the first 250 samples already have MEPs in
#   meps/gemini_gemini/finmme/fixes_v8_g3flash_color_area/train/ and are skipped;
#   the runner only processes the new samples finmme_000250..finmme_001249.
# After the chained judge batch eval, the metrics file rebuilds with 1250 rows.
# Usage: ./scripts/run_finmme_fixes_v8_g3flash.sh
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
    --run_tag fixes_v8_g3flash_color_area \
    --out_label finmme_train_fixes_v8_g3flash_color_area \
    --langfuse \
    --resume
