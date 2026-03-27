#!/usr/bin/env bash
# Submit MEP generation + batch judge submission as a chained SLURM pipeline.
# Job 2 only runs if Job 1 succeeds (afterok dependency).
#
# Usage:
#   scripts/submit_pipeline.sh [options]
#
# Options:
#   --split SPLIT          Dataset split slice (default: train[3000:5000])
#   --n N                  Number of samples (default: 2000)
#   --workers N            Parallel workers (default: 8)
#   --planner_model M      Planner model (default: gemini-2.5-flash)
#   --vision_model M       Vision model (default: gemini-2.5-flash)
#   --ocr_model M          OCR model (default: gemini-2.5-flash-lite)
#   --verifier_model M     Verifier model (default: gemini-2.5-flash)
#   --judge_model M        Judge model for batch eval (default: gemini-2.5-flash-lite)
#   --langfuse             Enable Langfuse tracing
#   --resume               Resume from existing MEPs
#   --out_label LABEL      Label for output files (default: same as split)
#   --config CONFIG        Backend config preset (default: gemini_gemini)
#   --after JOB_ID         Hold job 1 until JOB_ID completes successfully (optional)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

# Load SLURM module so sbatch is available
module --force purge
module load slurm/bonecho/25.05.2

# Defaults
SPLIT="train[3000:5000]"
N=2000
WORKERS=8
PLANNER_MODEL="gemini-2.5-flash"
VISION_MODEL="gemini-2.5-flash"
OCR_MODEL="gemini-2.5-flash-lite"
VERIFIER_MODEL="gemini-2.5-flash"
JUDGE_MODEL="gemini-2.5-flash-lite"
CONFIG="gemini_gemini"
LANGFUSE=0
RESUME=0
OUT_LABEL=""
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)         SPLIT="$2";          shift 2 ;;
        --n)             N="$2";              shift 2 ;;
        --workers)       WORKERS="$2";        shift 2 ;;
        --planner_model) PLANNER_MODEL="$2";  shift 2 ;;
        --vision_model)  VISION_MODEL="$2";   shift 2 ;;
        --ocr_model)     OCR_MODEL="$2";      shift 2 ;;
        --verifier_model) VERIFIER_MODEL="$2"; shift 2 ;;
        --judge_model)   JUDGE_MODEL="$2";    shift 2 ;;
        --config)        CONFIG="$2";         shift 2 ;;
        --langfuse)      LANGFUSE=1;          shift ;;
        --resume)        RESUME=1;            shift ;;
        --out_label)     OUT_LABEL="$2";      shift 2 ;;
        --after)         AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUT_LABEL="${OUT_LABEL:-$SPLIT}"

# Export vars so sbatch inherits them (avoids --export string parsing issues
# with special characters like brackets in split names e.g. train[0:1])
export SPLIT N WORKERS PLANNER_MODEL VISION_MODEL OCR_MODEL VERIFIER_MODEL LANGFUSE RESUME
export JUDGE_MODEL CONFIG OUT_LABEL

echo "=== Submitting FinMME pipeline ==="
echo "  Split          : $SPLIT"
echo "  N              : $N"
echo "  Workers        : $WORKERS"
echo "  Planner model  : $PLANNER_MODEL"
echo "  Vision model   : $VISION_MODEL"
echo "  OCR model      : $OCR_MODEL"
echo "  Verifier model : $VERIFIER_MODEL"
echo "  Judge model    : $JUDGE_MODEL"
echo "  Langfuse       : $LANGFUSE"
echo "  Config         : $CONFIG"
echo "  Resume         : $RESUME"
echo "  Output label   : $OUT_LABEL"
[[ -n "$AFTER_JOB" ]] && echo "  After job      : $AFTER_JOB"
echo ""

# Job 1: MEP generation (optionally held until AFTER_JOB completes)
JOB1_DEP=""
[[ -n "$AFTER_JOB" ]] && JOB1_DEP="--dependency=afterok:${AFTER_JOB}"
JOB1=$(sbatch --parsable --export=ALL $JOB1_DEP scripts/slurm_generate_meps.slrm)
echo "Job 1 (generate) submitted: $JOB1"

# Load SLURM module so sbatch is available
module --force purge
module load slurm/bonecho/25.05.2

# Job 2: submit Gemini Batch judge job (runs only if job 1 succeeds)
JOB2=$(sbatch --parsable --export=ALL \
    --dependency=afterok:${JOB1} \
    scripts/slurm_submit_judge_batch.slrm)
echo "Job 2 (judge submit) submitted: $JOB2 (depends on $JOB1)"

echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB1},${JOB2}"
echo "  tail -f logs/slurm_generate_${JOB1}.out"
