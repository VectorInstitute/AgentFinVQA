#!/usr/bin/env bash
# Submit MEP generation + batch judge submission as a chained SLURM pipeline.
# Job 2 only runs if Job 1 succeeds (afterok dependency).
#
# Usage:
#   scripts/submit_pipeline.sh [options]
#
# Options:
#   --dataset DATASET      Dataset to run (default: finmme)
#   --split SPLIT          Dataset split slice (default: train[3000:5000])
#   --n N                  Max samples (default: 2000). Use 0 for the full split (after --split slice).
#   --workers N            Parallel workers (default: 8)
#   --planner_model M      Planner model (default: gemini-2.5-flash)
#   --vision_model M       Vision model (default: gemini-2.5-flash)
#   --ocr_model M          OCR model (default: gemini-2.5-flash-lite)
#   --verifier_model M     Verifier model (default: gemini-2.5-flash)
#   --judge_model M        Judge model for batch eval (default: gemini-2.5-flash-lite)
#   --langfuse             Enable Langfuse tracing
#   --resume               Resume from existing MEPs
#   --no_verifier          Skip Pass 2.5 verifier (ablation: planner/vision answer kept)
#   --no_legend_grounding  Skip legend grounding stage (ablation)
#   --out_label LABEL      Batch metrics basename (default: {dataset}_{split}[_no_verifier], split chars sanitized)
#   --config CONFIG        Backend config preset (default: gemini_gemini)
#   --job_name NAME        SLURM job name (default: agentfinvqa_batch)
#   --after JOB_ID         Hold job 1 until JOB_ID completes successfully (optional)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

# Load SLURM module so sbatch is available
module --force purge
module load slurm/bonecho/25.05.2

# Defaults
DATASET="finmme"
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
NO_VERIFIER=0
NO_OCR=0
NO_LEGEND_GROUNDING=0
RUN_TAG=""
OUT_LABEL=""
AFTER_JOB=""
JOB_NAME="agentfinvqa_batch"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";        shift 2 ;;
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
        --no_verifier)   NO_VERIFIER=1;       shift ;;
        --no_ocr)              NO_OCR=1;              shift ;;
        --no_legend_grounding) NO_LEGEND_GROUNDING=1; shift ;;
        --run_tag)             RUN_TAG="$2";          shift 2 ;;
        --out_label)     OUT_LABEL="$2";      shift 2 ;;
        --after)         AFTER_JOB="$2";      shift 2 ;;
        --job_name)      JOB_NAME="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default batch-judge output label: avoids collisions (e.g. metrics_test.jsonl for every dataset).
# MEP dir: meps/${CONFIG}/${DATASET}/${SPLIT} or .../${DATASET}/no_verifier/${SPLIT} if NO_VERIFIER=1.
if [[ -z "${OUT_LABEL}" ]]; then
    SAFE_SPLIT=$(printf '%s' "$SPLIT" | sed 's/[^A-Za-z0-9._-]/_/g' | tr -s '_' | sed 's/^_//' | sed 's/_$//')
    OUT_LABEL="${DATASET}_${SAFE_SPLIT}"
    [[ "$NO_VERIFIER" == "1" ]] && OUT_LABEL="${OUT_LABEL}_no_verifier"
fi

if [[ "$NO_VERIFIER" == "1" ]]; then
    MEP_DIR_PREVIEW="meps/${CONFIG}/${DATASET}/no_verifier/${SPLIT}"
else
    MEP_DIR_PREVIEW="meps/${CONFIG}/${DATASET}/${SPLIT}"
fi

# Export vars so sbatch inherits them (avoids --export string parsing issues
# with special characters like brackets in split names e.g. train[0:1])
export DATASET SPLIT N WORKERS PLANNER_MODEL VISION_MODEL OCR_MODEL VERIFIER_MODEL LANGFUSE RESUME NO_VERIFIER NO_OCR NO_LEGEND_GROUNDING RUN_TAG
export JUDGE_MODEL CONFIG OUT_LABEL JOB_NAME

echo "=== Submitting ${DATASET} pipeline ==="
echo "  Dataset        : $DATASET"
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
echo "  No verifier    : $NO_VERIFIER"
echo "  No OCR         : $NO_OCR"
echo "  No legend grnd : $NO_LEGEND_GROUNDING"
echo "  Run tag        : ${RUN_TAG:-none}"
echo "  MEP directory  : $MEP_DIR_PREVIEW"
echo "  Batch metrics  : output/metrics_${OUT_LABEL}.jsonl"
echo "  Output label   : $OUT_LABEL"
echo "  Job name       : $JOB_NAME"
[[ -n "$AFTER_JOB" ]] && echo "  After job      : $AFTER_JOB"
echo ""

# Job 1: MEP generation (optionally held until AFTER_JOB completes)
JOB1_DEP=""
[[ -n "$AFTER_JOB" ]] && JOB1_DEP="--dependency=afterok:${AFTER_JOB}"
JOB1=$(sbatch --parsable --export=ALL --job-name="${JOB_NAME}" --cpus-per-task="${WORKERS}" $JOB1_DEP scripts/slurm_run_batch.slrm)
echo "Job 1 (generate) submitted: $JOB1"

# Load SLURM module so sbatch is available
module --force purge
module load slurm/bonecho/25.05.2

# Job 2: submit Gemini Batch judge job (runs only if job 1 succeeds)
JOB2=$(sbatch --parsable --export=ALL \
    --job-name="${JOB_NAME}_judge" \
    --dependency=afterok:${JOB1} \
    scripts/slurm_submit_judge_batch.slrm)
echo "Job 2 (judge submit) submitted: $JOB2 (depends on $JOB1)"

echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB1},${JOB2}"
echo "  tail -f logs/slurm_batch_${JOB1}.out"
