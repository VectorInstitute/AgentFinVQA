#!/usr/bin/env bash
# Submit a post-evaluation-only SLURM job on existing MEPs.
#
# Usage:
#   scripts/submit_eval.sh [options]
#
# Options:
#   --dataset DATASET      Dataset (default: finmme)
#   --split SPLIT          Split slice used during generation (default: train)
#   --config CONFIG        Backend config preset (default: gemini_gemini)
#   --workers N            Parallel eval workers (default: 8)
#   --judge_model M        Judge model (default: gemini-2.5-flash-lite)
#   --use_judge            Run inline LLM judge per sample
#   --langfuse             Push eval scores to Langfuse traces
#   --eval_dir DIR         Output directory for artifacts (default: output)
#   --out_label LABEL      Label for output files (default: DATASET_SPLIT)
#   --no_verifier          MEPs live under <dataset>/no_verifier/<split> (match generation)
#   --no_ocr               MEPs live under <dataset>/no_ocr/<split> (match generation)
#   --after JOB_ID         Hold until JOB_ID completes successfully (optional)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

module --force purge
module load slurm/bonecho/25.05.2

# Defaults
DATASET="finmme"
SPLIT="train"
CONFIG="gemini_gemini"
WORKERS=8
JUDGE_MODEL="gemini-2.5-flash-lite"
MEP_START=0
USE_JUDGE=0
LANGFUSE=0
EVAL_DIR="output"
OUT_LABEL=""
AFTER_JOB=""
NO_VERIFIER=0
NO_OCR=0
RUN_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2";   shift 2 ;;
        --split)     SPLIT="$2";     shift 2 ;;
        --config)    CONFIG="$2";    shift 2 ;;
        --workers)     WORKERS="$2";     shift 2 ;;
        --judge_model) JUDGE_MODEL="$2"; shift 2 ;;
        --mep_start)   MEP_START="$2";   shift 2 ;;
        --use_judge)   USE_JUDGE=1;      shift ;;
        --langfuse)  LANGFUSE=1;     shift ;;
        --no_verifier) NO_VERIFIER=1;  shift ;;
        --no_ocr)      NO_OCR=1;       shift ;;
        --run_tag)     RUN_TAG="$2";   shift 2 ;;
        --eval_dir)  EVAL_DIR="$2";  shift 2 ;;
        --out_label) OUT_LABEL="$2"; shift 2 ;;
        --after)     AFTER_JOB="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUT_LABEL="${OUT_LABEL:-${DATASET}_${SPLIT}}"

export DATASET SPLIT CONFIG WORKERS JUDGE_MODEL MEP_START USE_JUDGE LANGFUSE EVAL_DIR OUT_LABEL NO_VERIFIER NO_OCR RUN_TAG

echo "=== Submitting eval-only job ==="
echo "  Dataset      : $DATASET"
echo "  Split        : $SPLIT"
echo "  Config       : $CONFIG"
echo "  Workers      : $WORKERS"
echo "  Judge model  : $JUDGE_MODEL"
echo "  Use judge    : $USE_JUDGE"
echo "  Langfuse     : $LANGFUSE"
echo "  Eval dir     : $EVAL_DIR"
echo "  Output label : $OUT_LABEL"
echo "  No verifier  : $NO_VERIFIER"
echo "  No OCR       : $NO_OCR"
[[ -n "$AFTER_JOB" ]] && echo "  After job    : $AFTER_JOB"
echo ""

JOB_DEP=""
[[ -n "$AFTER_JOB" ]] && JOB_DEP="--dependency=afterok:${AFTER_JOB}"

JOB=$(sbatch --parsable --export=ALL $JOB_DEP scripts/slurm_eval_only.slrm)
echo "Eval job submitted: $JOB"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB"
echo "  tail -f logs/slurm_eval_${JOB}.out"
