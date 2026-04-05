#!/usr/bin/env bash
# Submit a zero-shot baseline SLURM job.
#
# Usage:
#   baselines/submit_zeroshot.sh --dataset chartqapro --backend gemini --model gemini-2.5-flash
#   baselines/submit_zeroshot.sh --dataset finmme --backend openai --model gpt-4o --after 12345
#
# Options:
#   --dataset   DATASET   chartqapro | finmme  (default: chartqapro)
#   --split     SPLIT     test | train | ...   (default: test)
#   --n         N         samples, 0=all       (default: 0)
#   --backend   BACKEND   gemini | openai      (default: gemini)
#   --model     MODEL     model name           (default: gemini-2.5-flash)
#   --workers   N         parallel workers     (default: 8)
#   --resume              skip done samples
#   --out       DIR       output dir           (default: output/baselines/)
#   --prompt_style STYLE  structured (default) | minimal — minimal = bare Q/A, no JSON rules
#   --after     JOB_ID    hold until job completes
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="chartqapro"
SPLIT="test"
N=0
BACKEND="gemini"
MODEL="gemini-2.5-flash"
WORKERS=8
RESUME=0
OUT="output/baselines/"
AFTER_JOB=""
PROMPT_STYLE="structured"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";  shift 2 ;;
        --split)    SPLIT="$2";    shift 2 ;;
        --n)        N="$2";        shift 2 ;;
        --backend)  BACKEND="$2";  shift 2 ;;
        --model)    MODEL="$2";    shift 2 ;;
        --workers)  WORKERS="$2";  shift 2 ;;
        --resume)        RESUME=1;              shift ;;
        --out)           OUT="$2";             shift 2 ;;
        --after)         AFTER_JOB="$2";       shift 2 ;;
        --prompt_style)  PROMPT_STYLE="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export DATASET SPLIT N BACKEND MODEL WORKERS RESUME OUT PROMPT_STYLE

echo "=== Submitting zero-shot baseline ==="
echo "  Dataset  : $DATASET / $SPLIT  (n=$N)"
echo "  Model    : $BACKEND / $MODEL"
echo "  Workers  : $WORKERS"
echo "  Prompt   : $PROMPT_STYLE"
echo "  Output   : $OUT"
[[ -n "$AFTER_JOB" ]] && echo "  After    : $AFTER_JOB"
echo ""

DEP=""
[[ -n "$AFTER_JOB" ]] && DEP="--dependency=afterok:${AFTER_JOB}"

JOB=$(sbatch --parsable --export=ALL \
    --job-name="zeroshot_${BACKEND}_${DATASET}" \
    $DEP \
    "$REPO_ROOT/baselines/slurm_zeroshot.slrm")
echo "Submitted job: $JOB"
