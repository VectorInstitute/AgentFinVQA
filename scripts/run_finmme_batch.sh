#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
ENV_FILE_DEFAULT="$REPO_ROOT/.env"
LOG_DIR_DEFAULT="$REPO_ROOT/logs"

LOG_DIR="${LOG_DIR:-$LOG_DIR_DEFAULT}"
mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/finmme_${timestamp}.log"

export CREWAI_TRACING_ENABLED=false
export OTEL_SDK_DISABLED=1

{
  echo "[$(date --iso-8601=seconds)] Starting FinMME batch → log $LOG_FILE"
  python3 "$REPO_ROOT/scripts/run_finmme_batch.py" --env_file "${ENV_FILE:-$ENV_FILE_DEFAULT}" "$@"
  status=$?
  echo "[$(date --iso-8601=seconds)] Run completed with status $status (log $LOG_FILE)"
  exit $status
} 2>&1 | tee "$LOG_FILE"
