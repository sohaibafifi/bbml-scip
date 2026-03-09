#!/usr/bin/env bash
# Master pipeline: generate, collect, convert, train, export, and run supported benchmarks.
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_runner

export BBML_ROOT
export RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
export DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
export GENERATE_JOBS="${GENERATE_JOBS:-$(bbml_default_generate_jobs)}"
export GENERATE_CHUNK_SIZE="${GENERATE_CHUNK_SIZE:-500}"
export COLLECT_JOBS="${COLLECT_JOBS:-$(bbml_default_solver_jobs)}"
export CONVERT_JOBS="${CONVERT_JOBS:-$(bbml_default_generate_jobs)}"
export RUN_JOBS="${RUN_JOBS:-$(bbml_default_solver_jobs)}"
export EXPORT_JOBS="${EXPORT_JOBS:-2}"

SKIP_GENERATE=false
SKIP_COLLECT=false
SKIP_TRAIN=false

for arg in "$@"; do
  case $arg in
    --skip-generate) SKIP_GENERATE=true ;;
    --skip-collect) SKIP_COLLECT=true ;;
    --skip-train) SKIP_TRAIN=true ;;
  esac
done

echo "=== BBML Benchmark Pipeline ==="
echo "  BBML_ROOT          : $BBML_ROOT"
echo "  RESULTS_DIR        : $RESULTS_DIR"
echo "  DATA_DIR           : $DATA_DIR"
echo "  BBML_RUNNER        : ${BBML_RUNNER_BIN:-<not found>}"
echo "  GENERATE_JOBS      : $GENERATE_JOBS"
echo "  GENERATE_CHUNK_SIZE: $GENERATE_CHUNK_SIZE"
echo "  COLLECT_JOBS       : $COLLECT_JOBS"
echo "  CONVERT_JOBS       : $CONVERT_JOBS"
echo "  EXPORT_JOBS        : $EXPORT_JOBS"
echo "  RUN_JOBS           : $RUN_JOBS"
echo ""

if ! $SKIP_GENERATE; then
  bash "$SCRIPT_DIR/00_generate.sh"
  echo ""
fi

if ! $SKIP_COLLECT; then
  bash "$SCRIPT_DIR/01_collect.sh"
  echo ""
  bash "$SCRIPT_DIR/02_convert.sh"
  echo ""
else
  echo "[skip] collection and conversion"
fi

if ! $SKIP_TRAIN; then
  bash "$SCRIPT_DIR/03_train.sh"
  echo ""
  bash "$SCRIPT_DIR/04_calibrate_export.sh"
  echo ""
else
  echo "[skip] training and export"
fi

bash "$SCRIPT_DIR/05_run.sh"
echo ""
echo "Pipeline complete. Results in $RESULTS_DIR"
