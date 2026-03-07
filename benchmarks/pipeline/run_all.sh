#!/usr/bin/env bash
# Master pipeline: runs steps 01-05 end-to-end.
# Usage: bash benchmarks/pipeline/run_all.sh [--skip-collect] [--skip-train]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- configurable via environment ----
export BBML_ROOT
export SCIP_BIN="${SCIP_BIN:-scip}"
export RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
export DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"

SKIP_GENERATE=false
SKIP_COLLECT=false
SKIP_TRAIN=false

for arg in "$@"; do
  case $arg in
    --skip-generate) SKIP_GENERATE=true ;;
    --skip-collect)  SKIP_COLLECT=true ;;
    --skip-train)    SKIP_TRAIN=true ;;
  esac
done

echo "=== BBML Benchmark Pipeline ==="
echo "  BBML_ROOT  : $BBML_ROOT"
echo "  RESULTS_DIR: $RESULTS_DIR"
echo "  DATA_DIR   : $DATA_DIR"
echo "  SCIP_BIN   : $SCIP_BIN"
echo ""

if ! $SKIP_GENERATE; then
  echo "[00/05] Generating instances (Gasse et al. 2019 protocol)..."
  bash "$SCRIPT_DIR/00_generate.sh"
  echo ""
fi

if ! $SKIP_COLLECT; then
  echo "[01/05] Collecting telemetry (strong branching)..."
  bash "$SCRIPT_DIR/01_collect.sh"
  echo ""

  echo "[02/05] Converting NDJSON -> Parquet..."
  bash "$SCRIPT_DIR/02_convert.sh"
  echo ""
else
  echo "[01-02] Skipping data collection."
fi

if ! $SKIP_TRAIN; then
  echo "[03/05] Training GNN and tabular baselines..."
  bash "$SCRIPT_DIR/03_train.sh"
  echo ""

  echo "[04/05] Calibrating temperature and exporting ONNX..."
  bash "$SCRIPT_DIR/04_calibrate_export.sh"
  echo ""
else
  echo "[03-04] Skipping training."
fi

echo "[05/05] Running benchmark (all methods + ablations)..."
bash "$SCRIPT_DIR/05_run.sh"
echo ""

echo "=== Pipeline complete. Results in $RESULTS_DIR ==="
echo "Next: run benchmarks/eval/kpis.py to compute SGM tables."
