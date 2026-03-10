#!/usr/bin/env bash
# Run a staged learn2branch-style protocol for one family.
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python

STAGE="pilot"
FAMILY=""
STEP="prepare"

for arg in "$@"; do
  case $arg in
    --stage=*) STAGE="${arg#*=}" ;;
    --family=*) FAMILY="${arg#*=}" ;;
    --step=*) STEP="${arg#*=}" ;;
    *)
      echo "ERROR: unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

if [ "$STAGE" = "pilote" ]; then
  STAGE="pilot"
fi

case "$STAGE" in
  pilot|dev|final) ;;
  *)
    echo "ERROR: --stage must be one of: pilot, dev, final" >&2
    exit 1
    ;;
esac

case "$FAMILY" in
  sc|ca|cfl|mis) ;;
  *)
    echo "ERROR: --family must be one of: sc, ca, cfl, mis" >&2
    exit 1
    ;;
esac

case "$STEP" in
  prepare|collect|convert|train|export|bench|eval|all) ;;
  *)
    echo "ERROR: --step must be one of: prepare, collect, convert, train, export, bench, eval, all" >&2
    exit 1
    ;;
esac

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data/learn2branch/$STAGE/$FAMILY}"
RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results/learn2branch/$STAGE/$FAMILY}"
INSTANCE_LIST_DIR="${INSTANCE_LIST_DIR:-$BBML_ROOT/benchmarks/instances/learn2branch/$STAGE/$FAMILY}"
BENCH_INSTANCE_SETS_DEFAULT="${FAMILY}_easy_test,${FAMILY}_medium_test,${FAMILY}_hard_test"
BENCH_METHODS_DEFAULT="${BENCH_METHODS:-scip-default,strong-branch,bbml-gnn-graph,bbml-mlp}"

stage_values="$("${PYTHON_CMD[@]}" - "$STAGE" <<'PY'
import sys

stage = sys.argv[1]
cfg = {
    "pilot": {
        "bench_seeds": "0",
        "collect_tl": "300",
        "collect_max_nodes": "500",
        "run_jobs": "4",
    },
    "dev": {
        "bench_seeds": "0 1",
        "collect_tl": "600",
        "collect_max_nodes": "1000",
        "run_jobs": "4",
    },
    "final": {
        "bench_seeds": "0 1 2 3 4",
        "collect_tl": "1200",
        "collect_max_nodes": "5000",
        "run_jobs": "4",
    },
}[stage]
print(cfg["bench_seeds"])
print(cfg["collect_tl"])
print(cfg["collect_max_nodes"])
print(cfg["run_jobs"])
PY
)"
BENCH_SEEDS_DEFAULT="$(printf '%s\n' "$stage_values" | sed -n '1p')"
COLLECT_TL_DEFAULT="$(printf '%s\n' "$stage_values" | sed -n '2p')"
COLLECT_MAX_NODES_DEFAULT="$(printf '%s\n' "$stage_values" | sed -n '3p')"
RUN_JOBS_DEFAULT="$(printf '%s\n' "$stage_values" | sed -n '4p')"

TRAIN_DEVICE="${TRAIN_DEVICE:-$(bbml_detect_torch_device)}"
CALIBRATE_DEVICE="${CALIBRATE_DEVICE:-$(bbml_detect_torch_device)}"
COLLECT_JOBS="${COLLECT_JOBS:-$(bbml_default_solver_jobs)}"
CONVERT_JOBS="${CONVERT_JOBS:-$(bbml_default_generate_jobs)}"
RUN_JOBS="${RUN_JOBS:-$RUN_JOBS_DEFAULT}"

echo "=== Learn2Branch Stage ==="
echo "  Stage         : $STAGE"
echo "  Family        : $FAMILY"
echo "  Step          : $STEP"
echo "  Data dir      : $DATA_DIR"
echo "  Results dir   : $RESULTS_DIR"
echo "  Instance dir  : $INSTANCE_LIST_DIR"
echo "  Train device  : $TRAIN_DEVICE"
echo "  Calib device  : $CALIBRATE_DEVICE"
echo ""

run_prepare() {
  mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$INSTANCE_LIST_DIR"
  "${PYTHON_CMD[@]}" "$SCRIPT_DIR/prepare_learn2branch_stage.py" \
    --stage "$STAGE" \
    --family "$FAMILY" \
    --data-dir "$DATA_DIR" \
    --instances-dir "$INSTANCE_LIST_DIR"
}

run_collect() {
  DATA_DIR="$DATA_DIR" \
  INSTANCE_LIST_DIR="$INSTANCE_LIST_DIR" \
  INSTANCE_FAMILIES="$FAMILY" \
  COLLECT_SEEDS="${COLLECT_SEEDS:-0}" \
  COLLECT_TL="${COLLECT_TL:-$COLLECT_TL_DEFAULT}" \
  COLLECT_MAX_NODES="${COLLECT_MAX_NODES:-$COLLECT_MAX_NODES_DEFAULT}" \
  COLLECT_STRONGBRANCH="${COLLECT_STRONGBRANCH:-0}" \
  COLLECT_JOBS="$COLLECT_JOBS" \
  COLLECT_SPLITS="${COLLECT_SPLITS:-train,val}" \
  bash "$SCRIPT_DIR/01_collect.sh"
}

run_convert() {
  DATA_DIR="$DATA_DIR" \
  INSTANCE_LIST_DIR="$INSTANCE_LIST_DIR" \
  CONVERT_SPLITS="${CONVERT_SPLITS:-train,val}" \
  CONVERT_JOBS="$CONVERT_JOBS" \
  bash "$SCRIPT_DIR/02_convert.sh"
}

run_train() {
  DATA_DIR="$DATA_DIR" \
  RESULTS_DIR="$RESULTS_DIR" \
  TRAIN_DEVICE="$TRAIN_DEVICE" \
  bash "$SCRIPT_DIR/03_train.sh"
}

run_export() {
  DATA_DIR="$DATA_DIR" \
  RESULTS_DIR="$RESULTS_DIR" \
  CALIBRATE_DEVICE="$CALIBRATE_DEVICE" \
  bash "$SCRIPT_DIR/04_calibrate_export.sh"
}

run_bench() {
  DATA_DIR="$DATA_DIR" \
  RESULTS_DIR="$RESULTS_DIR" \
  INSTANCE_LIST_DIR="$INSTANCE_LIST_DIR" \
  BENCH_TL="${BENCH_TL:-3600}" \
  BENCH_SEEDS="${BENCH_SEEDS:-$BENCH_SEEDS_DEFAULT}" \
  BENCH_INSTANCE_SETS="${BENCH_INSTANCE_SETS:-$BENCH_INSTANCE_SETS_DEFAULT}" \
  BENCH_METHODS="$BENCH_METHODS_DEFAULT" \
  RUN_JOBS="$RUN_JOBS" \
  BENCH_ROOT_CUTS_ONLY="${BENCH_ROOT_CUTS_ONLY:-1}" \
  BENCH_DISABLE_RESTARTS="${BENCH_DISABLE_RESTARTS:-1}" \
  bash "$SCRIPT_DIR/05_run.sh"
}

run_eval() {
  mkdir -p "$RESULTS_DIR/tables" "$RESULTS_DIR/figures"
  "${PYTHON_CMD[@]}" "$BBML_ROOT/benchmarks/eval/kpis.py" \
    --results "$RESULTS_DIR/runs" \
    --instance-sets "$INSTANCE_LIST_DIR" \
    --baseline scip-default \
    --out "$RESULTS_DIR/kpis.csv"
  "${PYTHON_CMD[@]}" "$BBML_ROOT/benchmarks/eval/summary_table.py" \
    --kpis "$RESULTS_DIR/kpis.csv" \
    --out "$RESULTS_DIR/tables" || true
  "${PYTHON_CMD[@]}" "$BBML_ROOT/benchmarks/eval/perf_profile.py" \
    --results "$RESULTS_DIR/runs" \
    --out "$RESULTS_DIR/figures" || true
}

case "$STEP" in
  prepare) run_prepare ;;
  collect) run_collect ;;
  convert) run_convert ;;
  train) run_train ;;
  export) run_export ;;
  bench) run_bench ;;
  eval) run_eval ;;
  all)
    run_prepare
    run_collect
    run_convert
    run_train
    run_export
    run_bench
    run_eval
    ;;
esac
