#!/usr/bin/env bash
# Step 05: Run all baselines and ablations on all test instance sets.
# Appends one JSON line per run to $RESULTS_DIR/runs/{instance_id}.jsonl
#
# Usage:
#   bash 05_run.sh [--baselines-only] [--ablations-only] [--set vrp|miplib_easy|...]
set -euo pipefail

BBML_ROOT="${BBML_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
MODEL_DIR="$RESULTS_DIR/models"
SCIP_BIN="${SCIP_BIN:-scip}"
TL="${BENCH_TL:-3600}"
SEEDS="${BENCH_SEEDS:-0 1 2 3 4}"

RUN_BASELINES=true
RUN_ABLATIONS=true
INSTANCE_SETS=("sc_test" "ca_test" "cfl_test" "mis_test" "vrp_test")

for arg in "$@"; do
  case $arg in
    --baselines-only) RUN_ABLATIONS=false ;;
    --ablations-only) RUN_BASELINES=false ;;
    --set) shift; INSTANCE_SETS=("$1") ;;
  esac
done

RUNS_DIR="$RESULTS_DIR/runs"
ALPHA_DIR="$RESULTS_DIR/alpha_logs"
mkdir -p "$RUNS_DIR" "$ALPHA_DIR"

# ---- Helper: run one SCIP instance ----
run_scip() {
  local method="$1"
  local inst="$2"
  local seed="$3"
  local ml_flag="$4"   # "true" or "false"
  local model="$5"     # onnx path or ""
  local extra_args="${6:-}"

  local id
  id=$(basename "$inst" | sed 's/\.\(lp\|mps\|gz\)//g')
  local out_jsonl="$RUNS_DIR/${id}.jsonl"
  local alpha_log="$ALPHA_DIR/${id}_${method}_s${seed}.csv"
  local scip_log="$RUNS_DIR/${id}_${method}_s${seed}.log"

  # Skip if already done
  if grep -q "\"solver\":\"$method\",\"seed\":$seed" "$out_jsonl" 2>/dev/null; then
    return 0
  fi

  local cmd_args=()
  cmd_args+=(-c "set limits/time $TL")
  cmd_args+=(-c "set randomization/randomseedshift $seed")
  cmd_args+=(-c "set bbml/telemetry/alpha_logfile $alpha_log")

  if [ "$ml_flag" = "true" ] && [ -n "$model" ]; then
    cmd_args+=(-c "set bbml/branchrule/enabled TRUE")
    cmd_args+=(-c "set bbml/branchrule/model $model")
    [ -f "$MODEL_DIR/temperature.txt" ] && \
      cmd_args+=(-c "set bbml/branchrule/temperature $(cat $MODEL_DIR/temperature.txt)")
  fi

  # Append extra solver settings (from configs/baselines.yaml via caller)
  [ -n "$extra_args" ] && cmd_args+=($extra_args)

  cmd_args+=(-c "read $inst")
  cmd_args+=(-c "optimize")
  cmd_args+=(-c "quit")

  local t0
  t0=$(date +%s%N)
  "$SCIP_BIN" "${cmd_args[@]}" > "$scip_log" 2>&1
  local t1
  t1=$(date +%s%N)
  local elapsed
  elapsed=$(echo "scale=3; ($t1 - $t0) / 1000000000" | bc)

  # Parse SCIP log for KPIs
  local status n_nodes time_solve first_inc root_time
  status=$(grep -oP 'SCIP Status\s*:\s*\K\S+' "$scip_log" | head -1 || echo "unknown")
  n_nodes=$(grep -oP 'nodes.*?:\s*\K[0-9]+' "$scip_log" | head -1 || echo "-1")
  time_solve=$(grep -oP 'Solving Time.*?:\s*\K[0-9.]+' "$scip_log" | head -1 || echo "$elapsed")
  first_inc=$(grep -oP 'First Solution.*?:\s*\K[0-9.]+' "$scip_log" | head -1 || echo "-1")
  root_time=$(grep -oP 'Root Node.*?LP Time.*?:\s*\K[0-9.]+' "$scip_log" | head -1 || echo "-1")

  # Append result
  printf '{"instance_id":"%s","solver":"%s","seed":%s,"status":"%s","solve_time":%s,"n_nodes":%s,"time_to_first_inc":%s,"root_time":%s}\n' \
    "$id" "$method" "$seed" "$status" "$time_solve" "$n_nodes" "$first_inc" "$root_time" \
    >> "$out_jsonl"
}

# ---- Baselines ----
if $RUN_BASELINES; then
  echo "=== Running baselines ==="
  METHODS=(
    "scip-default:false:"
    "strong-branch:false:-c 'set branching/fullstrong/priority 1000000'"
    "pure-imitation:true:$MODEL_DIR/bbml_gnn.onnx"
    "alpha-fixed-0.5:true:$MODEL_DIR/bbml_gnn.onnx"
    "tabular-xgb:true:$MODEL_DIR/bbml_mlp.onnx"
    "bbml-nonode:true:$MODEL_DIR/bbml_gnn.onnx"
    "bbml-full:true:$MODEL_DIR/bbml_gnn.onnx"
  )

  for inst_set in "${INSTANCE_SETS[@]}"; do
    LIST="$BBML_ROOT/benchmarks/instances/${inst_set}.txt"
    [ ! -f "$LIST" ] && echo "  WARNING: $LIST not found; skipping." && continue
    echo "  Instance set: $inst_set"
    instances=()
    while IFS= read -r line; do instances+=("$line"); done < "$LIST"
    for inst in "${instances[@]}"; do
      [ -z "$inst" ] && continue
      for method_str in "${METHODS[@]}"; do
        IFS=':' read -r method ml_flag model <<< "$method_str"
        for seed in $SEEDS; do
          run_scip "$method" "$inst" "$seed" "$ml_flag" "$model" ""
        done
      done
    done
  done
fi

# ---- Ablations (VRP test only to save compute) ----
if $RUN_ABLATIONS; then
  echo "=== Running ablations ==="
  # Run ablations on Set Covering (main Gasse benchmark) to save compute
  ABL_LIST="$BBML_ROOT/benchmarks/instances/sc_test.txt"
  [ ! -f "$ABL_LIST" ] && ABL_LIST="$BBML_ROOT/benchmarks/instances/vrp_test.txt"
  [ ! -f "$ABL_LIST" ] && echo "WARNING: no ablation instance list found; skipping ablations." && exit 0

  instances=()
  while IFS= read -r line; do instances+=("$line"); done < "$ABL_LIST"
  for inst in "${instances[@]}"; do
    [ -z "$inst" ] && continue
    # A2 ablations (blend weight)
    for seed in $SEEDS; do
      run_scip "a2_fixed_00" "$inst" "$seed" "false" "" ""
      run_scip "a2_fixed_10" "$inst" "$seed" "true" "$MODEL_DIR/bbml_gnn.onnx" \
        "-c 'set bbml/branchrule/alpha_max 1.0' -c 'set bbml/branchrule/alpha_min 1.0'"
      run_scip "a2_adaptive" "$inst" "$seed" "true" "$MODEL_DIR/bbml_gnn.onnx" ""
    done
    # A5 ablations (FP16 vs FP32)
    for seed in 0 1; do
      run_scip "a5_fp32" "$inst" "$seed" "true" "$MODEL_DIR/bbml_gnn_fp32.onnx" ""
      run_scip "a5_fp16" "$inst" "$seed" "true" "$MODEL_DIR/bbml_gnn_fp16.onnx" ""
    done
  done
fi

echo ""
echo "=== Runs complete. Results in $RUNS_DIR ==="
