#!/usr/bin/env bash
# Step 01: Run SCIP with full strong branching on training instances to collect
# telemetry (node features + SB scores) as NDJSON.
#
# Instance families (Gasse et al. 2019 protocol + VRP-MIP domain set):
#   sc_train.txt   -- Set Covering
#   ca_train.txt   -- Combinatorial Auctions
#   cfl_train.txt  -- Capacitated Facility Location
#   mis_train.txt  -- Maximum Independent Set
#   vrp_train.txt  -- VRP-MIP (optional domain set)
#
# Each file should contain one .lp / .mps path per line.
# Generate instances first: bash benchmarks/pipeline/00_generate.sh
#
# Output: $DATA_DIR/logs/{family}/{instance_id}_s{seed}.ndjson
#
# Requires:
#   $SCIP_BIN      - SCIP binary with BBML plugin compiled in
#   $BBML_ROOT     - project root
#   $DATA_DIR      - output data directory (default: $BBML_ROOT/data)
set -euo pipefail

BBML_ROOT="${BBML_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
SCIP_BIN="${SCIP_BIN:-scip}"
INSTANCES_DIR="$BBML_ROOT/benchmarks/instances"

SEEDS="${COLLECT_SEEDS:-0 1 2}"
TL="${COLLECT_TL:-3600}"
MAX_NODES="${COLLECT_MAX_NODES:-5000}"   # cap nodes to control SB cost

# ---- discover instance lists ----
# Accept an explicit list via INSTANCE_FAMILIES env var, otherwise auto-detect
# all *_train.txt files in the instances directory.
if [ -n "${INSTANCE_FAMILIES:-}" ]; then
  IFS=',' read -ra FAMILY_FILES <<< "$INSTANCE_FAMILIES"
else
  FAMILY_FILES=()
  while IFS= read -r f; do FAMILY_FILES+=("$f"); done \
    < <(find "$INSTANCES_DIR" -maxdepth 1 -name '*_train.txt' | sort)
fi

if [ ${#FAMILY_FILES[@]} -eq 0 ]; then
  echo "ERROR: no instance list files found in $INSTANCES_DIR"
  echo ""
  echo "Expected files (one .lp/.mps path per line):"
  echo "  sc_train.txt   -- Set Covering       (10 000 instances)"
  echo "  ca_train.txt   -- Combinatorial Auctions"
  echo "  cfl_train.txt  -- Capacitated Facility Location"
  echo "  mis_train.txt  -- Maximum Independent Set"
  echo "  vrp_train.txt  -- VRP-MIP (optional)"
  echo ""
  echo "Generate synthetic families with:"
  echo "  bash benchmarks/pipeline/00_generate.sh"
  echo ""
  echo "Or point to existing instances:"
  echo "  ls /path/to/sc/*.lp > benchmarks/instances/sc_train.txt"
  exit 1
fi

echo "=== SB telemetry collection ==="
echo "  Families   : ${#FAMILY_FILES[@]}"
echo "  Seeds      : $SEEDS"
echo "  Max nodes  : $MAX_NODES"
echo "  Time limit : ${TL}s"
echo ""

total_new=0; total_skip=0; total_fail=0

for list_file in "${FAMILY_FILES[@]}"; do
  family=$(basename "$list_file" _train.txt)
  log_dir="$DATA_DIR/logs/$family"
  mkdir -p "$log_dir"

  instances=()
  while IFS= read -r line; do instances+=("$line"); done < "$list_file"
  # Remove blank lines
  instances=("${instances[@]/#$'\n'/}")
  n_inst=0
  for inst in "${instances[@]}"; do [ -n "$inst" ] && n_inst=$((n_inst+1)); done

  echo "--- Family: $family ($n_inst instances) ---"

  for inst in "${instances[@]}"; do
    [ -z "$inst" ] && continue
    id=$(basename "$inst" | sed 's/\.\(lp\|mps\|gz\)$//;s/\.mps\.gz$//')
    for seed in $SEEDS; do
      out="$log_dir/${id}_s${seed}.ndjson"
      if [ -f "$out" ] && [ -s "$out" ]; then
        total_skip=$((total_skip+1))
        continue
      fi
      echo "  $family/$id  seed=$seed"
      "$SCIP_BIN" \
        -s "$BBML_ROOT/configs/solver/integration.yaml" \
        -c "set branching/fullstrong/priority 1000000" \
        -c "set limits/time $TL" \
        -c "set limits/nodes $MAX_NODES" \
        -c "set randomization/randomseedshift $seed" \
        -c "set bbml/telemetry/logfile $out" \
        -c "set bbml/telemetry/enabled TRUE" \
        -c "set bbml/telemetry/strongbranch TRUE" \
        -c "read $inst" \
        -c "optimize" \
        -c "quit" \
        > "$log_dir/${id}_s${seed}.scip.log" 2>&1 \
        && total_new=$((total_new+1)) \
        || { echo "    WARNING: SCIP failed (check $log_dir/${id}_s${seed}.scip.log)"; total_fail=$((total_fail+1)); }
    done
  done
done

echo ""
echo "Collection complete."
echo "  Collected : $total_new   Skipped (cached): $total_skip   Failed: $total_fail"
echo "  Output    : $DATA_DIR/logs/{family}/"
