#!/usr/bin/env bash
# Step 00: Generate synthetic benchmark instances following Gasse et al. 2019.
#
# Families (matching Gasse 2019 Appendix A):
#   sc   -- Set Covering        (1000 rows, 500 cols, density 0.05)
#   ca   -- Combinatorial Auctions (100 items, 500 bids)
#   cfl  -- Capacitated Facility Location (100 customers, 100 facilities)
#   mis  -- Maximum Independent Set (Barabasi-Albert, 500 nodes)
#
# Default counts: 10 000 train / 2 000 val / 2 000 test per family.
# Override via N_TRAIN / N_VAL / N_TEST environment variables.
#
# No ecole required -- uses benchmarks/pipeline/generate_instances.py
# (numpy + networkx only).
#
# Usage:
#   bash benchmarks/pipeline/00_generate.sh [--families sc,ca,cfl,mis] [--dry-run]
set -euo pipefail

BBML_ROOT="${BBML_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
INST_DIR="$BBML_ROOT/benchmarks/instances"
SCRIPT="$BBML_ROOT/benchmarks/pipeline/generate_instances.py"
PY="uv run --project $BBML_ROOT/py python"

FAMILIES="${GENERATE_FAMILIES:-sc,ca,cfl,mis}"
N_TRAIN="${N_TRAIN:-10000}"
N_VAL="${N_VAL:-2000}"
N_TEST="${N_TEST:-2000}"
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --families=*) FAMILIES="${arg#*=}" ;;
    --dry-run)    DRY_RUN=true ;;
  esac
done

IFS=',' read -ra FAM_LIST <<< "$FAMILIES"

echo "=== Instance generation (Gasse et al. 2019 protocol) ==="
echo "  Families : $FAMILIES"
echo "  Counts   : train=$N_TRAIN  val=$N_VAL  test=$N_TEST"
echo "  Output   : $DATA_DIR/instances/"
echo ""

if $DRY_RUN; then
  echo "[dry-run] Would generate instances for: $FAMILIES"
  exit 0
fi

mkdir -p "$INST_DIR"

# Seed offsets per split so instances are always independent across splits
TRAIN_OFFSET=0
VAL_OFFSET=100000
TEST_OFFSET=200000

for family in "${FAM_LIST[@]}"; do
  echo "--- $family ---"
  for split in train val test; do
    case $split in
      train) n=$N_TRAIN; offset=$TRAIN_OFFSET ;;
      val)   n=$N_VAL;   offset=$VAL_OFFSET ;;
      test)  n=$N_TEST;  offset=$TEST_OFFSET ;;
    esac

    out_dir="$DATA_DIR/instances/$family/$split"
    list_file="$INST_DIR/${family}_${split}.txt"

    # Count already-existing LP files
    existing=0
    if [ -d "$out_dir" ]; then
      existing=$(find "$out_dir" -name '*.lp' | wc -l | tr -d ' ')
    fi

    if [ "$existing" -ge "$n" ]; then
      echo "  $split: $existing/$n already exist -- skipping generation"
    else
      to_gen=$((n - existing))
      echo "  $split: generating $to_gen instances (have $existing/$n)..."
      $PY "$SCRIPT" "$family" "$out_dir" \
        --start "$existing" \
        --count "$to_gen" \
        --seed-offset "$offset"
    fi

    # (Re)write the list file
    find "$out_dir" -name '*.lp' | sort > "$list_file"
    actual=$(wc -l < "$list_file" | tr -d ' ')
    echo "  $split: list written ($actual paths) -> $list_file"
  done
  echo ""
done

echo "Generation complete."
echo "Instance lists are in $INST_DIR/"
