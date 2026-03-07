#!/usr/bin/env bash
# Step 02: Convert NDJSON telemetry logs to Parquet, one file per (family, split).
#
# Output: $DATA_DIR/parquet/{family}/{train,val}.parquet
set -euo pipefail

BBML_ROOT="${BBML_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
INSTANCES_DIR="$BBML_ROOT/benchmarks/instances"
LOG_DIR="$DATA_DIR/logs"
PARQUET_DIR="$DATA_DIR/parquet"

PY="uv run --project $BBML_ROOT/py python"

convert_split() {
  local family="$1"
  local split="$2"    # train | val
  local inst_list="$INSTANCES_DIR/${family}_${split}.txt"
  local out_dir="$PARQUET_DIR/$family"
  local out_parquet="$out_dir/${split}.parquet"
  local tmp_merged="$DATA_DIR/logs/${family}_${split}_merged.ndjson"

  [ ! -f "$inst_list" ] && echo "  skip $family/$split (no list)" && return

  mkdir -p "$out_dir"

  if [ -f "$out_parquet" ]; then
    echo "  $family/$split: already exists, skipping."
    return
  fi

  echo "  $family/$split: merging logs..."
  > "$tmp_merged"
  while IFS= read -r inst; do
    [ -z "$inst" ] && continue
    id=$(basename "$inst" | sed 's/\.\(lp\|mps\|gz\)$//;s/\.mps\.gz$//')
    for f in "$LOG_DIR/$family/${id}"_s*.ndjson; do
      [ -f "$f" ] && cat "$f" >> "$tmp_merged"
    done
  done < "$inst_list"

  local n_lines
  n_lines=$(wc -l < "$tmp_merged")
  echo "  $family/$split: $n_lines lines -> $out_parquet"

  $PY -m bbml.data.json_to_parquet \
    --in  "$tmp_merged" \
    --out "$out_parquet" \
    --chunksize 200000 \
    --row-group-size 100000 \
    --compression snappy

  rm -f "$tmp_merged"
}

echo "Converting NDJSON logs to Parquet..."

# Discover families from train list files
families=()
while IFS= read -r f; do
  families+=("$(basename "$f" _train.txt)")
done < <(find "$INSTANCES_DIR" -maxdepth 1 -name '*_train.txt' | sort)

if [ ${#families[@]} -eq 0 ]; then
  echo "WARNING: no *_train.txt files in $INSTANCES_DIR; nothing to convert."
  exit 0
fi

for family in "${families[@]}"; do
  convert_split "$family" "train"
  convert_split "$family" "val"
done

echo ""
echo "Parquet files written to $PARQUET_DIR"
find "$PARQUET_DIR" -name '*.parquet' -exec ls -lh {} \; 2>/dev/null || echo "(no files yet)"
