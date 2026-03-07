#!/usr/bin/env bash
# Step 00: Generate synthetic benchmark instances following Gasse et al. 2019.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
INST_DIR="$BBML_ROOT/benchmarks/instances"
SCRIPT="$SCRIPT_DIR/generate_instances.py"

FAMILIES="${GENERATE_FAMILIES:-sc,ca,cfl,mis}"
N_TRAIN="${N_TRAIN:-10000}"
N_VAL="${N_VAL:-2000}"
N_TEST="${N_TEST:-2000}"
GENERATE_JOBS="${GENERATE_JOBS:-$(bbml_default_generate_jobs)}"
GENERATE_CHUNK_SIZE="${GENERATE_CHUNK_SIZE:-500}"
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --families=*) FAMILIES="${arg#*=}" ;;
    --dry-run) DRY_RUN=true ;;
  esac
done

IFS=',' read -ra FAM_LIST <<< "$FAMILIES"
PY_LAUNCH_JSON="$(bbml_python_json_array)"

echo "=== Instance generation ==="
echo "  Families            : $FAMILIES"
echo "  Counts              : train=$N_TRAIN val=$N_VAL test=$N_TEST"
echo "  Generate jobs       : $GENERATE_JOBS"
echo "  Generate chunk size : $GENERATE_CHUNK_SIZE"
echo "  Output              : $DATA_DIR/instances"
echo ""

if $DRY_RUN; then
  echo "[dry-run] Would generate instances for: $FAMILIES"
  exit 0
fi

mkdir -p "$INST_DIR" "$DATA_DIR/manifests" "$DATA_DIR/pipeline_logs/generate"

TRAIN_OFFSET=0
VAL_OFFSET=100000
TEST_OFFSET=200000

for family in "${FAM_LIST[@]}"; do
  echo "--- $family ---"
  for split in train val test; do
    case $split in
      train) n=$N_TRAIN; offset=$TRAIN_OFFSET ;;
      val) n=$N_VAL; offset=$VAL_OFFSET ;;
      test) n=$N_TEST; offset=$TEST_OFFSET ;;
    esac

    out_dir="$DATA_DIR/instances/$family/$split"
    list_file="$INST_DIR/${family}_${split}.txt"
    manifest="$DATA_DIR/manifests/generate_${family}_${split}.jsonl"
    mkdir -p "$out_dir"

    existing=0
    if [ -d "$out_dir" ]; then
      existing="$(find "$out_dir" -name '*.lp' | wc -l | tr -d ' ')"
    fi

    if [ "$existing" -ge "$n" ]; then
      echo "  $split: $existing/$n already exist"
    else
      echo "  $split: scheduling $((n - existing)) instances"
      FAMILY="$family" \
      SPLIT="$split" \
      OUT_DIR="$out_dir" \
      MANIFEST="$manifest" \
      SCRIPT="$SCRIPT" \
      PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
      EXISTING="$existing" \
      TARGET_COUNT="$n" \
      SEED_OFFSET="$offset" \
      CHUNK_SIZE="$GENERATE_CHUNK_SIZE" \
      LOG_DIR="$DATA_DIR/pipeline_logs/generate" \
      python3 - <<'PY'
import json
import os
from pathlib import Path

family = os.environ["FAMILY"]
split = os.environ["SPLIT"]
out_dir = os.environ["OUT_DIR"]
manifest = Path(os.environ["MANIFEST"])
script = os.environ["SCRIPT"]
py_launch = json.loads(os.environ["PY_LAUNCH_JSON"])
existing = int(os.environ["EXISTING"])
target = int(os.environ["TARGET_COUNT"])
seed_offset = int(os.environ["SEED_OFFSET"])
chunk_size = max(1, int(os.environ["CHUNK_SIZE"]))
log_dir = Path(os.environ["LOG_DIR"])

manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w") as fh:
    for start in range(existing, target, chunk_size):
        count = min(chunk_size, target - start)
        cmd = py_launch + [
            script,
            family,
            out_dir,
            "--start",
            str(start),
            "--count",
            str(count),
            "--seed-offset",
            str(seed_offset),
        ]
        rec = {
            "name": f"generate:{family}:{split}:{start}",
            "cmd": cmd,
            "cwd": os.getcwd(),
            "log_path": str(log_dir / f"{family}_{split}_{start:05d}.log"),
        }
        fh.write(json.dumps(rec) + "\n")
PY
      if [ -s "$manifest" ]; then
        bbml_python "$SCRIPT_DIR/task_runner.py" --manifest "$manifest" --jobs "$GENERATE_JOBS"
      fi
    fi

    find "$out_dir" -name '*.lp' | sort > "$list_file"
    actual="$(wc -l < "$list_file" | tr -d ' ')"
    echo "  $split: list written ($actual paths) -> $list_file"
  done
  echo ""
done

echo "Generation complete."
