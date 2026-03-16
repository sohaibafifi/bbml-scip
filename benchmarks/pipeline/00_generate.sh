#!/usr/bin/env bash
# Step 00: Generate synthetic benchmark instances following Gasse et al. 2019.
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

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
INSTANCE_COMPRESSION="${GENERATE_INSTANCE_COMPRESSION:-gzip}"
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --families=*) FAMILIES="${arg#*=}" ;;
    --dry-run) DRY_RUN=true ;;
  esac
done

IFS=',' read -ra FAM_LIST <<< "$FAMILIES"
PY_LAUNCH_JSON="$(python3 - "${PYTHON_CMD[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"
if [ -z "$PY_LAUNCH_JSON" ]; then
  echo "ERROR: failed to resolve Python launcher command." >&2
  exit 1
fi

echo "=== Instance generation ==="
echo "  Families            : $FAMILIES"
echo "  Counts              : train=$N_TRAIN val=$N_VAL test=$N_TEST"
echo "  Generate jobs       : $GENERATE_JOBS"
echo "  Generate chunk size : $GENERATE_CHUNK_SIZE"
echo "  Compression         : $INSTANCE_COMPRESSION"
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
      existing="$(
        OUT_DIR="$out_dir" INSTANCE_COMPRESSION="$INSTANCE_COMPRESSION" python3 - <<'PY'
from pathlib import Path
import gzip
import os
import shutil

out_dir = Path(os.environ["OUT_DIR"])
compression = os.environ["INSTANCE_COMPRESSION"]
seen = set()
for path in out_dir.iterdir():
    if not path.is_file():
        continue
    name = path.name
    if compression == "gzip" and name.endswith(".lp"):
        gzip_path = path.with_name(f"{name}.gz")
        if not gzip_path.exists():
            with path.open("rb") as in_fh, gzip.open(gzip_path, "wb") as out_fh:
                shutil.copyfileobj(in_fh, out_fh)
            path.unlink()
        path = gzip_path
        name = path.name
    if name.endswith(".lp.gz"):
        seen.add(name[:-3])
    elif name.endswith(".lp"):
        seen.add(name)
print(len(seen))
PY
      )"
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
      INSTANCE_COMPRESSION="$INSTANCE_COMPRESSION" \
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
instance_compression = os.environ["INSTANCE_COMPRESSION"]
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
            "--compression",
            instance_compression,
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
        "${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$manifest" --jobs "$GENERATE_JOBS"
      fi
    fi

    OUT_DIR="$out_dir" LIST_FILE="$list_file" INSTANCE_COMPRESSION="$INSTANCE_COMPRESSION" python3 - <<'PY'
from pathlib import Path
import gzip
import os
import shutil

out_dir = Path(os.environ["OUT_DIR"])
list_file = Path(os.environ["LIST_FILE"])
compression = os.environ["INSTANCE_COMPRESSION"]
selected = {}
for path in sorted(out_dir.iterdir()):
    if not path.is_file():
        continue
    name = path.name
    if compression == "gzip" and name.endswith(".lp"):
        gzip_path = path.with_name(f"{name}.gz")
        if not gzip_path.exists():
            with path.open("rb") as in_fh, gzip.open(gzip_path, "wb") as out_fh:
                shutil.copyfileobj(in_fh, out_fh)
            path.unlink()
        path = gzip_path
        name = path.name
    if name.endswith(".lp.gz"):
        key = name[:-3]
        priority = 0
    elif name.endswith(".lp"):
        key = name
        priority = 1
    else:
        continue
    prev = selected.get(key)
    if prev is None or priority < prev[0]:
        selected[key] = (priority, path.resolve())
entries = [str(path) for _, path in sorted(selected.values(), key=lambda item: str(item[1]))]
list_file.write_text("".join(f"{entry}\n" for entry in entries))
print(len(entries))
PY
    actual="$(tail -n 1 "$list_file" >/dev/null 2>&1; wc -l < "$list_file" | tr -d ' ')"
    echo "  $split: list written ($actual paths) -> $list_file"
  done
  echo ""
done

echo "Generation complete."
