#!/usr/bin/env bash
# Step 02: Build manifest files and convert candidate telemetry to Parquet.
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
INSTANCES_DIR="${INSTANCE_LIST_DIR:-$BBML_ROOT/benchmarks/instances}"
LOG_DIR="$DATA_DIR/logs"
PARQUET_DIR="$DATA_DIR/parquet"
MANIFEST_DIR="$DATA_DIR/manifests"
CONVERT_SPLITS="${CONVERT_SPLITS:-train,val}"
CONVERT_JOBS="${CONVERT_JOBS:-$(bbml_default_generate_jobs)}"
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

if ! CONVERTER_CHECK="$("${PYTHON_CMD[@]}" - <<'PY' 2>&1
import importlib.util

missing = [name for name in ("pandas", "pyarrow") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing Python modules: " + ", ".join(missing))
print("ok")
PY
)"; then
  echo "ERROR: telemetry conversion requires pandas and pyarrow in the resolved Python environment." >&2
  echo "Resolved launcher: ${PYTHON_CMD[*]}" >&2
  printf '%s\n' "$CONVERTER_CHECK" >&2
  echo "Set BBML_PYTHON to a Python with the project deps installed, or install them into py/.venv." >&2
  exit 1
fi

IFS=',' read -ra SPLIT_LIST <<< "$CONVERT_SPLITS"

families=()
while IFS= read -r list_file; do
  families+=("$(basename "$list_file" | sed 's/_train\.txt$//')")
done < <(find "$INSTANCES_DIR" -maxdepth 1 -name '*_train.txt' | sort)

if [ ${#families[@]} -eq 0 ]; then
  echo "WARNING: no *_train.txt files in $INSTANCES_DIR; nothing to convert."
  exit 0
fi

mkdir -p "$PARQUET_DIR" "$MANIFEST_DIR/candidates" "$MANIFEST_DIR/graph" "$DATA_DIR/pipeline_logs/convert"

echo "=== Telemetry conversion ==="
echo "  Families      : ${families[*]}"
echo "  Splits        : ${SPLIT_LIST[*]}"
echo "  Convert jobs  : $CONVERT_JOBS"
echo ""

for split in "${SPLIT_LIST[@]}"; do
  : > "$MANIFEST_DIR/candidates/${split}.txt"
  : > "$MANIFEST_DIR/graph/${split}.txt"
done

for family in "${families[@]}"; do
  for split in "${SPLIT_LIST[@]}"; do
    list_file="$INSTANCES_DIR/${family}_${split}.txt"
    [ ! -f "$list_file" ] && continue
    candidate_manifest="$MANIFEST_DIR/candidates/${family}_${split}.txt"
    graph_manifest="$MANIFEST_DIR/graph/${family}_${split}.txt"
    FAMILY="$family" \
    SPLIT="$split" \
    LIST_FILE="$list_file" \
    DATA_DIR="$DATA_DIR" \
    CANDIDATE_MANIFEST="$candidate_manifest" \
    GRAPH_MANIFEST="$graph_manifest" \
    python3 - <<'PY'
from pathlib import Path
import os

def instance_id(path: Path) -> str:
    name = path.name
    for suffix in (".lp.gz", ".mps.gz", ".lp", ".mps", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name

family = os.environ["FAMILY"]
split = os.environ["SPLIT"]
list_file = Path(os.environ["LIST_FILE"])
data_dir = Path(os.environ["DATA_DIR"])
candidate_manifest = Path(os.environ["CANDIDATE_MANIFEST"])
graph_manifest = Path(os.environ["GRAPH_MANIFEST"])

candidate_dir = data_dir / "logs" / family / split / "candidates"
graph_dir = data_dir / "logs" / family / split / "graph"

candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
graph_manifest.parent.mkdir(parents=True, exist_ok=True)

candidate_paths = []
graph_paths = []
for line in list_file.read_text().splitlines():
    inst = line.strip()
    if not inst:
        continue
    iid = instance_id(Path(inst))
    candidate_compact = sorted(candidate_dir.glob(f"{iid}_s*.ndjson.gz"))
    candidate_legacy = sorted(candidate_dir.glob(f"{iid}_s*.ndjson"))
    graph_compact = sorted(graph_dir.glob(f"{iid}_s*.pt"))
    graph_legacy = sorted(graph_dir.glob(f"{iid}_s*.ndjson"))
    candidate_paths.extend(candidate_compact or candidate_legacy)
    graph_paths.extend(graph_compact or graph_legacy)

candidate_manifest.write_text(
    "".join(f"{path.resolve()}\n" for path in candidate_paths if path.is_file() and path.stat().st_size > 0)
)
graph_manifest.write_text(
    "".join(f"{path.resolve()}\n" for path in graph_paths if path.is_file() and path.stat().st_size > 0)
)
PY
    cat "$candidate_manifest" >> "$MANIFEST_DIR/candidates/${split}.txt"
    cat "$graph_manifest" >> "$MANIFEST_DIR/graph/${split}.txt"
  done
done

convert_manifest="$MANIFEST_DIR/convert_tasks.jsonl"
PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
MANIFEST_DIR="$MANIFEST_DIR" \
PARQUET_DIR="$PARQUET_DIR" \
DATA_DIR="$DATA_DIR" \
SCRIPT="$BBML_ROOT/py/bbml/data/json_to_parquet.py" \
python3 - <<'PY'
import json
import os
from pathlib import Path

manifest_dir = Path(os.environ["MANIFEST_DIR"])
parquet_dir = Path(os.environ["PARQUET_DIR"])
data_dir = Path(os.environ["DATA_DIR"])
script = os.environ["SCRIPT"]
py_launch = json.loads(os.environ["PY_LAUNCH_JSON"])
task_manifest = manifest_dir / "convert_tasks.jsonl"

def candidate_manifests():
    for manifest in sorted((manifest_dir / "candidates").glob("*.txt")):
        if manifest.name in {"train.txt", "val.txt"}:
            out = parquet_dir / manifest.stem
            out = out.with_suffix(".parquet")
        else:
            family, split = manifest.stem.rsplit("_", 1)
            out = parquet_dir / family / f"{split}.parquet"
        yield manifest, out

with task_manifest.open("w") as out_fh:
    for manifest, out_path in candidate_manifests():
        lines = [line for line in manifest.read_text().splitlines() if line.strip()]
        skip = out_path.is_file() and out_path.stat().st_size > 0
        rec = {
            "name": f"convert:{manifest.stem}",
            "cmd": py_launch
            + [
                script,
                "--manifest",
                str(manifest),
                "--out",
                str(out_path),
                "--chunksize",
                "200000",
                "--row-group-size",
                "100000",
                "--compression",
                "snappy",
            ],
            "cwd": os.getcwd(),
            "log_path": str(data_dir / "pipeline_logs" / "convert" / f"{manifest.stem}.log"),
            "skip": skip or not lines,
        }
        out_fh.write(json.dumps(rec) + "\n")
PY

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$convert_manifest" --jobs "$CONVERT_JOBS"

echo ""
echo "Candidate parquet written to $PARQUET_DIR"
echo "Graph manifests written to $MANIFEST_DIR/graph"
