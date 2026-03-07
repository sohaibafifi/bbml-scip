#!/usr/bin/env bash
# Step 01: Collect candidate and graph telemetry for train/val splits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python
bbml_require_runner

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
INSTANCES_DIR="$BBML_ROOT/benchmarks/instances"

SEEDS="${COLLECT_SEEDS:-0 1 2}"
TL="${COLLECT_TL:-3600}"
MAX_NODES="${COLLECT_MAX_NODES:-5000}"
COLLECT_SPLITS="${COLLECT_SPLITS:-train,val}"
COLLECT_JOBS="${COLLECT_JOBS:-$(bbml_default_solver_jobs)}"

IFS=',' read -ra SPLIT_LIST <<< "$COLLECT_SPLITS"
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

families=()
if [ -n "${INSTANCE_FAMILIES:-}" ]; then
  IFS=',' read -ra families <<< "$INSTANCE_FAMILIES"
else
  while IFS= read -r list_file; do
    families+=("$(basename "$list_file" | sed 's/_train\.txt$//')")
  done < <(find "$INSTANCES_DIR" -maxdepth 1 -name '*_train.txt' | sort)
fi

if [ ${#families[@]} -eq 0 ]; then
  echo "ERROR: no *_train.txt lists found in $INSTANCES_DIR"
  exit 1
fi

mkdir -p "$DATA_DIR/manifests" "$DATA_DIR/pipeline_logs/collect"

echo "=== Telemetry collection ==="
echo "  Families     : ${families[*]}"
echo "  Splits       : ${SPLIT_LIST[*]}"
echo "  Seeds        : $SEEDS"
echo "  Time limit   : ${TL}s"
echo "  Max nodes    : $MAX_NODES"
echo "  Collect jobs : $COLLECT_JOBS"
echo "  Runner       : $BBML_RUNNER_BIN"
echo ""

for family in "${families[@]}"; do
  for split in "${SPLIT_LIST[@]}"; do
    list_file="$INSTANCES_DIR/${family}_${split}.txt"
    [ ! -f "$list_file" ] && continue
    manifest="$DATA_DIR/manifests/collect_${family}_${split}.jsonl"
    echo "--- $family/$split ---"
    FAMILY="$family" \
    SPLIT="$split" \
    LIST_FILE="$list_file" \
    MANIFEST="$manifest" \
    DATA_DIR="$DATA_DIR" \
    SCRIPT="$SCRIPT_DIR/collect_task.py" \
    PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
    BBML_RUNNER="$BBML_RUNNER_BIN" \
    SEEDS="$SEEDS" \
    TL="$TL" \
    MAX_NODES="$MAX_NODES" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

def instance_id(path: Path) -> str:
    name = path.name
    if name.endswith(".mps.gz"):
        return name[:-7]
    for suffix in (".lp", ".mps", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name

family = os.environ["FAMILY"]
split = os.environ["SPLIT"]
list_file = Path(os.environ["LIST_FILE"])
manifest = Path(os.environ["MANIFEST"])
data_dir = Path(os.environ["DATA_DIR"])
script = os.environ["SCRIPT"]
py_launch = json.loads(os.environ["PY_LAUNCH_JSON"])
runner_bin = os.environ["BBML_RUNNER"]
seeds = [seed for seed in os.environ["SEEDS"].split() if seed]
time_limit = os.environ["TL"]
max_nodes = os.environ["MAX_NODES"]

candidate_dir = data_dir / "logs" / family / split / "candidates"
graph_dir = data_dir / "logs" / family / split / "graph"
scip_dir = data_dir / "logs" / family / split / "scip"
for path in (candidate_dir, graph_dir, scip_dir):
    path.mkdir(parents=True, exist_ok=True)

manifest.parent.mkdir(parents=True, exist_ok=True)
with list_file.open() as src, manifest.open("w") as out:
    for line in src:
        inst = line.strip()
        if not inst:
            continue
        inst_path = Path(inst)
        iid = instance_id(inst_path)
        for seed in seeds:
            candidate_out = candidate_dir / f"{iid}_s{seed}.ndjson"
            graph_out = graph_dir / f"{iid}_s{seed}.ndjson"
            scip_log = scip_dir / f"{iid}_s{seed}.log"
            skip = candidate_out.is_file() and candidate_out.stat().st_size > 0 and graph_out.is_file() and graph_out.stat().st_size > 0
            rec = {
                "name": f"collect:{family}:{split}:{iid}:s{seed}",
                "cmd": py_launch
                + [
                    script,
                    "--runner-bin",
                    runner_bin,
                    "--instance",
                    inst,
                    "--seed",
                    seed,
                    "--time-limit",
                    time_limit,
                    "--max-nodes",
                    max_nodes,
                    "--candidate-out",
                    str(candidate_out),
                    "--graph-out",
                    str(graph_out),
                    "--scip-log",
                    str(scip_log),
                ],
                "cwd": os.getcwd(),
                "log_path": str(data_dir / "pipeline_logs" / "collect" / f"{family}_{split}_{iid}_s{seed}.log"),
                "skip": skip,
            }
            out.write(json.dumps(rec) + "\n")
PY
    "${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$manifest" --jobs "$COLLECT_JOBS"
    echo ""
  done
done

echo "Collection complete."
