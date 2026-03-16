#!/usr/bin/env bash
# Step 01: Collect candidate and graph telemetry for train/val splits.
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python
bbml_require_runner

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
INSTANCES_DIR="${INSTANCE_LIST_DIR:-$BBML_ROOT/benchmarks/instances}"

SEEDS="${COLLECT_SEEDS:-0 1 2}"
TL="${COLLECT_TL:-3600}"
MAX_NODES="${COLLECT_MAX_NODES:-5000}"
TELEMETRY_MAX_NODES_PER_INSTANCE="${COLLECT_TELEMETRY_MAX_NODES_PER_INSTANCE:-0}"
TELEMETRY_QUERY_EXPERT_PROB="${COLLECT_QUERY_EXPERT_PROB:-1.0}"
SAMPLE_BUDGET_ALL="${COLLECT_SAMPLE_BUDGET:-0}"
SAMPLE_BUDGET_TRAIN="${COLLECT_SAMPLE_BUDGET_TRAIN:-$SAMPLE_BUDGET_ALL}"
SAMPLE_BUDGET_VAL="${COLLECT_SAMPLE_BUDGET_VAL:-$SAMPLE_BUDGET_ALL}"
SAMPLE_BUDGET_REPORT_STEP_PCT="${COLLECT_SAMPLE_BUDGET_REPORT_STEP_PCT:-5}"
COLLECT_SPLITS="${COLLECT_SPLITS:-train,val}"
COLLECT_JOBS="${COLLECT_JOBS:-$(bbml_default_solver_jobs)}"
COLLECT_FORCE="${COLLECT_FORCE:-0}"
if [ -n "${COLLECT_ORACLE:-}" ]; then
  COLLECT_ORACLE_VALUE="$COLLECT_ORACLE"
elif [ "${COLLECT_STRONGBRANCH:-0}" = "1" ]; then
  COLLECT_ORACLE_VALUE="strongbranch"
else
  COLLECT_ORACLE_VALUE="vanillafullstrong"
fi

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
echo "  Telemetry cap: $([ "$TELEMETRY_MAX_NODES_PER_INSTANCE" -gt 0 ] && printf '%s nodes/instance' "$TELEMETRY_MAX_NODES_PER_INSTANCE" || printf 'off')"
echo "  Query prob   : $TELEMETRY_QUERY_EXPERT_PROB"
echo "  Sample budget: train=$SAMPLE_BUDGET_TRAIN val=$SAMPLE_BUDGET_VAL"
echo "  Budget step  : ${SAMPLE_BUDGET_REPORT_STEP_PCT}%"
echo "  Collect jobs : $COLLECT_JOBS"
echo "  Oracle       : $COLLECT_ORACLE_VALUE"
echo "  Resume mode  : $([ "$COLLECT_FORCE" = "1" ] && printf 'off (force rerun)' || printf 'on')"
echo "  Runner       : $BBML_RUNNER_BIN"
echo ""

manifest="$DATA_DIR/manifests/collect_tasks.jsonl"
: > "$manifest"

for family in "${families[@]}"; do
  for split in "${SPLIT_LIST[@]}"; do
    list_file="$INSTANCES_DIR/${family}_${split}.txt"
    [ ! -f "$list_file" ] && continue
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
    TELEMETRY_MAX_NODES_PER_INSTANCE="$TELEMETRY_MAX_NODES_PER_INSTANCE" \
    TELEMETRY_QUERY_EXPERT_PROB="$TELEMETRY_QUERY_EXPERT_PROB" \
    SAMPLE_BUDGET_TRAIN="$SAMPLE_BUDGET_TRAIN" \
    SAMPLE_BUDGET_VAL="$SAMPLE_BUDGET_VAL" \
    SAMPLE_BUDGET_REPORT_STEP_PCT="$SAMPLE_BUDGET_REPORT_STEP_PCT" \
    FORCE="$COLLECT_FORCE" \
    ORACLE="$COLLECT_ORACLE_VALUE" \
    python3 - <<'PY'
import json
import os
from pathlib import Path
import torch

def instance_id(path: Path) -> str:
    name = path.name
    for suffix in (".lp.gz", ".mps.gz", ".lp", ".mps", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name

def completed_log(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    return "BBML_SUMMARY" in text

def resolve_instance(raw: str, family: str, split: str, data_dir: Path) -> Path:
    inst_path = Path(raw).expanduser()
    if inst_path.exists():
        return inst_path.resolve()
    fallback = data_dir / "instances" / family / split / inst_path.name
    if fallback.exists():
        return fallback.resolve()
    if inst_path.name.endswith(".lp"):
        gzip_fallback = data_dir / "instances" / family / split / f"{inst_path.name}.gz"
        if gzip_fallback.exists():
            return gzip_fallback.resolve()
    return inst_path

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
telemetry_max_nodes_per_instance = int(os.environ["TELEMETRY_MAX_NODES_PER_INSTANCE"])
telemetry_query_expert_prob = os.environ["TELEMETRY_QUERY_EXPERT_PROB"]
sample_budget_value = int(os.environ[f"SAMPLE_BUDGET_{split.upper()}"])
sample_budget_report_step_pct = max(1, int(os.environ["SAMPLE_BUDGET_REPORT_STEP_PCT"]))
force = os.environ["FORCE"] == "1"
oracle = os.environ["ORACLE"].strip() or "vanillafullstrong"

candidate_dir = data_dir / "logs" / family / split / "candidates"
graph_dir = data_dir / "logs" / family / split / "graph"
scip_dir = data_dir / "logs" / family / split / "scip"
for path in (candidate_dir, graph_dir, scip_dir):
    path.mkdir(parents=True, exist_ok=True)

manifest.parent.mkdir(parents=True, exist_ok=True)
budget_state_path = data_dir / "manifests" / f"{family}_{split}.sample_budget.json"
existing_samples = 0
if sample_budget_value > 0:
    for path in sorted(graph_dir.glob("*.pt")):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        payload = torch.load(path, map_location="cpu")
        existing_samples += int(len(payload.get("items", [])))
    if existing_samples < sample_budget_value:
        for path in sorted(graph_dir.glob("*.ndjson")):
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            with path.open() as fh:
                existing_samples += sum(1 for line in fh if line.strip())
            if existing_samples >= sample_budget_value:
                break
    budget_payload = {
        "family": family,
        "split": split,
        "budget": sample_budget_value,
        "used": min(existing_samples, sample_budget_value),
        "report_step_pct": sample_budget_report_step_pct,
        "reported_bucket": (min(existing_samples, sample_budget_value) * 100 // sample_budget_value) // sample_budget_report_step_pct,
    }
    budget_state_path.write_text(json.dumps(budget_payload, indent=2, sort_keys=True) + "\n")
total = 0
skipped = 0
runnable = 0
with list_file.open() as src, manifest.open("a") as out:
    for line in src:
        inst = line.strip()
        if not inst:
            continue
        inst_path = resolve_instance(inst, family, split, data_dir)
        iid = instance_id(inst_path)
        for seed in seeds:
            total += 1
            candidate_out = candidate_dir / f"{iid}_s{seed}.ndjson.gz"
            graph_out = graph_dir / f"{iid}_s{seed}.pt"
            legacy_candidate_out = candidate_dir / f"{iid}_s{seed}.ndjson"
            legacy_graph_out = graph_dir / f"{iid}_s{seed}.ndjson"
            scip_log = scip_dir / f"{iid}_s{seed}.log"
            done_marker = candidate_out.with_suffix(candidate_out.suffix + ".done")
            skip = False
            if not force:
                skip = (
                    (
                        completed_log(scip_log)
                        and (
                            (
                                (
                                    (candidate_out.is_file() and candidate_out.stat().st_size > 0)
                                    or (legacy_candidate_out.is_file() and legacy_candidate_out.stat().st_size > 0)
                                )
                                and (
                                    (graph_out.is_file() and graph_out.stat().st_size > 0)
                                    or (legacy_graph_out.is_file() and legacy_graph_out.stat().st_size > 0)
                                )
                            )
                            or done_marker.is_file()
                        )
                    )
                    or (sample_budget_value > 0 and existing_samples >= sample_budget_value)
                )
            if skip:
                skipped += 1
            else:
                runnable += 1
            rec = {
                "name": f"collect:{family}:{split}:{iid}:s{seed}",
                "cmd": py_launch
                + [
                    script,
                    "--runner-bin",
                        runner_bin,
                        "--instance",
                        str(inst_path),
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
                    "--telemetry-max-nodes-per-instance",
                    str(telemetry_max_nodes_per_instance),
                    "--telemetry-query-expert-prob",
                    telemetry_query_expert_prob,
                    "--sample-budget-state",
                    str(budget_state_path),
                    "--telemetry-oracle",
                    oracle,
                ],
                "cwd": os.getcwd(),
                "log_path": str(data_dir / "pipeline_logs" / "collect" / f"{family}_{split}_{iid}_s{seed}.log"),
                "skip": skip,
            }
            if oracle == "strongbranch":
                rec["cmd"].append("--telemetry-strongbranch")
            out.write(json.dumps(rec) + "\n")
print(f"  queued total={total} runnable={runnable} skipped={skipped}")
if sample_budget_value > 0:
    print(f"  existing samples={existing_samples} budget={sample_budget_value}")
PY
    echo ""
  done
done

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$manifest" --jobs "$COLLECT_JOBS"

echo ""
echo "Collection complete."
