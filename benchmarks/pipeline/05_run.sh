#!/usr/bin/env bash
# Step 05: Run supported baselines on test instances with one JSON artifact per run.
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python
bbml_require_runner

RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
MODEL_DIR="$RESULTS_DIR/models"
TL="${BENCH_TL:-3600}"
SEEDS="${BENCH_SEEDS:-0 1 2 3 4}"
RUN_JOBS="${RUN_JOBS:-$(bbml_default_solver_jobs)}"

METHODS="${BENCH_METHODS:-scip-default,strong-branch,bbml-mlp,bbml-gnn-varonly,bbml-gnn-graph,bbml-gnn-graph-fp32,bbml-gnn-graph-fp16}"
INSTANCE_SET_FILTER=""
INCLUDE_ABLATIONS="${BENCH_INCLUDE_ABLATIONS:-0}"
ABLATION_METHODS="pure-imitation,alpha-fixed-0.5,solver-only-alpha0,no-temperature,no-cond-gate,no-conf-gate"

for arg in "$@"; do
  case $arg in
    --set=*) INSTANCE_SET_FILTER="${arg#*=}" ;;
    --methods=*) METHODS="${arg#*=}" ;;
    --include-ablations) INCLUDE_ABLATIONS=1 ;;
  esac
done

if [ "$INCLUDE_ABLATIONS" = "1" ]; then
  if [ -n "$METHODS" ]; then
    METHODS="$METHODS,$ABLATION_METHODS"
  else
    METHODS="$ABLATION_METHODS"
  fi
fi

RUNS_DIR="$RESULTS_DIR/runs"
SCIP_LOG_DIR="$RESULTS_DIR/scip_logs"
MANIFEST_DIR="${DATA_DIR:-$BBML_ROOT/data}/manifests"
TASK_LOG_DIR="${DATA_DIR:-$BBML_ROOT/data}/pipeline_logs/run"
mkdir -p "$RUNS_DIR" "$SCIP_LOG_DIR" "$MANIFEST_DIR" "$TASK_LOG_DIR"

INSTANCE_SETS=()
if [ -n "$INSTANCE_SET_FILTER" ]; then
  INSTANCE_SETS+=("$INSTANCE_SET_FILTER")
else
  while IFS= read -r list_file; do
    INSTANCE_SETS+=("$(basename "$list_file" .txt)")
  done < <(find "$BBML_ROOT/benchmarks/instances" -maxdepth 1 -name '*_test.txt' | sort)
fi

if [ ${#INSTANCE_SETS[@]} -eq 0 ]; then
  echo "ERROR: no test instance lists found."
  exit 1
fi

if [[ ",$METHODS," == *",strong-branch,"* ]]; then
  probe_instance="$(bbml_find_probe_instance)"
  if [ -z "$probe_instance" ]; then
    echo "WARNING: no probe instance found; dropping strong-branch."
    METHODS="$(printf '%s' "$METHODS" | sed 's/,\?strong-branch,\?/,/g; s/^,//; s/,$//; s/,,*/,/g')"
  else
    probe_out="$("$BBML_RUNNER_BIN" --problem "$probe_instance" --param 'bbml/enable=FALSE' --param 'branching/fullstrong/priority=1000000' 2>&1 || true)"
    if printf '%s' "$probe_out" | grep -Eq 'ERROR:|error:|wrong parameter type|unknown parameter'; then
      echo "WARNING: branching/fullstrong/priority is unavailable in this SCIP build; dropping strong-branch."
      METHODS="$(printf '%s' "$METHODS" | sed 's/,\?strong-branch,\?/,/g; s/^,//; s/,$//; s/,,*/,/g')"
    fi
  fi
fi

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
manifest="$MANIFEST_DIR/run_tasks.jsonl"

echo "=== Benchmark run ==="
echo "  Instance sets : ${INSTANCE_SETS[*]}"
echo "  Methods       : $METHODS"
echo "  Seeds         : $SEEDS"
echo "  Time limit    : ${TL}s"
echo "  Run jobs      : $RUN_JOBS"
echo "  Runner        : $BBML_RUNNER_BIN"
echo ""

PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
BBML_ROOT="$BBML_ROOT" \
RESULTS_DIR="$RESULTS_DIR" \
MODEL_DIR="$MODEL_DIR" \
BBML_RUNNER="$BBML_RUNNER_BIN" \
TIME_LIMIT="$TL" \
SEEDS="$SEEDS" \
METHODS="$METHODS" \
INSTANCE_SETS="$(IFS=,; echo "${INSTANCE_SETS[*]}")" \
SCRIPT="$SCRIPT_DIR/run_benchmark_task.py" \
TASK_LOG_DIR="$TASK_LOG_DIR" \
MANIFEST="$manifest" \
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

py_launch = json.loads(os.environ["PY_LAUNCH_JSON"])
root = Path(os.environ["BBML_ROOT"])
results_dir = Path(os.environ["RESULTS_DIR"])
model_dir = Path(os.environ["MODEL_DIR"])
runner_bin = os.environ["BBML_RUNNER"]
time_limit = os.environ["TIME_LIMIT"]
seeds = [seed for seed in os.environ["SEEDS"].split() if seed]
methods = [method for method in os.environ["METHODS"].split(",") if method]
instance_sets = [name for name in os.environ["INSTANCE_SETS"].split(",") if name]
script = os.environ["SCRIPT"]
task_log_dir = Path(os.environ["TASK_LOG_DIR"])
manifest = Path(os.environ["MANIFEST"])

graph_ensemble_paths = sorted(model_dir.glob("bbml_gnn_graph_member*.onnx"))
graph_ensemble_model = ",".join(str(path) for path in graph_ensemble_paths) if graph_ensemble_paths else str(model_dir / "bbml_gnn_graph.onnx")

method_specs = {
    "scip-default": {"disable_ml": True},
    "strong-branch": {"disable_ml": True},
    "bbml-mlp": {
        "model": str(model_dir / "bbml_mlp.onnx"),
        "temperature_file": str(model_dir / "bbml_mlp.temperature.txt"),
    },
    "bbml-gnn-varonly": {
        "model": str(model_dir / "bbml_gnn_varonly.onnx"),
        "temperature_file": str(model_dir / "bbml_gnn_varonly.temperature.txt"),
    },
    "bbml-gnn-graph": {
        "model": graph_ensemble_model,
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
    },
    "bbml-gnn-graph-fp32": {
        "model": str(model_dir / "bbml_gnn_graph_fp32.onnx"),
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
    },
    "bbml-gnn-graph-fp16": {
        "model": str(model_dir / "bbml_gnn_graph_fp16.onnx"),
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
    },
    "pure-imitation": {
        "model": graph_ensemble_model,
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
        "alpha_min": 1.0,
        "alpha_max": 1.0,
        "depth_penalty": 0.0,
    },
    "alpha-fixed-0.5": {
        "model": graph_ensemble_model,
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
        "alpha_min": 0.5,
        "alpha_max": 0.5,
        "depth_penalty": 0.0,
    },
    "solver-only-alpha0": {
        "model": graph_ensemble_model,
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
        "alpha_min": 0.0,
        "alpha_max": 0.0,
        "depth_penalty": 0.0,
    },
    "no-temperature": {
        "model": graph_ensemble_model,
        "temperature": 1.0,
    },
    "no-cond-gate": {
        "model": graph_ensemble_model,
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
        "cond_threshold": 1.0e20,
    },
    "no-conf-gate": {
        "model": graph_ensemble_model,
        "temperature_file": str(model_dir / "bbml_gnn_graph.temperature.txt"),
        "disable_confidence_gate": True,
    },
}

with manifest.open("w") as fh:
    for instance_set in instance_sets:
        list_file = root / "benchmarks" / "instances" / f"{instance_set}.txt"
        if not list_file.exists():
            continue
        for line in list_file.read_text().splitlines():
            inst = line.strip()
            if not inst:
                continue
            iid = instance_id(Path(inst))
            for method in methods:
                spec = method_specs.get(method)
                if spec is None:
                    raise ValueError(f"Unsupported benchmark method: {method}")
                for seed in seeds:
                    result_out = results_dir / "runs" / method / f"{iid}_s{seed}.json"
                    scip_log = results_dir / "scip_logs" / method / f"{iid}_s{seed}.log"
                    cmd = py_launch + [
                        script,
                        "--runner-bin",
                        runner_bin,
                        "--instance",
                        inst,
                        "--solver",
                        method,
                        "--seed",
                        seed,
                        "--time-limit",
                        time_limit,
                        "--result-out",
                        str(result_out),
                        "--scip-log",
                        str(scip_log),
                    ]
                    if spec.get("disable_ml"):
                        cmd.append("--disable-ml")
                    if spec.get("model"):
                        cmd += ["--model", str(spec["model"])]
                    if spec.get("temperature_file"):
                        cmd += ["--temperature-file", str(spec["temperature_file"])]
                    if spec.get("temperature") is not None:
                        cmd += ["--temperature", str(spec["temperature"])]
                    if spec.get("alpha_min") is not None:
                        cmd += ["--alpha-min", str(spec["alpha_min"])]
                    if spec.get("alpha_max") is not None:
                        cmd += ["--alpha-max", str(spec["alpha_max"])]
                    if spec.get("depth_penalty") is not None:
                        cmd += ["--depth-penalty", str(spec["depth_penalty"])]
                    if spec.get("alpha_theta") is not None:
                        cmd += ["--alpha-theta", str(spec["alpha_theta"])]
                    if spec.get("confidence") is not None:
                        cmd += ["--confidence", str(spec["confidence"])]
                    if spec.get("cond_threshold") is not None:
                        cmd += ["--cond-threshold", str(spec["cond_threshold"])]
                    if spec.get("disable_confidence_gate"):
                        cmd.append("--disable-confidence-gate")
                    fh.write(
                        json.dumps(
                            {
                                "name": f"run:{instance_set}:{method}:{iid}:s{seed}",
                                "cmd": cmd,
                                "cwd": os.getcwd(),
                                "log_path": str(task_log_dir / f"{instance_set}_{method}_{iid}_s{seed}.log"),
                                "skip": result_out.is_file() and result_out.stat().st_size > 0,
                            }
                        )
                        + "\n"
                    )
PY

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$manifest" --jobs "$RUN_JOBS"

echo ""
echo "Runs complete. Results in $RUNS_DIR"
