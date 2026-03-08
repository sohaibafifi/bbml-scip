#!/usr/bin/env bash
# Step 04: Calibrate supported checkpoints, export ONNX, and benchmark latency.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
MODEL_DIR="$RESULTS_DIR/models"
EXPORT_JOBS="${EXPORT_JOBS:-2}"
VAL_PARQUET="$DATA_DIR/parquet/val.parquet"
GRAPH_VAL_MANIFEST="$DATA_DIR/manifests/graph/val.txt"

GNN_GRAPH_CKPT="$MODEL_DIR/bbml_gnn_graph_best.pt"
GNN_VARONLY_CKPT="$MODEL_DIR/bbml_gnn_varonly_best.pt"
MLP_CKPT="$MODEL_DIR/bbml_mlp_best.pt"

for path in "$VAL_PARQUET" "$GRAPH_VAL_MANIFEST" "$GNN_GRAPH_CKPT" "$GNN_VARONLY_CKPT" "$MLP_CKPT"; do
  [ ! -f "$path" ] && echo "ERROR: missing required artifact: $path" && exit 1
done

mkdir -p "$RESULTS_DIR" "$DATA_DIR/manifests" "$DATA_DIR/pipeline_logs/export"
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

echo "=== Calibration and export ==="
echo "  Export jobs : $EXPORT_JOBS"
echo ""

echo "[1/3] Fitting checkpoint-aware temperatures..."
"${PYTHON_CMD[@]}" -m bbml.train.calibrate \
  --ckpt "$GNN_GRAPH_CKPT" \
  --parquet "$VAL_PARQUET" \
  --graph_manifest "$GRAPH_VAL_MANIFEST" \
  --device cpu \
  --out "$MODEL_DIR/bbml_gnn_graph.temperature.txt"

"${PYTHON_CMD[@]}" -m bbml.train.calibrate \
  --ckpt "$GNN_VARONLY_CKPT" \
  --parquet "$VAL_PARQUET" \
  --device cpu \
  --out "$MODEL_DIR/bbml_gnn_varonly.temperature.txt"

"${PYTHON_CMD[@]}" -m bbml.train.calibrate \
  --ckpt "$MLP_CKPT" \
  --parquet "$VAL_PARQUET" \
  --device cpu \
  --out "$MODEL_DIR/bbml_mlp.temperature.txt"

export_manifest="$DATA_DIR/manifests/export_tasks.jsonl"
PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
MODEL_DIR="$MODEL_DIR" \
DATA_DIR="$DATA_DIR" \
python3 - <<'PY'
import json
import os
from pathlib import Path

py_launch = json.loads(os.environ["PY_LAUNCH_JSON"])
model_dir = Path(os.environ["MODEL_DIR"])
data_dir = Path(os.environ["DATA_DIR"])
manifest = data_dir / "manifests" / "export_tasks.jsonl"
log_dir = data_dir / "pipeline_logs" / "export"
log_dir.mkdir(parents=True, exist_ok=True)

tasks = [
    ("bbml_gnn_graph_fp32", [*py_launch, "-m", "bbml.export.export_onnx", "--ckpt", str(model_dir / "bbml_gnn_graph_best.pt"), "--out", str(model_dir / "bbml_gnn_graph_fp32.onnx")]),
    ("bbml_gnn_graph_fp16", [*py_launch, "-m", "bbml.export.export_onnx", "--ckpt", str(model_dir / "bbml_gnn_graph_best.pt"), "--fp16", "--out", str(model_dir / "bbml_gnn_graph_fp16.onnx")]),
    ("bbml_gnn_varonly", [*py_launch, "-m", "bbml.export.export_onnx", "--ckpt", str(model_dir / "bbml_gnn_varonly_best.pt"), "--out", str(model_dir / "bbml_gnn_varonly.onnx")]),
    ("bbml_mlp", [*py_launch, "-m", "bbml.export.export_onnx", "--ckpt", str(model_dir / "bbml_mlp_best.pt"), "--out", str(model_dir / "bbml_mlp.onnx")]),
]

with manifest.open("w") as fh:
    for name, cmd in tasks:
        out_path = Path(cmd[-1])
        fh.write(
            json.dumps(
                {
                    "name": f"export:{name}",
                    "cmd": cmd,
                    "cwd": os.getcwd(),
                    "log_path": str(log_dir / f"{name}.log"),
                    "skip": out_path.is_file() and out_path.stat().st_size > 0,
                }
            )
            + "\n"
        )
PY

echo "[2/3] Exporting ONNX models..."
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$export_manifest" --jobs "$EXPORT_JOBS"
ln -sf bbml_gnn_graph_fp16.onnx "$MODEL_DIR/bbml_gnn_graph.onnx"

latency_manifest="$DATA_DIR/manifests/latency_tasks.jsonl"
PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
MODEL_DIR="$MODEL_DIR" \
RESULTS_DIR="$RESULTS_DIR" \
DATA_DIR="$DATA_DIR" \
"${PYTHON_CMD[@]}" - <<'PY'
import json
import os
from pathlib import Path
import torch

py_launch = json.loads(os.environ["PY_LAUNCH_JSON"])
model_dir = Path(os.environ["MODEL_DIR"])
results_dir = Path(os.environ["RESULTS_DIR"])
data_dir = Path(os.environ["DATA_DIR"])
manifest = data_dir / "manifests" / "latency_tasks.jsonl"
log_dir = data_dir / "pipeline_logs" / "export"
log_dir.mkdir(parents=True, exist_ok=True)


def load_cfg(name: str) -> dict:
    ckpt = torch.load(model_dir / name, map_location="cpu")
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(cfg, dict):
        raise ValueError(f"missing cfg metadata in checkpoint {name}")
    return cfg


graph_cfg = load_cfg("bbml_gnn_graph_best.pt")
varonly_cfg = load_cfg("bbml_gnn_varonly_best.pt")
mlp_cfg = load_cfg("bbml_mlp_best.pt")

tasks = [
    (
        "latency_gnn_graph_fp32",
        [
            *py_launch,
            "-m",
            "bbml.bench.latency",
            "--onnx",
            str(model_dir / "bbml_gnn_graph_fp32.onnx"),
            "--d_var",
            str(graph_cfg.get("d_var", 13)),
            "--d_con",
            str(graph_cfg.get("d_con", 4)),
            "--runs",
            "100",
        ],
        results_dir / "latency_gnn_graph_fp32.txt",
    ),
    (
        "latency_gnn_graph_fp16",
        [
            *py_launch,
            "-m",
            "bbml.bench.latency",
            "--onnx",
            str(model_dir / "bbml_gnn_graph_fp16.onnx"),
            "--d_var",
            str(graph_cfg.get("d_var", 13)),
            "--d_con",
            str(graph_cfg.get("d_con", 4)),
            "--runs",
            "100",
        ],
        results_dir / "latency_gnn_graph_fp16.txt",
    ),
    (
        "latency_gnn_varonly",
        [
            *py_launch,
            "-m",
            "bbml.bench.latency",
            "--onnx",
            str(model_dir / "bbml_gnn_varonly.onnx"),
            "--d",
            str(varonly_cfg.get("d_var", 10)),
            "--runs",
            "100",
        ],
        results_dir / "latency_gnn_varonly.txt",
    ),
    (
        "latency_mlp",
        [
            *py_launch,
            "-m",
            "bbml.bench.latency",
            "--onnx",
            str(model_dir / "bbml_mlp.onnx"),
            "--d",
            str(mlp_cfg.get("d_in", 10)),
            "--runs",
            "100",
        ],
        results_dir / "latency_mlp.txt",
    ),
]

with manifest.open("w") as fh:
    for name, cmd, out_path in tasks:
        fh.write(
            json.dumps(
                {
                    "name": name,
                    "cmd": cmd,
                    "cwd": os.getcwd(),
                    "log_path": str(out_path),
                    "skip": False,
                }
            )
            + "\n"
        )
PY

echo "[3/3] Benchmarking ONNX latency..."
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$latency_manifest" --jobs "$EXPORT_JOBS"

echo ""
echo "Calibration and export complete. Outputs in $MODEL_DIR and $RESULTS_DIR"
