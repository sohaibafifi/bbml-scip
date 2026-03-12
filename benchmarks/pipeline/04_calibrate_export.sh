#!/usr/bin/env bash
# Step 04: Calibrate supported checkpoints, export ONNX, and benchmark latency.
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
MODEL_DIR="$RESULTS_DIR/models"
EXPORT_JOBS="${EXPORT_JOBS:-2}"
CALIBRATE_DEVICE="${CALIBRATE_DEVICE:-$(bbml_detect_torch_device)}"
CALIBRATE_NUM_WORKERS="${CALIBRATE_NUM_WORKERS:--1}"
CALIBRATE_PIN_MEMORY="${CALIBRATE_PIN_MEMORY:--1}"
EXPORT_FORCE="${EXPORT_FORCE:-0}"
VAL_PARQUET="$DATA_DIR/parquet/val.parquet"
GRAPH_VAL_MANIFEST="$DATA_DIR/manifests/graph/val.txt"

GNN_GRAPH_CKPT="$MODEL_DIR/bbml_gnn_graph_best.pt"
GNN_VARONLY_CKPT="$MODEL_DIR/bbml_gnn_varonly_best.pt"
MLP_CKPT="$MODEL_DIR/bbml_mlp_best.pt"
shopt -s nullglob
GRAPH_ENSEMBLE_CKPTS=("$GNN_GRAPH_CKPT" "$MODEL_DIR"/bbml_gnn_graph_member*_best.pt)
shopt -u nullglob
GRAPH_ENSEMBLE_SPEC="$(IFS=,; printf '%s' "${GRAPH_ENSEMBLE_CKPTS[*]}")"

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

output_meta_path() {
  printf '%s.meta.json' "$1"
}

build_path_signature() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(json.dumps({"path": str(path), "exists": False}, sort_keys=True))
else:
    print(json.dumps({
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }, sort_keys=True))
PY
}

build_calibration_signature() {
  local ckpt_spec="$1"
  local parquet_path="$2"
  local graph_manifest="${3:-}"
  python3 - "$ckpt_spec" "$parquet_path" "$graph_manifest" "$CALIBRATE_NUM_WORKERS" "$CALIBRATE_PIN_MEMORY" <<'PY'
import json
import sys
from pathlib import Path

ckpt_spec, parquet_path, graph_manifest, num_workers, pin_memory = sys.argv[1:]

def stat_payload(path: Path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }

payload = {
    "schema_version": 1,
    "type": "calibration",
    "ckpts": [stat_payload(Path(part)) for part in ckpt_spec.split(",") if part],
    "parquet": stat_payload(Path(parquet_path)),
    "num_workers": int(num_workers),
    "pin_memory": int(pin_memory),
}
if graph_manifest:
    manifest = Path(graph_manifest)
    payload["graph_manifest"] = stat_payload(manifest)
    if manifest.exists():
        lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload["graph_sources"] = [stat_payload(Path(line)) for line in lines]
print(json.dumps(payload, sort_keys=True))
PY
}

build_export_signature() {
  local ckpt_path="$1"
  local output_path="$2"
  local fp16_flag="$3"
  python3 - "$ckpt_path" "$output_path" "$fp16_flag" <<'PY'
import json
import sys
from pathlib import Path

ckpt_path, output_path, fp16_flag = sys.argv[1:]

def stat_payload(path: Path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }

payload = {
    "schema_version": 1,
    "type": "onnx_export",
    "ckpt": stat_payload(Path(ckpt_path)),
    "output": str(output_path),
    "fp16": fp16_flag == "1",
}
print(json.dumps(payload, sort_keys=True))
PY
}

build_latency_signature() {
  local onnx_path="$1"
  local d_var="$2"
  local d_con="$3"
  local runs="$4"
  python3 - "$onnx_path" "$d_var" "$d_con" "$runs" <<'PY'
import json
import sys
from pathlib import Path

onnx_path, d_var, d_con, runs = sys.argv[1:]
path = Path(onnx_path)
payload = {
    "schema_version": 1,
    "type": "latency",
    "onnx": {
        "path": str(path),
        "exists": path.exists(),
        **({"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} if path.exists() else {}),
    },
    "d_var": int(d_var),
    "d_con": int(d_con),
    "runs": int(runs),
}
print(json.dumps(payload, sort_keys=True))
PY
}

output_task_complete() {
  local output_path="$1"
  local signature_json="$2"
  python3 - "$output_path" "$signature_json" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
sig = json.loads(sys.argv[2])
meta = Path(str(out) + ".meta.json")
if not out.is_file() or out.stat().st_size <= 0:
    raise SystemExit(1)
if not meta.is_file() or meta.stat().st_size <= 0:
    raise SystemExit(1)
try:
    payload = json.loads(meta.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("signature") == sig else 1)
PY
}

run_output_command() {
  local signature_json="$1"
  local output_path="$2"
  shift 2
  "${PYTHON_CMD[@]}" "$SCRIPT_DIR/run_output_task.py" \
    --output "$output_path" \
    --meta "$(output_meta_path "$output_path")" \
    --signature-json "$signature_json" \
    -- \
    "${PYTHON_CMD[@]}" "$@"
}

echo "=== Calibration and export ==="
echo "  Export jobs : $EXPORT_JOBS"
echo "  Device      : $CALIBRATE_DEVICE"
echo "  Workers     : $CALIBRATE_NUM_WORKERS"
echo "  Pin memory  : $CALIBRATE_PIN_MEMORY"
echo "  Resume      : $( [ "$EXPORT_FORCE" = "1" ] && printf 'off (EXPORT_FORCE=1)' || printf 'on' )"
echo ""

echo "[1/3] Fitting checkpoint-aware temperatures..."
graph_temp_sig="$(build_calibration_signature "$GRAPH_ENSEMBLE_SPEC" "$VAL_PARQUET" "$GRAPH_VAL_MANIFEST")"
if [ "$EXPORT_FORCE" != "1" ] && output_task_complete "$MODEL_DIR/bbml_gnn_graph.temperature.txt" "$graph_temp_sig"; then
  echo "  - graph ensemble temperature already fitted -> $MODEL_DIR/bbml_gnn_graph.temperature.txt"
else
  run_output_command "$graph_temp_sig" "$MODEL_DIR/bbml_gnn_graph.temperature.txt" \
    -m bbml.train.calibrate \
    --ckpt "$GRAPH_ENSEMBLE_SPEC" \
    --parquet "$VAL_PARQUET" \
    --graph_manifest "$GRAPH_VAL_MANIFEST" \
    --device "$CALIBRATE_DEVICE" \
    --num_workers "$CALIBRATE_NUM_WORKERS" \
    --pin_memory "$CALIBRATE_PIN_MEMORY" \
    --out "$MODEL_DIR/bbml_gnn_graph.temperature.txt"
fi

varonly_temp_sig="$(build_calibration_signature "$GNN_VARONLY_CKPT" "$VAL_PARQUET")"
if [ "$EXPORT_FORCE" != "1" ] && output_task_complete "$MODEL_DIR/bbml_gnn_varonly.temperature.txt" "$varonly_temp_sig"; then
  echo "  - var-only temperature already fitted -> $MODEL_DIR/bbml_gnn_varonly.temperature.txt"
else
  run_output_command "$varonly_temp_sig" "$MODEL_DIR/bbml_gnn_varonly.temperature.txt" \
    -m bbml.train.calibrate \
    --ckpt "$GNN_VARONLY_CKPT" \
    --parquet "$VAL_PARQUET" \
    --device "$CALIBRATE_DEVICE" \
    --num_workers "$CALIBRATE_NUM_WORKERS" \
    --pin_memory "$CALIBRATE_PIN_MEMORY" \
    --out "$MODEL_DIR/bbml_gnn_varonly.temperature.txt"
fi

mlp_temp_sig="$(build_calibration_signature "$MLP_CKPT" "$VAL_PARQUET")"
if [ "$EXPORT_FORCE" != "1" ] && output_task_complete "$MODEL_DIR/bbml_mlp.temperature.txt" "$mlp_temp_sig"; then
  echo "  - MLP temperature already fitted -> $MODEL_DIR/bbml_mlp.temperature.txt"
else
  run_output_command "$mlp_temp_sig" "$MODEL_DIR/bbml_mlp.temperature.txt" \
    -m bbml.train.calibrate \
    --ckpt "$MLP_CKPT" \
    --parquet "$VAL_PARQUET" \
    --device "$CALIBRATE_DEVICE" \
    --num_workers "$CALIBRATE_NUM_WORKERS" \
    --pin_memory "$CALIBRATE_PIN_MEMORY" \
    --out "$MODEL_DIR/bbml_mlp.temperature.txt"
fi

export_manifest="$DATA_DIR/manifests/export_tasks.jsonl"
PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
MODEL_DIR="$MODEL_DIR" \
DATA_DIR="$DATA_DIR" \
EXPORT_FORCE="$EXPORT_FORCE" \
SCRIPT_DIR="$SCRIPT_DIR" \
"${PYTHON_CMD[@]}" - <<'PY'
import json
import os
from pathlib import Path

py_launch = json.loads(os.environ["PY_LAUNCH_JSON"])
model_dir = Path(os.environ["MODEL_DIR"])
data_dir = Path(os.environ["DATA_DIR"])
manifest = data_dir / "manifests" / "export_tasks.jsonl"
log_dir = data_dir / "pipeline_logs" / "export"
log_dir.mkdir(parents=True, exist_ok=True)
force = os.environ.get("EXPORT_FORCE", "0") == "1"
script_dir = Path(os.environ["SCRIPT_DIR"])

tasks = [
    ("bbml_gnn_graph_fp32", model_dir / "bbml_gnn_graph_best.pt", model_dir / "bbml_gnn_graph_fp32.onnx", False),
    ("bbml_gnn_graph_fp16", model_dir / "bbml_gnn_graph_best.pt", model_dir / "bbml_gnn_graph_fp16.onnx", True),
    ("bbml_gnn_varonly", model_dir / "bbml_gnn_varonly_best.pt", model_dir / "bbml_gnn_varonly.onnx", False),
    ("bbml_mlp", model_dir / "bbml_mlp_best.pt", model_dir / "bbml_mlp.onnx", False),
]

member_ckpts = [model_dir / "bbml_gnn_graph_best.pt"]
member_ckpts.extend(sorted(model_dir.glob("bbml_gnn_graph_member*_best.pt")))
for idx, ckpt in enumerate(member_ckpts[1:], start=1):
    tasks.append((f"bbml_gnn_graph_member{idx}", ckpt, model_dir / f"bbml_gnn_graph_member{idx}.onnx", True))


def export_signature(ckpt: Path, out_path: Path, fp16: bool) -> dict:
    return {
        "schema_version": 1,
        "type": "onnx_export",
        "ckpt": {
            "path": str(ckpt),
            "exists": ckpt.exists(),
            **({"size": ckpt.stat().st_size, "mtime_ns": ckpt.stat().st_mtime_ns} if ckpt.exists() else {}),
        },
        "output": str(out_path),
        "fp16": fp16,
    }


def completed(path: Path, signature: dict) -> bool:
    meta = Path(str(path) + ".meta.json")
    if not path.is_file() or path.stat().st_size <= 0 or not meta.is_file() or meta.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(meta.read_text())
    except Exception:
        return False
    return payload.get("signature") == signature

with manifest.open("w") as fh:
    for name, ckpt, out_path, fp16 in tasks:
        inner = [*py_launch, "-m", "bbml.export.export_onnx", "--ckpt", str(ckpt)]
        if fp16:
            inner.append("--fp16")
        inner.extend(["--out", str(out_path)])
        signature = json.dumps(export_signature(ckpt, out_path, fp16), sort_keys=True)
        cmd = [
            *py_launch,
            str(script_dir / "run_output_task.py"),
            "--output",
            str(out_path),
            "--meta",
            str(Path(str(out_path) + ".meta.json")),
            "--signature-json",
            signature,
            "--",
            *inner,
        ]
        fh.write(
            json.dumps(
                {
                    "name": f"export:{name}",
                    "cmd": cmd,
                    "cwd": os.getcwd(),
                    "log_path": str(log_dir / f"{name}.log"),
                    "skip": (not force) and completed(out_path, export_signature(ckpt, out_path, fp16)),
                }
            )
            + "\n"
        )
PY

echo "[2/3] Exporting ONNX models..."
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$export_manifest" --jobs "$EXPORT_JOBS"
ln -sf bbml_gnn_graph_fp16.onnx "$MODEL_DIR/bbml_gnn_graph.onnx"
ln -sf bbml_gnn_graph_fp16.onnx "$MODEL_DIR/bbml_gnn_graph_member0.onnx"

latency_manifest="$DATA_DIR/manifests/latency_tasks.jsonl"
PY_LAUNCH_JSON="$PY_LAUNCH_JSON" \
MODEL_DIR="$MODEL_DIR" \
RESULTS_DIR="$RESULTS_DIR" \
DATA_DIR="$DATA_DIR" \
EXPORT_FORCE="$EXPORT_FORCE" \
SCRIPT_DIR="$SCRIPT_DIR" \
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
force = os.environ.get("EXPORT_FORCE", "0") == "1"
script_dir = Path(os.environ["SCRIPT_DIR"])


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


def completed(path: Path, signature: dict) -> bool:
    meta = Path(str(path) + ".meta.json")
    if not path.is_file() or path.stat().st_size <= 0 or not meta.is_file() or meta.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(meta.read_text())
    except Exception:
        return False
    return payload.get("signature") == signature

with manifest.open("w") as fh:
    for name, cmd, out_path in tasks:
        onnx_path = Path(cmd[cmd.index("--onnx") + 1])
        d_var = int(cmd[cmd.index("--d_var") + 1]) if "--d_var" in cmd else int(cmd[cmd.index("--d") + 1])
        d_con = int(cmd[cmd.index("--d_con") + 1]) if "--d_con" in cmd else d_var
        runs = int(cmd[cmd.index("--runs") + 1])
        signature_obj = {
            "schema_version": 1,
            "type": "latency",
            "onnx": {
                "path": str(onnx_path),
                "exists": onnx_path.exists(),
                **({"size": onnx_path.stat().st_size, "mtime_ns": onnx_path.stat().st_mtime_ns} if onnx_path.exists() else {}),
            },
            "d_var": d_var,
            "d_con": d_con,
            "runs": runs,
        }
        wrapper_cmd = [
            *py_launch,
            str(script_dir / "run_output_task.py"),
            "--output",
            str(out_path),
            "--meta",
            str(Path(str(out_path) + ".meta.json")),
            "--signature-json",
            json.dumps(signature_obj, sort_keys=True),
            "--",
            *cmd,
        ]
        fh.write(
            json.dumps(
                {
                    "name": name,
                    "cmd": wrapper_cmd,
                    "cwd": os.getcwd(),
                    "log_path": str(out_path),
                    "skip": (not force) and completed(out_path, signature_obj),
                }
            )
            + "\n"
        )
PY

echo "[3/3] Benchmarking ONNX latency..."
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$latency_manifest" --jobs "$EXPORT_JOBS"

echo ""
echo "Calibration and export complete. Outputs in $MODEL_DIR and $RESULTS_DIR"
