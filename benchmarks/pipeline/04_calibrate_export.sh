#!/usr/bin/env bash
# Step 04: Temperature calibration + ONNX export.
# Reads best GNN checkpoint, calibrates T*, exports FP32 and FP16 ONNX models.
set -euo pipefail

BBML_ROOT="${BBML_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
MODEL_DIR="$RESULTS_DIR/models"

PY="uv run --project $BBML_ROOT/py python"
VAL_PARQUET="$DATA_DIR/parquet/val.parquet"
GNN_CKPT="$MODEL_DIR/bbml_gnn_best.pt"
MLP_CKPT="$MODEL_DIR/bbml_mlp_best.pt"

for f in "$VAL_PARQUET" "$GNN_CKPT"; do
  [ ! -f "$f" ] && echo "ERROR: $f not found." && exit 1
done

# ---- Temperature calibration ----
echo "[4.1] Fitting temperature scalar on validation set..."
TEMP_OUT="$MODEL_DIR/temperature.txt"
$PY -m bbml.train.calibrate \
  --parquet "$VAL_PARQUET" \
  --hidden 64 \
  --dropout 0.1 \
  --device cpu \
  2>&1 | tee /tmp/calibrate_out.txt

T_STAR=$(grep -oP 'T=\K[0-9.]+' /tmp/calibrate_out.txt | tail -1)
echo "$T_STAR" > "$TEMP_OUT"
echo "  T* = $T_STAR -> $TEMP_OUT"

# ---- Export GNN FP32 (full graph signature) ----
echo "[4.2] Exporting GNN FP32 ONNX (full graph)..."
$PY -m bbml.export.export_onnx \
  --model gnn \
  --ckpt "$GNN_CKPT" \
  --d_var 32 \
  --d_con 32 \
  --hidden 64 \
  --layers 3 \
  --graph_inputs \
  --out "$MODEL_DIR/bbml_gnn_fp32.onnx"

# ---- Export GNN FP16 ----
echo "[4.3] Exporting GNN FP16 ONNX..."
$PY -m bbml.export.export_onnx \
  --model gnn \
  --ckpt "$GNN_CKPT" \
  --d_var 32 \
  --d_con 32 \
  --hidden 64 \
  --layers 3 \
  --graph_inputs \
  --fp16 \
  --out "$MODEL_DIR/bbml_gnn_fp16.onnx"

# ---- Export GNN var-only FP32 (ablation A1) ----
echo "[4.4] Exporting var-only GNN FP32..."
$PY -m bbml.export.export_onnx \
  --model gnn \
  --ckpt "$MODEL_DIR/bbml_gnn_varonly_best.pt" \
  --d_var 32 \
  --d_con 32 \
  --hidden 64 \
  --layers 3 \
  --out "$MODEL_DIR/bbml_gnn_varonly.onnx"

# ---- Export MLP FP32 ----
echo "[4.5] Exporting tabular MLP ONNX..."
$PY -m bbml.export.export_onnx \
  --model mlp \
  --ckpt "$MLP_CKPT" \
  --d_in 6 \
  --hidden 64 \
  --out "$MODEL_DIR/bbml_mlp.onnx"

# Create default symlink used by bbml-full
ln -sf bbml_gnn_fp16.onnx "$MODEL_DIR/bbml_gnn.onnx"

# ---- Latency benchmark ----
echo "[4.6] Running ONNX latency benchmark..."
$PY -m bbml.bench.latency \
  --onnx "$MODEL_DIR/bbml_gnn_fp32.onnx" \
  --d 32 \
  --dims 100 250 500 1000 1500 2000 \
  --runs 100 \
  | tee "$RESULTS_DIR/latency_fp32.txt"

$PY -m bbml.bench.latency \
  --onnx "$MODEL_DIR/bbml_gnn_fp16.onnx" \
  --d 32 \
  --dims 100 250 500 1000 1500 2000 \
  --runs 100 \
  | tee "$RESULTS_DIR/latency_fp16.txt"

echo ""
echo "[4] Calibration + export complete."
ls -lh "$MODEL_DIR"/*.onnx "$TEMP_OUT" 2>/dev/null
