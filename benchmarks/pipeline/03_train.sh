#!/usr/bin/env bash
# Step 03: Train supported deployable models only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BBML_ROOT="${BBML_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
. "$SCRIPT_DIR/common.sh"
bbml_resolve_python

DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
MODEL_DIR="$RESULTS_DIR/models"
mkdir -p "$MODEL_DIR"

TRAIN_PARQUET="$DATA_DIR/parquet/train.parquet"
GRAPH_TRAIN_MANIFEST="$DATA_DIR/manifests/graph/train.txt"

for path in "$TRAIN_PARQUET" "$GRAPH_TRAIN_MANIFEST"; do
  [ ! -f "$path" ] && echo "ERROR: missing required training artifact: $path" && exit 1
done

EPOCHS="${TRAIN_EPOCHS:-20}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
LR="${TRAIN_LR:-3e-4}"
HIDDEN="${TRAIN_HIDDEN:-64}"
DROPOUT="${TRAIN_DROPOUT:-0.1}"
SEED="${TRAIN_SEED:-0}"

echo "=== Model training ==="
echo "  Epochs    : $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR        : $LR"
echo "  Hidden    : $HIDDEN"
echo "  Dropout   : $DROPOUT"
echo ""

echo "[1/3] Training graph GNN from aggregate graph manifests..."
"${PYTHON_CMD[@]}" -m bbml.train.train_rank \
  --model gnn \
  --graph_manifest "$GRAPH_TRAIN_MANIFEST" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --dropout "$DROPOUT" \
  --seed "$SEED" \
  --metric loss \
  --ckpt_best "$MODEL_DIR/bbml_gnn_graph_best.pt"

echo "[2/3] Training var-only GNN from aggregate parquet..."
"${PYTHON_CMD[@]}" -m bbml.train.train_rank \
  --model gnn \
  --parquet "$TRAIN_PARQUET" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --dropout "$DROPOUT" \
  --seed "$SEED" \
  --metric loss \
  --ckpt_best "$MODEL_DIR/bbml_gnn_varonly_best.pt"

echo "[3/3] Training MLP from aggregate parquet..."
"${PYTHON_CMD[@]}" -m bbml.train.train_rank \
  --model mlp \
  --parquet "$TRAIN_PARQUET" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --dropout "$DROPOUT" \
  --seed "$SEED" \
  --metric loss \
  --ckpt_best "$MODEL_DIR/bbml_mlp_best.pt"

echo ""
echo "Training complete. Models in $MODEL_DIR"
