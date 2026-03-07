#!/usr/bin/env bash
# Step 03: Train the GNN ranker and XGBoost tabular baseline.
# Outputs checkpoints to $RESULTS_DIR/models/.
set -euo pipefail

BBML_ROOT="${BBML_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-$BBML_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$BBML_ROOT/results}"
MODEL_DIR="$RESULTS_DIR/models"
mkdir -p "$MODEL_DIR"

PY="uv run --project $BBML_ROOT/py python"
TRAIN_PARQUET="$DATA_DIR/parquet/train.parquet"
VAL_PARQUET="$DATA_DIR/parquet/val.parquet"

# ---- Check data exists ----
for f in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found. Run 02_convert.sh first."
    exit 1
  fi
done

# ---- Train full bipartite GNN ----
echo "[3.1] Training bipartite GNN ranker..."
$PY -m bbml.train.train_rank \
  --model gnn \
  --parquet "$TRAIN_PARQUET" \
  --epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --hidden 64 \
  --dropout 0.1 \
  --seed 0 \
  --metric loss \
  --ckpt "$MODEL_DIR/bbml_gnn_last.pt" \
  --ckpt_best "$MODEL_DIR/bbml_gnn_best.pt"

echo "  GNN checkpoint: $MODEL_DIR/bbml_gnn_best.pt"

# ---- Train var-only GNN (ablation A1) ----
echo "[3.2] Training var-only GNN (no constraint features, ablation A1)..."
$PY -m bbml.train.train_rank \
  --model gnn \
  --parquet "$TRAIN_PARQUET" \
  --epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --hidden 64 \
  --dropout 0.1 \
  --seed 0 \
  --metric loss \
  --ckpt_best "$MODEL_DIR/bbml_gnn_varonly_best.pt"

echo "  Var-only GNN checkpoint: $MODEL_DIR/bbml_gnn_varonly_best.pt"

# ---- Train tabular MLP baseline ----
echo "[3.3] Training tabular MLP ranker..."
$PY -m bbml.train.train_rank \
  --model mlp \
  --parquet "$TRAIN_PARQUET" \
  --epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --hidden 64 \
  --dropout 0.1 \
  --seed 0 \
  --metric loss \
  --ckpt_best "$MODEL_DIR/bbml_mlp_best.pt"

echo "  MLP checkpoint: $MODEL_DIR/bbml_mlp_best.pt"

# ---- Train XGBoost baseline ----
echo "[3.4] Training XGBoost tabular baseline..."
$PY - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["BBML_ROOT"] + "/py")
import pandas as pd
from bbml.train.baselines import BaselineConfig, fit_xgboost
import pickle

train_pq = os.environ["TRAIN_PARQUET"]
model_dir = os.environ["MODEL_DIR"]

feats = ["obj", "reduced_cost", "fracval", "domain_width",
         "pseudocost_up", "pseudocost_down", "pc_obs_up", "pc_obs_down",
         "is_binary", "is_integer"]
df = pd.read_parquet(train_pq)
cfg = BaselineConfig(features=feats, learning_rate=0.05, max_depth=6, max_iter=300)
bst = fit_xgboost(df, cfg)
out = os.path.join(model_dir, "bbml_xgb.pkl")
with open(out, "wb") as f:
    pickle.dump(bst, f)
print(f"XGBoost model saved to {out}")
PYEOF

echo "  XGBoost model: $MODEL_DIR/bbml_xgb.pkl"
echo ""
echo "[3] Training complete. Models in $MODEL_DIR"
