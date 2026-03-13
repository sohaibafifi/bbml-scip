#!/usr/bin/env bash
# Step 03: Train supported deployable models only.
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
mkdir -p "$MODEL_DIR"
TRAIN_LOG_DIR="$DATA_DIR/pipeline_logs/train"
TRAIN_MANIFEST_DIR="$DATA_DIR/manifests"
mkdir -p "$TRAIN_LOG_DIR" "$TRAIN_MANIFEST_DIR"

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
ENSEMBLE_SIZE="${TRAIN_ENSEMBLE_SIZE:-3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-$(bbml_detect_torch_device)}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:--1}"
TRAIN_PIN_MEMORY="${TRAIN_PIN_MEMORY:--1}"
TRAIN_FORCE="${TRAIN_FORCE:-0}"

if [ -n "${TRAIN_JOBS:-}" ]; then
  TRAIN_JOBS_VALUE="$TRAIN_JOBS"
elif [ "$TRAIN_DEVICE" = "cpu" ]; then
  TRAIN_JOBS_VALUE="$(bbml_default_train_jobs)"
else
  TRAIN_JOBS_VALUE=1
fi

if [ "$ENSEMBLE_SIZE" -lt 1 ]; then
  echo "ERROR: TRAIN_ENSEMBLE_SIZE must be >= 1"
  exit 1
fi

if [ "$TRAIN_JOBS_VALUE" -lt 1 ]; then
  echo "ERROR: TRAIN_JOBS must be >= 1"
  exit 1
fi

if [ "$TRAIN_JOBS_VALUE" -gt 1 ] && [ "$TRAIN_DEVICE" != "cpu" ]; then
  echo "WARNING: TRAIN_JOBS=$TRAIN_JOBS_VALUE with device=$TRAIN_DEVICE may be slower due to device contention."
fi

export EPOCHS BATCH_SIZE LR HIDDEN DROPOUT TRAIN_DEVICE TRAIN_LOG_EVERY TRAIN_NUM_WORKERS TRAIN_PIN_MEMORY

PYTHON_CMD_JSON="$(bbml_python_json_array)"

ckpt_meta_path() {
  printf '%s.trainmeta.json' "$1"
}

build_train_signature() {
  local model_kind="$1"
  local seed_value="$2"
  local mode="$3"
  local source_path="$4"
  python3 - "$model_kind" "$seed_value" "$mode" "$source_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_kind, seed_value, mode, source_path = sys.argv[1:]
source = Path(source_path)

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
    "model": model_kind,
    "seed": int(seed_value),
    "epochs": int(__import__("os").environ["EPOCHS"]),
    "batch_size": int(__import__("os").environ["BATCH_SIZE"]),
    "lr": __import__("os").environ["LR"],
    "hidden": int(__import__("os").environ["HIDDEN"]),
    "dropout": __import__("os").environ["DROPOUT"],
    "source_mode": mode,
}
if mode == "graph_manifest":
    payload["graph_manifest"] = stat_payload(source)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256()
    for line in lines:
        digest.update(json.dumps(stat_payload(Path(line)), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    payload["graph_sources_count"] = len(lines)
    payload["graph_sources_digest"] = digest.hexdigest()
else:
    payload["parquet"] = stat_payload(source)
print(json.dumps(payload, sort_keys=True))
PY
}

train_task_complete() {
  local ckpt_path="$1"
  local signature_json="$2"
  python3 - "$ckpt_path" "$signature_json" <<'PY'
import json
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
sig = json.loads(sys.argv[2])
meta = Path(str(ckpt) + ".trainmeta.json")
if not ckpt.is_file() or ckpt.stat().st_size <= 0:
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

append_train_task() {
  local manifest="$1"
  local name="$2"
  local log_path="$3"
  local ckpt_path="$4"
  local model_kind="$5"
  local seed_value="$6"
  local skip_value="$7"
  local signature_json="$8"
  shift 8

  TRAIN_TASK_PYTHON_CMD_JSON="$PYTHON_CMD_JSON" python3 - "$manifest" "$name" "$log_path" "$BBML_ROOT" "$ckpt_path" "$model_kind" "$seed_value" "$skip_value" "$signature_json" "$@" <<'PY'
import json
import os
import sys

manifest, name, log_path, cwd, ckpt_path, model_kind, seed_value, skip_value, signature_json, *extra_args = sys.argv[1:]
cmd = json.loads(os.environ["TRAIN_TASK_PYTHON_CMD_JSON"])
inner_cmd = [
    *cmd,
    "-m",
    "bbml.train.train_rank",
    "--model",
    model_kind,
    "--epochs",
    os.environ["EPOCHS"],
    "--batch_size",
    os.environ["BATCH_SIZE"],
    "--lr",
    os.environ["LR"],
    "--hidden",
    os.environ["HIDDEN"],
    "--dropout",
    os.environ["DROPOUT"],
    "--device",
    os.environ["TRAIN_DEVICE"],
    "--seed",
    seed_value,
    "--metric",
    "loss",
    "--log_every",
    os.environ["TRAIN_LOG_EVERY"],
    "--num_workers",
    os.environ["TRAIN_NUM_WORKERS"],
    "--pin_memory",
    os.environ["TRAIN_PIN_MEMORY"],
    "--ckpt_best",
    ckpt_path,
]
inner_cmd.extend(extra_args)
wrapper_cmd = [
    *cmd,
    os.path.join(cwd, "benchmarks", "pipeline", "run_train_task.py"),
    "--ckpt",
    ckpt_path,
    "--meta",
    ckpt_path + ".trainmeta.json",
    "--signature-json",
    signature_json,
    "--",
]
wrapper_cmd.extend(inner_cmd)
with open(manifest, "a", encoding="utf-8") as fh:
    fh.write(
        json.dumps(
            {
                "name": name,
                "cmd": wrapper_cmd,
                "cwd": cwd,
                "log_path": log_path,
                "env": {},
                "skip": skip_value == "1",
            }
        )
        + "\n"
    )
PY
}

run_train_command() {
  local signature_json="$1"
  local ckpt_path="$2"
  shift 2
  "${PYTHON_CMD[@]}" "$SCRIPT_DIR/run_train_task.py" \
    --ckpt "$ckpt_path" \
    --meta "$(ckpt_meta_path "$ckpt_path")" \
    --signature-json "$signature_json" \
    -- \
    "${PYTHON_CMD[@]}" "$@"
}

run_train_phase_parallel() {
  local label="$1"
  local manifest="$2"
  local phase_jobs="$3"
  echo "$label"
  "${PYTHON_CMD[@]}" "$SCRIPT_DIR/task_runner.py" --manifest "$manifest" --jobs "$phase_jobs"
}

echo "=== Model training ==="
echo "  Epochs    : $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR        : $LR"
echo "  Hidden    : $HIDDEN"
echo "  Dropout   : $DROPOUT"
echo "  Device    : $TRAIN_DEVICE"
echo "  Ensemble  : $ENSEMBLE_SIZE graph checkpoints"
echo "  Train jobs: $TRAIN_JOBS_VALUE"
echo "  Workers   : $TRAIN_NUM_WORKERS"
echo "  Pin memory: $TRAIN_PIN_MEMORY"
echo "  Resume    : $( [ "$TRAIN_FORCE" = "1" ] && printf 'off (TRAIN_FORCE=1)' || printf 'on' )"
echo ""

if [ "$TRAIN_JOBS_VALUE" -le 1 ]; then
  echo "[1/3] Training graph GNN ensemble from aggregate graph manifests..."
  for member_idx in $(seq 0 $((ENSEMBLE_SIZE - 1))); do
    member_seed=$((SEED + member_idx))
    member_ckpt="$MODEL_DIR/bbml_gnn_graph_member${member_idx}_best.pt"
    if [ "$member_idx" -eq 0 ]; then
      member_ckpt="$MODEL_DIR/bbml_gnn_graph_best.pt"
    fi
    member_sig="$(build_train_signature "gnn" "$member_seed" "graph_manifest" "$GRAPH_TRAIN_MANIFEST")"
    if [ "$TRAIN_FORCE" != "1" ] && train_task_complete "$member_ckpt" "$member_sig"; then
      echo "  - graph member $member_idx (seed=$member_seed) already trained -> $member_ckpt"
      continue
    fi
    echo "  - graph member $member_idx (seed=$member_seed) -> $member_ckpt"
    run_train_command "$member_sig" "$member_ckpt" \
      -m bbml.train.train_rank \
      --model gnn \
      --graph_manifest "$GRAPH_TRAIN_MANIFEST" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --hidden "$HIDDEN" \
      --dropout "$DROPOUT" \
      --device "$TRAIN_DEVICE" \
      --seed "$member_seed" \
      --metric loss \
      --log_every "$TRAIN_LOG_EVERY" \
      --num_workers "$TRAIN_NUM_WORKERS" \
      --pin_memory "$TRAIN_PIN_MEMORY" \
      --ckpt_best "$member_ckpt"
  done

  echo "[2/3] Training var-only GNN from aggregate parquet..."
  varonly_sig="$(build_train_signature "gnn" "$SEED" "parquet" "$TRAIN_PARQUET")"
  if [ "$TRAIN_FORCE" != "1" ] && train_task_complete "$MODEL_DIR/bbml_gnn_varonly_best.pt" "$varonly_sig"; then
    echo "  - var-only GNN already trained -> $MODEL_DIR/bbml_gnn_varonly_best.pt"
  else
    run_train_command "$varonly_sig" "$MODEL_DIR/bbml_gnn_varonly_best.pt" \
      -m bbml.train.train_rank \
      --model gnn \
      --parquet "$TRAIN_PARQUET" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --hidden "$HIDDEN" \
      --dropout "$DROPOUT" \
      --device "$TRAIN_DEVICE" \
      --seed "$SEED" \
      --metric loss \
      --log_every "$TRAIN_LOG_EVERY" \
      --num_workers "$TRAIN_NUM_WORKERS" \
      --pin_memory "$TRAIN_PIN_MEMORY" \
      --ckpt_best "$MODEL_DIR/bbml_gnn_varonly_best.pt"
  fi

  echo "[3/3] Training MLP from aggregate parquet..."
  mlp_sig="$(build_train_signature "mlp" "$SEED" "parquet" "$TRAIN_PARQUET")"
  if [ "$TRAIN_FORCE" != "1" ] && train_task_complete "$MODEL_DIR/bbml_mlp_best.pt" "$mlp_sig"; then
    echo "  - MLP already trained -> $MODEL_DIR/bbml_mlp_best.pt"
  else
    run_train_command "$mlp_sig" "$MODEL_DIR/bbml_mlp_best.pt" \
      -m bbml.train.train_rank \
      --model mlp \
      --parquet "$TRAIN_PARQUET" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --hidden "$HIDDEN" \
      --dropout "$DROPOUT" \
      --device "$TRAIN_DEVICE" \
      --seed "$SEED" \
      --metric loss \
      --log_every "$TRAIN_LOG_EVERY" \
      --num_workers "$TRAIN_NUM_WORKERS" \
      --pin_memory "$TRAIN_PIN_MEMORY" \
      --ckpt_best "$MODEL_DIR/bbml_mlp_best.pt"
  fi
else
  GRAPH_MANIFEST="$TRAIN_MANIFEST_DIR/train_graph_tasks.jsonl"
  TABULAR_MANIFEST="$TRAIN_MANIFEST_DIR/train_tabular_tasks.jsonl"
  : > "$GRAPH_MANIFEST"
  : > "$TABULAR_MANIFEST"

  echo "[1/3] Training graph GNN ensemble from aggregate graph manifests..."
  for member_idx in $(seq 0 $((ENSEMBLE_SIZE - 1))); do
    member_seed=$((SEED + member_idx))
    member_ckpt="$MODEL_DIR/bbml_gnn_graph_member${member_idx}_best.pt"
    if [ "$member_idx" -eq 0 ]; then
      member_ckpt="$MODEL_DIR/bbml_gnn_graph_best.pt"
    fi
    member_log="$TRAIN_LOG_DIR/bbml_gnn_graph_member${member_idx}.log"
    echo "  - graph member $member_idx (seed=$member_seed) -> $member_ckpt"
    member_sig="$(build_train_signature "gnn" "$member_seed" "graph_manifest" "$GRAPH_TRAIN_MANIFEST")"
    skip_task=0
    if [ "$TRAIN_FORCE" != "1" ] && train_task_complete "$member_ckpt" "$member_sig"; then
      skip_task=1
      echo "    resume: skipping existing checkpoint"
    fi
    append_train_task \
      "$GRAPH_MANIFEST" \
      "train:graph:member${member_idx}" \
      "$member_log" \
      "$member_ckpt" \
      "gnn" \
      "$member_seed" \
      "$skip_task" \
      "$member_sig" \
      --graph_manifest "$GRAPH_TRAIN_MANIFEST"
  done
  graph_jobs="$TRAIN_JOBS_VALUE"
  if [ "$graph_jobs" -gt "$ENSEMBLE_SIZE" ]; then
    graph_jobs="$ENSEMBLE_SIZE"
  fi
  run_train_phase_parallel "[parallel] graph logs -> $TRAIN_LOG_DIR" "$GRAPH_MANIFEST" "$graph_jobs"

  echo "[2/3] Training var-only GNN and MLP from aggregate parquet..."
  varonly_sig="$(build_train_signature "gnn" "$SEED" "parquet" "$TRAIN_PARQUET")"
  append_train_task \
    "$TABULAR_MANIFEST" \
    "train:gnn:varonly" \
    "$TRAIN_LOG_DIR/bbml_gnn_varonly.log" \
    "$MODEL_DIR/bbml_gnn_varonly_best.pt" \
    "gnn" \
    "$SEED" \
    "$( [ "$TRAIN_FORCE" != "1" ] && train_task_complete "$MODEL_DIR/bbml_gnn_varonly_best.pt" "$varonly_sig" && printf '1' || printf '0' )" \
    "$varonly_sig" \
    --parquet "$TRAIN_PARQUET"
  mlp_sig="$(build_train_signature "mlp" "$SEED" "parquet" "$TRAIN_PARQUET")"
  append_train_task \
    "$TABULAR_MANIFEST" \
    "train:mlp" \
    "$TRAIN_LOG_DIR/bbml_mlp.log" \
    "$MODEL_DIR/bbml_mlp_best.pt" \
    "mlp" \
    "$SEED" \
    "$( [ "$TRAIN_FORCE" != "1" ] && train_task_complete "$MODEL_DIR/bbml_mlp_best.pt" "$mlp_sig" && printf '1' || printf '0' )" \
    "$mlp_sig" \
    --parquet "$TRAIN_PARQUET"
  tabular_jobs="$TRAIN_JOBS_VALUE"
  if [ "$tabular_jobs" -gt 2 ]; then
    tabular_jobs=2
  fi
  run_train_phase_parallel "[parallel] tabular logs -> $TRAIN_LOG_DIR" "$TABULAR_MANIFEST" "$tabular_jobs"
fi

echo ""
echo "Training complete. Models in $MODEL_DIR"
