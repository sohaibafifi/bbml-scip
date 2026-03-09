#!/usr/bin/env bash
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${BBML_PYTHON:-}"
if [ -z "$PY" ] && [ -x "$ROOT_DIR/py/.venv/bin/python" ]; then PY="$ROOT_DIR/py/.venv/bin/python"; fi
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1; then PY=python3; fi
if [ -z "$PY" ] && command -v python >/dev/null 2>&1; then PY=python; fi
if [ -z "$PY" ]; then
  echo "Python not found." >&2
  exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 <mlp|gnn> <train.parquet|graph_manifest.txt> [extra train_rank args...]" >&2
  exit 1
fi

MODEL="$1"; shift
DATA_ARG="$1"; shift

CMD=("$PY" -m bbml.train.train_rank --model "$MODEL")
if [ "$MODEL" = "gnn" ] && [[ "$DATA_ARG" == *.txt ]]; then
  CMD+=(--graph_manifest "$DATA_ARG")
else
  CMD+=(--parquet "$DATA_ARG")
fi

CMD+=("$@")
PYTHONPATH="$ROOT_DIR/py:${PYTHONPATH:-}" "${CMD[@]}"
