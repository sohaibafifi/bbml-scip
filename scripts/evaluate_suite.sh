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

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
INSTANCES_DIR="${INSTANCES_DIR:-$ROOT_DIR/benchmarks/instances}"
OUT_DIR="${EVAL_OUT_DIR:-$RESULTS_DIR/eval}"
mkdir -p "$OUT_DIR"

PYTHONPATH="$ROOT_DIR/py:${PYTHONPATH:-}" "$PY" "$ROOT_DIR/benchmarks/eval/kpis.py" \
  --results "$RESULTS_DIR/runs" \
  --instance-sets "$INSTANCES_DIR" \
  --out "$OUT_DIR/kpis.csv"

PYTHONPATH="$ROOT_DIR/py:${PYTHONPATH:-}" "$PY" "$ROOT_DIR/benchmarks/eval/summary_table.py" \
  --kpis "$OUT_DIR/kpis.csv" \
  --out "$OUT_DIR/tables"

PYTHONPATH="$ROOT_DIR/py:${PYTHONPATH:-}" "$PY" "$ROOT_DIR/benchmarks/eval/perf_profile.py" \
  --results "$RESULTS_DIR/runs" \
  --out "$OUT_DIR/figures"
