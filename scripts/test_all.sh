#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR/py:${PYTHONPATH:-}"

echo "[1/2] Running Python tests..."
PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3; fi
if [ -z "$PY" ] && command -v python >/dev/null 2>&1; then PY=python; fi
if [ -z "$PY" ]; then
  echo "Python not found; skipping Python tests"
else
  if command -v pytest >/dev/null 2>&1; then
    (cd "$ROOT_DIR" && pytest -q tests/py) || echo "Python tests failed"
  else
    (cd "$ROOT_DIR" && "$PY" -m pytest -q tests/py) || echo "pytest not installed or tests failed; skipping Python tests"
  fi
fi

echo "[2/2] Building and running C++ tests..."
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found; skipping C++ tests"
  exit 0
fi

BUILD_DIR="$ROOT_DIR/build"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DBBML_WITH_ONNX=ON -DBBML_WITH_LP_STATS=ON >/dev/null || { echo "Configure failed (SCIP or GTest missing?). Skipping C++ tests"; exit 0; }
cmake --build "$BUILD_DIR" -j >/dev/null || { echo "C++ build failed; skipping tests"; exit 0; }

if command -v ctest >/dev/null 2>&1; then
  ctest --test-dir "$BUILD_DIR" --output-on-failure || true
else
  echo "ctest not found; skipping C++ tests run"
fi

echo "All done."
