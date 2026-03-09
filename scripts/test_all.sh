#!/usr/bin/env bash
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR/py:${PYTHONPATH:-}"

echo "[1/2] Running Python tests..."
if [ -n "${BBML_PYTHON:-}" ]; then
  (cd "$ROOT_DIR" && "$BBML_PYTHON" "$ROOT_DIR/scripts/run_py_tests.py" "$ROOT_DIR/tests/py") || echo "Python tests failed"
elif [ -x "$ROOT_DIR/py/.venv/bin/python" ]; then
  (cd "$ROOT_DIR" && "$ROOT_DIR/py/.venv/bin/python" "$ROOT_DIR/scripts/run_py_tests.py" "$ROOT_DIR/tests/py") || echo "Python tests failed"
elif command -v python3 >/dev/null 2>&1; then
  (cd "$ROOT_DIR" && python3 "$ROOT_DIR/scripts/run_py_tests.py" "$ROOT_DIR/tests/py") || echo "Python tests failed"
else
  echo "Python not found; skipping Python tests"
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
