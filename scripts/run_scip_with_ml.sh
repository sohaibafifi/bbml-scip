#!/usr/bin/env bash
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 <instance.lp|instance.mps> [model.onnx] [extra name=value params...]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE="$1"; shift
MODEL_PATH="${1:-}"
if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
  shift
fi

BBML_ROOT="$ROOT_DIR"
. "$ROOT_DIR/benchmarks/pipeline/common.sh"
bbml_require_runner
RUNNER="$BBML_RUNNER_BIN"

CMD=(
  "$RUNNER"
  --problem "$INSTANCE"
  --param "bbml/telemetry=FALSE"
)

if [ -n "$MODEL_PATH" ]; then
  CMD+=(--param "bbml/enable=TRUE")
  CMD+=(--param "bbml/model_path=$MODEL_PATH")
else
  CMD+=(--param "bbml/enable=FALSE")
fi

for extra in "$@"; do
  CMD+=(--param "$extra")
done
"${CMD[@]}"
