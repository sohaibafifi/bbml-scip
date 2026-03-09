#!/usr/bin/env bash
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 <instance.lp> [extra name=value params...]" >&2
  exit 1
fi

INSTANCE="$1"; shift
INSTANCE_ABS="$(cd "$(dirname "$INSTANCE")" && pwd)/$(basename "$INSTANCE")"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BBML_ROOT="$ROOT_DIR"
. "$ROOT_DIR/benchmarks/pipeline/common.sh"
bbml_require_runner
RUNNER="$BBML_RUNNER_BIN"

mkdir -p examples/out
echo "Running bbml_run on instance $INSTANCE_ABS"
CMD=("$RUNNER" --problem "$INSTANCE_ABS")
for extra in "$@"; do
  CMD+=(--param "$extra")
done
"${CMD[@]}"
