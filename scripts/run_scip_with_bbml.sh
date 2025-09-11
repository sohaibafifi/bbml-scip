#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <plugin_so> <instance.lp> [extra SCIP -c commands...]" >&2
  exit 1
fi

PLUGIN_SO="$1"; shift
INSTANCE="$1"; shift

# Resolve absolute paths
if [ ! -f "$PLUGIN_SO" ]; then
  echo "Plugin library not found: $PLUGIN_SO" >&2
  exit 1
fi
PLUGIN_SO_ABS="$(cd "$(dirname "$PLUGIN_SO")" && pwd)/$(basename "$PLUGIN_SO")"
INSTANCE_ABS="$(cd "$(dirname "$INSTANCE")" && pwd)/$(basename "$INSTANCE")"

SCIP_BIN="${SCIP_BIN:-scip}"
if ! command -v "$SCIP_BIN" >/dev/null 2>&1; then
  echo "scip binary not found (set SCIP_BIN)." >&2
  exit 1
fi

mkdir -p examples/out
echo "Running SCIP with plugin $PLUGIN_SO_ABS on instance $INSTANCE_ABS"
echo "Running SCIP with plugin $PLUGIN_SO_ABS on instance $INSTANCE_ABS"
"$SCIP_BIN" \
  -c "load plugins $PLUGIN_SO_ABS" \
  -c "read $INSTANCE_ABS" \
  "$@" \
  -c "optimize" \
  -c "quit"
