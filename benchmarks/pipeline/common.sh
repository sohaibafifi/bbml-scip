#!/usr/bin/env bash

bbml_cpu_count() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  echo 1
}

bbml_default_generate_jobs() {
  local cpu
  cpu="$(bbml_cpu_count)"
  if [ "$cpu" -lt 4 ]; then
    echo "$cpu"
  else
    echo 4
  fi
}

bbml_default_solver_jobs() {
  local cpu half
  cpu="$(bbml_cpu_count)"
  half=$((cpu / 2))
  if [ "$half" -lt 1 ]; then
    half=1
  fi
  if [ "$half" -gt 4 ]; then
    half=4
  fi
  echo "$half"
}

bbml_default_train_jobs() {
  local cpu
  cpu="$(bbml_cpu_count)"
  if [ "$cpu" -lt 2 ]; then
    echo 1
  elif [ "$cpu" -gt 2 ]; then
    echo 2
  else
    echo "$cpu"
  fi
}

bbml_export_repo_pythonpath() {
  local repo_py
  repo_py="$BBML_ROOT/py"
  case ":${PYTHONPATH:-}:" in
    *":$repo_py:"*) ;;
    *)
      if [ -n "${PYTHONPATH:-}" ]; then
        export PYTHONPATH="$repo_py:$PYTHONPATH"
      else
        export PYTHONPATH="$repo_py"
      fi
      ;;
  esac
}

bbml_resolve_python() {
  if [ -n "${BBML_PYTHON:-}" ]; then
    PYTHON_CMD=("$BBML_PYTHON")
  elif [ -x "$BBML_ROOT/py/.venv/bin/python" ]; then
    PYTHON_CMD=("$BBML_ROOT/py/.venv/bin/python")
  else
    PYTHON_CMD=(uv run --project "$BBML_ROOT/py" python)
  fi
  bbml_export_repo_pythonpath
}

bbml_detect_torch_device() {
  local detected
  detected="$("${PYTHON_CMD[@]}" - <<'PY' 2>/dev/null || true
import importlib.util

spec = importlib.util.find_spec("torch")
if spec is None:
    print("cpu")
else:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
  if [ -z "$detected" ]; then
    echo "cpu"
  else
    echo "$detected"
  fi
}

bbml_resolve_runner() {
  if [ -n "${BBML_RUNNER:-}" ] && [ -x "${BBML_RUNNER}" ]; then
    BBML_RUNNER_BIN="${BBML_RUNNER}"
    return
  fi
  if [ -x "$BBML_ROOT/build/bbml_run" ]; then
    BBML_RUNNER_BIN="$BBML_ROOT/build/bbml_run"
    return
  fi
  if [ -x "$BBML_ROOT/build/cpp-tests/bbml_run" ]; then
    BBML_RUNNER_BIN="$BBML_ROOT/build/cpp-tests/bbml_run"
    return
  fi
  BBML_RUNNER_BIN=""
}

bbml_find_probe_instance() {
  if [ -n "${BBML_PROBE_INSTANCE:-}" ] && [ -f "${BBML_PROBE_INSTANCE}" ]; then
    printf '%s\n' "$BBML_PROBE_INSTANCE"
    return
  fi

  local candidate
  for candidate in \
    "$BBML_ROOT/examples/data/branching.lp" \
    "$BBML_ROOT/examples/data/toy.lp"
  do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  candidate="$(
    find "$BBML_ROOT/data/instances" "$BBML_ROOT/examples/data" -type f \
      \( -name '*.lp' -o -name '*.lp.gz' -o -name '*.mps' -o -name '*.mps.gz' \) 2>/dev/null \
      | sort | sed -n '1p'
  )"
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    printf '%s\n' "$candidate"
    return
  fi

  local list_file inst
  while IFS= read -r list_file; do
    [ -z "$list_file" ] && continue
    while IFS= read -r inst; do
      if [ -n "$inst" ] && [ -f "$inst" ]; then
        printf '%s\n' "$inst"
        return
      fi
    done < "$list_file"
  done < <(find "$BBML_ROOT/benchmarks/instances" -maxdepth 1 -name '*_*.txt' 2>/dev/null | sort)
}

bbml_verify_runner() {
  if [ -z "${BBML_RUNNER_BIN:-}" ]; then
    echo "ERROR: BBML_RUNNER_BIN is empty." >&2
    exit 1
  fi
  if [ "${BBML_RUNNER_CHECKED:-}" = "$BBML_RUNNER_BIN" ]; then
    return
  fi
  local probe_instance
  probe_instance="$(bbml_find_probe_instance)"
  if [ -z "$probe_instance" ]; then
    echo "ERROR: no probe instance found." >&2
    echo "Set BBML_PROBE_INSTANCE to any .lp/.mps instance file, or generate instances first." >&2
    exit 1
  fi

  local probe_out probe_rc
  probe_out="$("$BBML_RUNNER_BIN" \
    --problem "$probe_instance" \
    --param 'bbml/enable=FALSE' \
    --param 'bbml/telemetry=FALSE' \
    --param 'limits/time=1' \
    2>&1)" && probe_rc=0 || probe_rc=$?

  if [ "$probe_rc" -ne 0 ] || ! printf '%s' "$probe_out" | grep -q 'BBML_SUMMARY'; then
    echo "ERROR: $BBML_RUNNER_BIN is not a BBML-enabled bbml_run executable." >&2
    echo "Set BBML_RUNNER to the built runner, for example:" >&2
    echo "  export BBML_RUNNER=\"$BBML_ROOT/build/bbml_run\"" >&2
    echo "Probe output:" >&2
    printf '%s\n' "$probe_out" | sed -n '1,20p' >&2
    exit 1
  fi

  BBML_RUNNER_CHECKED="$BBML_RUNNER_BIN"
}

bbml_require_runner() {
  bbml_resolve_runner
  if [ -z "${BBML_RUNNER_BIN:-}" ]; then
    echo "ERROR: bbml_run not found. Build the project first or set BBML_RUNNER." >&2
    exit 1
  fi
  bbml_verify_runner
}

bbml_python() {
  "${PYTHON_CMD[@]}" "$@"
}

bbml_python_json_array() {
  python3 - "${PYTHON_CMD[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

bbml_abs_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
}

bbml_nonempty_file() {
  [ -f "$1" ] && [ -s "$1" ]
}
