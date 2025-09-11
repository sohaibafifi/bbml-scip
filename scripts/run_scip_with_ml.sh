#!/usr/bin/env bash
set -euo pipefail
SCIP_DIR=${SCIP_DIR:-/opt/scip}
cmake -S . -B build -DSCIP_DIR=$SCIP_DIR -DBBML_WITH_ONNX=OFF
cmake --build build -j
# TODO: load bbml_scip module into SCIP
