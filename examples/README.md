##  End-to-end examples for bb-ml

This folder shows minimal flows for MLP and GNN models.

## Prerequisites
- C++: CMake, a SCIP install (`SCIP_DIR`), optionally ONNX Runtime (`ONNXRUNTIME_DIR`)
- Python: `pip install -e ./py pytest`
- LP files (e.g., from MIPLIB or Netlib) in data/train and data/val

## Build entrypoint
- Build from the repository root with CMake or from this folder with `make build`.
- Do not run `cmake ..` inside `examples/`; this folder is a demo workspace with its own checked-in `Makefile`, not a standalone CMake project.
- For ONNX support, either set `ONNXRUNTIME_DIR` explicitly or place an unpacked ONNX Runtime release under `libs/onnxruntime-*`.

##  Makefile knobs
- `MODEL` = `mlp` or `gnn` (default: `mlp`)
- `GRAPH` = `1` to log graph snapshots and train a graph-input GNN

##  Common steps
1) Build the runner
   - `make -C .. build` or from this folder: `make build`
   - Set `SCIP_DIR` to your SCIP install prefix.
   - Optional ONNX: set `ONNXRUNTIME_DIR` to your ORT install (with include/ and lib/).

2) Collect telemetry
   - MLP/var-only: `make scip-log` writes `examples/out/train.ndjson`
   - Graph snapshots: `make scip-log GRAPH=1` writes `examples/out/train_graph.ndjson`

3) Train and export
   - MLP from Parquet: `make parquet && make train-export`
   - Var-only GNN from Parquet: `make parquet && make MODEL=gnn train-export`
   - Graph GNN from NDJSON: `make MODEL=gnn GRAPH=1 train-export`

4) Run SCIP with the exported model
   - `make scip-ml` (uses `examples/out/score_<MODEL>.onnx`)

##  Notes
- The supported example entrypoint is `build/bbml_run`; it already includes the BBML plugins.
- The example runner prints a machine-readable `BBML_SUMMARY` line (nodes, time, gap, etc.).
- Configure runtime behavior through `.set` files or `--param name=value`, for example:
  - `bbml/enable = TRUE`
  - `bbml/telemetry = TRUE`
  - `bbml/telemetry/graph = TRUE`
  - `bbml/telemetry/graph_path = "examples/out/train_graph.ndjson"`
  - `bbml/model_path = "examples/out/score_<model>.onnx"`
  - `bbml/confidence = 0.6`
  - `bbml/reload = 1`
- A stock `scip` binary will reject `bbml/*` parameters unless you built those plugins into that exact executable.
