##  End-to-end examples for bb-ml

This folder shows minimal flows for MLP and GNN models.

## Prerequisites
- C++: CMake, a SCIP install (`SCIP_DIR`), optionally ONNX Runtime (`ONNXRUNTIME_DIR`)
- Python: `pip install -e ./py pytest`
- LP files (e.g., from MIPLIB or Netlib) in data/train and data/val

##  Makefile knobs
- `MODEL` = `mlp` or `gnn` (default: `mlp`)
- `GRAPH` = `1` to log graph snapshots and train a graph-input GNN

##  Common steps
1) Build the plugin
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
- The plugin library is built under `build/cpp/` as a module named `bbml_scip`.
- The example runner (`bbml_run`) prints a machine-readable summary line (nodes, time, gap, etc.).
- Runtime parameters (examples):
  - `set bbml/enable true`
  - `set bbml/telemetry true`
  - `set bbml/telemetry/graph true` and `set bbml/telemetry/graph_path "examples/out/train_graph.ndjson"` (for graph logging)
  - `set bbml/model_path examples/out/score_<model>.onnx`
  - `set bbml/confidence 0.6`
  - `set bbml/reload 1`
