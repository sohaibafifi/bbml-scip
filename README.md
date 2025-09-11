# bb-ml

A SCIP plugin and Python package for ML-assisted branching.


## Main idea
- Collect per‑candidate telemetry during solving (features at each LP branching decision).
- Convert telemetry to a columnar dataset and train a ranker offline (MLP or GNN).
- Export to ONNX and blend its scores with SCIP’s default heuristic inside a custom branchrule.

## Algorithm overview
- At each LP branching decision the plugin extracts features for each branching candidate (variable) describing local LP state, constraint structure, and solver signals.
- An offline-trained ML ranker (e.g., an MLP or GNN) scores candidates; the model is trained to prefer candidates that historically led to smaller search trees or faster solves.
- At runtime the branchrule queries the ONNX model (or a safe fallback) and computes a blended score: a weighted combination of SCIP's native heuristic score and the ML score. Blending respects runtime controls such as `bbml/confidence`, `bbml/temperature` (score scaling), and depth penalties so the influence of ML can vary by depth or confidence.
- The branchrule then selects the candidate with the highest blended score. If ONNX or a model path is unavailable, the plugin falls back gracefully to SCIP-only behavior (ML contribution = 0).

## Key points
- Data-driven ranking: the model learns a preference ordering over candidates rather than predicting absolute numeric outcomes, which is well‑suited to the branching decision.
- Offline training: telemetry → Parquet → train → export ONNX; this separates heavy ML work from solver runtime.
- Safe integration: numeric and confidence gates prevent ML from destabilizing solves; hot‑reload and parameter tuning allow iterative deployment.

## Quick start
- Build C++ (requires `SCIP_DIR`): `cmake -S . -B build && cmake --build build`
  - Graph extraction auto-enables when SCIP LP headers are available (`scip/lp.h`, `scip/lpi.h`); otherwise it falls back to var-only features.
- Python package: `pip install -e ./py`
- Example pipeline (recommended): see `examples/` below.

## Examples
- Makefile knobs:
  - `MODEL` = `mlp` or `gnn` (default: `mlp`)
  - `GRAPH` = `1` to log graph snapshots and train a graph-input GNN
- Single‑instance demo (MLP): `make -C examples pipeline`
  - Logs → Parquet → train MLP → export ONNX → run solver with ML blending.
- Train/Val split + benchmark (MLP or var-only GNN):
  - `make -C examples scip-log-train parquet-train`
  - `make -C examples MODEL=mlp train-export` (or `MODEL=gnn` for var-only)
  - `make -C examples scip-log-val parquet-val`
  - `make -C examples benchmark`
  - Baseline: `make -C examples benchmark-nobbml`
  - Compare: `make -C examples benchmark-compare`
- Full graph GNN:
  - Collect graphs: `make -C examples scip-log-train GRAPH=1`
  - Train + export: `make -C examples MODEL=gnn GRAPH=1 train-export`
  - Run with GNN graph ONNX: `make -C examples MODEL=gnn GRAPH=1 scip-ml`

## Tuning solve limits for examples
- All example targets accept two variables to bound solve effort consistently:
  - `TIME` (seconds, default 60)
  - `NODES` (max B&B nodes, default 1000)
- Override on the command line, for example:
  - `make -C examples scip-log-train TIME=30 NODES=500`
  - `make -C examples benchmark-compare TIME=120 NODES=2000`

## Folders
- `cpp/`: SCIP plugins (branchrule, nodesel) and feature extraction
- `py/bbml/`: ML models, data loaders, training, export
- `configs/`: solver and model configs
- `data/`: logs and derived datasets (Parquet)
- `results/`: models and reports
 - `examples/`: ready‑to‑run LPs (train/val), Makefile targets, and a runner that includes the plugins

## Tests
- Run both Python and C++ tests: `make test` or `./scripts/test_all.sh`
- Python only: `make test-py`
- C++ only: `make test-cpp` (requires CMake and GTest)

## SCIP Runtime Parameters (bbml/*)
- `bbml/enable`: enable ML-assisted branching (default: true)
- `bbml/telemetry`: enable branch-time telemetry logging (default: true)
- `bbml/telemetry/path`: telemetry NDJSON path (string, quoted)
- `bbml/telemetry/append`: append vs truncate on first open (bool)
- `bbml/telemetry/strongbranch`: log strong‑branch up/down scores (expensive)
- `bbml/telemetry/graph`: log graph snapshots (var_feat, con_feat, edge_index) per node
- `bbml/telemetry/graph_path`: path to graph NDJSON output
- `bbml/model_path`: path to ONNX model (string, quoted)
- `bbml/reload`: increment to hot-reload model (int)
- `bbml/alpha/min,max,depth_penalty`: blending controls
- `bbml/confidence`: base confidence in ML scores [0..1]
- `bbml/temperature`: scale ML scores by 1/T (T>0)
- `bbml/numerics/cond_threshold`: gate ML off when condition estimate > threshold

## Example (C++)
- `SCIPsetStringParam(scip, "bbml/model_path", "/path/to/score_mlp.onnx");`
- `SCIPsetBoolParam(scip, "bbml/enable", TRUE);`
- `SCIPsetRealParam(scip, "bbml/confidence", 0.7);`
- `SCIPsetRealParam(scip, "bbml/alpha/depth_penalty", 0.03);`
- Hot reload: bump `bbml/reload` (e.g., `SCIPsetIntParam(scip, "bbml/reload", 1);`)

## Notes on ONNX and fallbacks
- If `BBML_WITH_ONNX` is enabled and `ONNXRUNTIME_DIR` points to your ORT install, the branchrule calls ONNXRuntime.
- If ONNX isn’t available or the model path is empty/invalid, the plugin safely falls back to zeros (default heuristic dominates).
 - GNN export supports var-only and graph signatures. The exporter reads the model config from the checkpoint to choose the correct signature; you can override via CLI flags.

## Runner and plugin loading
- The examples build a small runner (`bbml_run`) that links the plugins and solves an instance,
  applying `.set` files for bbml/* parameters (and standard SCIP limits/*).
- This avoids solver‑CLI quirks and ensures parameters are applied deterministically.

Telemetry format
- Candidate NDJSON: one JSON object per line per candidate at each LP branching decision.
  - Optional `sb_score_up` / `sb_score_down` when strong‑branch logging is enabled.
  - Convert NDJSON → Parquet: `python -m bbml.data.json_to_parquet --in <path>.ndjson --out <path>.parquet`
- Graph NDJSON (when `bbml/telemetry/graph = TRUE`): one JSON object per node containing arrays
  - `var_feat` [n_var, d_var], `con_feat` [n_con, d_con]
  - `edge_index` [2, E] (rows = constraints, cols = variables), `edge_val` [E]
  - `chosen_idx` and optional `sb_score_up/down`
  - Load directly in Python with `GraphJsonNodeDataset` (no Parquet step needed)

## Benchmarks and metrics
- The example runner prints a machine-readable line per instance:
  - `BBML_SUMMARY nodes=<n> time=<sec> gap=<rel> primal_integral=<pi> solved=<0|1> status=<code>`
- Makefile targets aggregate: avg_nodes, avg_time, avg_gap, avg_primal_integral, solved count, and PAR‑2.

## Best practices
- Use a diverse set of training instances and keep train/val/test splits disjoint.
- For telemetry collection, consider setting a node cap (e.g., NODES=500) for consistency across instances.
- Disable `bbml/telemetry/strongbranch` on larger instances; SB is expensive and stresses the LP interface.
- For benchmarking, compare with/without BBML using identical limits (see targets above).
