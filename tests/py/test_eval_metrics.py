import importlib.util

import pandas as pd


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_kpis_includes_wtl_columns():
    mod = _load_module("benchmarks/eval/kpis.py", "bbml_kpis")
    df = pd.DataFrame(
        [
            {"instance_id": "a", "seed": 0, "solver": "scip-default", "status": "optimal", "solve_time": 10.0, "n_nodes": 100, "instance_set": "toy"},
            {"instance_id": "b", "seed": 0, "solver": "scip-default", "status": "optimal", "solve_time": 10.0, "n_nodes": 100, "instance_set": "toy"},
            {"instance_id": "a", "seed": 0, "solver": "bbml-gnn-graph", "status": "optimal", "solve_time": 8.0, "n_nodes": 90, "instance_set": "toy"},
            {"instance_id": "b", "seed": 0, "solver": "bbml-gnn-graph", "status": "optimal", "solve_time": 10.2, "n_nodes": 110, "instance_set": "toy"},
        ]
    )
    out = mod.compute_kpis(df, baseline="scip-default", wtl_threshold=0.05)
    row = out[(out["solver"] == "bbml-gnn-graph") & (out["instance_set"] == "toy")].iloc[0]
    assert row["wtl_wins"] == 1
    assert row["wtl_ties"] == 1
    assert row["wtl_losses"] == 0
    assert row["wtl"] == "1/1/0"


def test_summary_table_builds_wtl_table():
    mod = _load_module("benchmarks/eval/summary_table.py", "bbml_summary")
    df = pd.DataFrame(
        [
            {
                "instance_set": "all",
                "solver": "scip-default",
                "sgm_time": 1.0,
                "sgm_nodes": 1.0,
                "wtl": "0/2/0",
                "wtl_win_pct": 0.0,
            },
            {
                "instance_set": "all",
                "solver": "bbml-gnn-graph",
                "sgm_time": 0.9,
                "sgm_nodes": 0.8,
                "wtl": "1/1/0",
                "wtl_win_pct": 50.0,
            },
        ]
    )
    tex = mod.build_wtl_table(df, iset="all")
    assert "W/T/L" in tex
    assert "1/1/0" in tex
