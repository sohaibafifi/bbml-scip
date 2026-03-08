import subprocess
import sys

import onnx
import pandas as pd
import torch

from bbml.models.graph_ranker import GraphRanker
from bbml.train.train_rank import DEFAULT_FEATS, GRAPH_VAR_FEATS, ScoreMLP


def _run_module(*args):
    proc = subprocess.run([sys.executable, "-m", *args], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_calibrate_uses_checkpoint_metadata(tmp_path):
    parquet = tmp_path / "val.parquet"
    rows = []
    for node_id in [1, 2]:
        for vid in range(3):
            rows.append(
                {
                    "node_id": node_id,
                    "var_id": vid,
                    "obj": 0.1 * vid,
                    "reduced_cost": float(vid),
                    "fracval": 0.5,
                    "domain_width": 1.0,
                    "is_binary": 1,
                    "is_integer": 1,
                    "pseudocost_up": float(vid + 1),
                    "pseudocost_down": float(vid + 2),
                    "pc_obs_up": float(vid + 3),
                    "pc_obs_down": float(vid + 4),
                    "sb_score": float(vid + 1),
                }
            )
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    model = ScoreMLP(d_in=len(DEFAULT_FEATS), hidden=8, dropout=0.1)
    ckpt = tmp_path / "mlp.pt"
    torch.save(
        {
            "model": "mlp",
            "cfg": {"model": "mlp", "d_in": len(DEFAULT_FEATS), "hidden": 8, "dropout": 0.1},
            "state_dict": model.state_dict(),
        },
        ckpt,
    )

    out_path = tmp_path / "temperature.txt"
    _run_module("bbml.train.calibrate", "--ckpt", str(ckpt), "--parquet", str(parquet), "--out", str(out_path))
    temperature = float(out_path.read_text().strip())
    assert temperature > 0.0


def test_calibrate_supports_checkpoint_ensemble(tmp_path):
    parquet = tmp_path / "val.parquet"
    rows = []
    for node_id in [1, 2]:
        for vid in range(3):
            rows.append(
                {
                    "node_id": node_id,
                    "var_id": vid,
                    "obj": 0.05 * vid,
                    "reduced_cost": float(vid),
                    "fracval": 0.5,
                    "domain_width": 1.0,
                    "is_binary": 1,
                    "is_integer": 1,
                    "pseudocost_up": float(vid + 1),
                    "pseudocost_down": float(vid + 2),
                    "pc_obs_up": float(vid + 3),
                    "pc_obs_down": float(vid + 4),
                    "sb_score": float(vid + 1),
                }
            )
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    ckpts = []
    for idx in range(2):
        model = ScoreMLP(d_in=len(DEFAULT_FEATS), hidden=8, dropout=0.1)
        ckpt = tmp_path / f"mlp_{idx}.pt"
        torch.save(
            {
                "model": "mlp",
                "cfg": {"model": "mlp", "d_in": len(DEFAULT_FEATS), "hidden": 8, "dropout": 0.1},
                "state_dict": model.state_dict(),
            },
            ckpt,
        )
        ckpts.append(str(ckpt))

    out_path = tmp_path / "temperature_ensemble.txt"
    _run_module(
        "bbml.train.calibrate",
        "--ckpt",
        ",".join(ckpts),
        "--parquet",
        str(parquet),
        "--out",
        str(out_path),
    )
    temperature = float(out_path.read_text().strip())
    assert temperature > 0.0


def test_export_onnx_uses_checkpoint_graph_signature(tmp_path):
    ckpt = tmp_path / "graph.pt"
    model = GraphRanker(d_var=len(GRAPH_VAR_FEATS), d_con=4, hidden=8, layers=1, dropout=0.0)
    torch.save(
        {
            "model": "gnn",
            "cfg": {
                "model": "gnn",
                "d_var": len(GRAPH_VAR_FEATS),
                "d_con": 4,
                "hidden": 8,
                "layers": 1,
                "dropout": 0.0,
                "graph_inputs": True,
            },
            "state_dict": model.state_dict(),
        },
        ckpt,
    )

    out_path = tmp_path / "graph.onnx"
    _run_module("bbml.export.export_onnx", "--ckpt", str(ckpt), "--out", str(out_path))
    graph = onnx.load(str(out_path))
    assert len(graph.graph.input) == 3
