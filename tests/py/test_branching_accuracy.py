import csv
import subprocess
import sys

import pandas as pd
import torch

from bbml.train.train_rank import DEFAULT_FEATS, ScoreMLP


def test_branching_accuracy_script_runs(tmp_path):
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
                    "chosen_idx": 2,
                }
            )
    pd.DataFrame(rows).to_parquet(parquet, index=False)

    ckpt = tmp_path / "mlp.pt"
    model = ScoreMLP(d_in=len(DEFAULT_FEATS), hidden=8, dropout=0.0)
    torch.save(
        {
            "model": "mlp",
            "cfg": {"model": "mlp", "d_in": len(DEFAULT_FEATS), "hidden": 8, "dropout": 0.0},
            "state_dict": model.state_dict(),
        },
        ckpt,
    )

    out_csv = tmp_path / "accuracy.csv"
    proc = subprocess.run(
        [
            sys.executable,
            "benchmarks/eval/branching_accuracy.py",
            "--ckpt",
            str(ckpt),
            "--parquet",
            str(parquet),
            "--name",
            "bbml-mlp",
            "--out",
            str(out_csv),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    with out_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["model"] == "bbml-mlp"
    assert int(rows[0]["n_groups"]) == 2
    acc = float(rows[0]["top1_acc"])
    assert 0.0 <= acc <= 1.0
