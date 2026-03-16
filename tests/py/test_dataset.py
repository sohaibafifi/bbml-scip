import json

import pandas as pd
import torch
from bbml.train.train_rank import NodeDataset, GraphJsonNodeDataset, DEFAULT_FEATS


def test_node_dataset_grouping(tmp_path):
    # Build a tiny dataframe with two nodes and simple features
    rows = []
    for nid in [1, 2]:
        for vid in range(3):
            rows.append(
                {
                    "node_id": nid,
                    "var_id": vid,
                    "obj": 0.0,
                    "reduced_cost": float(vid),
                    "fracval": 0.5,
                    "domain_width": 1.0,
                    "is_binary": 1,
                    "is_integer": 1,
                    "pseudocost_up": float(vid + 1),
                    "pseudocost_down": float(vid + 2),
                    "pc_obs_up": float(vid + 3),
                    "pc_obs_down": float(vid + 4),
                    "chosen_idx": 1,
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "test_df.parquet"
    df.to_parquet(path, index=False)
    ds = NodeDataset(str(path), feature_cols=DEFAULT_FEATS)
    assert len(ds) == 2
    # Should infer chosen index from column
    for grp in ds:
        assert grp.chosen == 1


def test_node_dataset_falls_back_when_sb_targets_are_nan(tmp_path):
    rows = []
    for vid in range(3):
        rows.append(
            {
                "node_id": 1,
                "var_id": vid,
                "obj": 0.0,
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
                "sb_score_up": float("nan"),
                "sb_score_down": float("nan"),
            }
        )
    path = tmp_path / "bad_targets.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    ds = NodeDataset(str(path), feature_cols=DEFAULT_FEATS)
    grp = ds[0]
    assert grp.y_true is None
    assert grp.chosen == 2


def test_graph_json_dataset_reuses_offset_cache(tmp_path):
    graph_path = tmp_path / "graph.ndjson"
    line = {
        "var_feat": [[0.1] * 9, [0.2] * 9],
        "con_feat": [[0.0] * 4],
        "edge_index": [[0, 0], [0, 1]],
        "sb_score_up": [1.0, 2.0],
        "sb_score_down": [1.5, 1.0],
        "chosen_idx": 1,
    }
    graph_path.write_text(json.dumps(line) + "\n")
    ds = GraphJsonNodeDataset(ndjson_path=str(graph_path))
    assert len(ds) == 1
    cache_path = graph_path.with_suffix(".ndjson.offsets.pt")
    assert cache_path.exists()
    ds_cached = GraphJsonNodeDataset(ndjson_path=str(graph_path))
    assert len(ds_cached) == 1
    assert ds_cached.d_var == 9
    assert ds_cached.d_con == 4


def test_graph_json_dataset_reads_compact_pt_shards(tmp_path):
    shard_path = tmp_path / "graph.pt"
    torch.save(
        {
            "version": 1,
            "items": [
                {
                    "var_feat": torch.tensor([[0.1] * 9, [0.2] * 9], dtype=torch.float16),
                    "con_feat": torch.tensor([[0.0] * 4], dtype=torch.float16),
                    "edge_index": torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
                    "y_true": torch.tensor([0.0, 1.0], dtype=torch.float32),
                    "chosen": 1,
                }
            ],
        },
        shard_path,
    )
    ds = GraphJsonNodeDataset(ndjson_path=str(shard_path))
    assert len(ds) == 1
    grp = ds[0]
    assert grp.var_feat.dtype == torch.float32
    assert grp.edge_index.dtype == torch.long
    assert grp.chosen == 1
