import pandas as pd
from bbml.train.train_rank import NodeDataset, DEFAULT_FEATS


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
