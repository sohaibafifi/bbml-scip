import gzip
import json

import pandas as pd

from bbml.data.json_to_parquet import convert_sources_to_parquet


def test_convert_sources_to_parquet_manifest_sources(tmp_path):
    src_a = tmp_path / "a.ndjson"
    src_b = tmp_path / "b.ndjson"
    src_a.write_text("\n".join(json.dumps(row) for row in [{"node_id": 1, "obj": 1.0}, {"node_id": 2, "obj": 2.0}]) + "\n")
    src_b.write_text("\n".join(json.dumps(row) for row in [{"node_id": 3, "obj": 3.0, "extra": "x"}]) + "\n")

    out_path = tmp_path / "rows.parquet"
    rows = convert_sources_to_parquet(
        paths=[src_a, src_b],
        out_path=out_path,
        chunksize=10,
        row_group_size=10,
        compression="none",
        one_pass=False,
    )

    assert rows == 3
    df = pd.read_parquet(out_path)
    assert set(df["node_id"].tolist()) == {1, 2, 3}
    assert "extra" in df.columns


def test_convert_sources_to_parquet_reads_gz_sources(tmp_path):
    src = tmp_path / "rows.ndjson.gz"
    with gzip.open(src, "wt") as fh:
        fh.write(json.dumps({"node_id": 1, "obj": 1.0}) + "\n")
        fh.write(json.dumps({"node_id": 2, "obj": 2.0}) + "\n")

    out_path = tmp_path / "rows.parquet"
    rows = convert_sources_to_parquet(
        paths=[src],
        out_path=out_path,
        chunksize=10,
        row_group_size=10,
        compression="none",
        one_pass=False,
    )

    assert rows == 2
    df = pd.read_parquet(out_path)
    assert set(df["node_id"].tolist()) == {1, 2}
