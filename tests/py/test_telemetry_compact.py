import gzip
import json

import torch

from bbml.data.telemetry_compact import compact_collection_outputs


def test_compact_collection_outputs_samples_and_filters(tmp_path):
    candidate_src = tmp_path / "candidates.ndjson"
    graph_src = tmp_path / "graph.ndjson"
    candidate_src.write_text(
        "\n".join(
            json.dumps(
                {
                    "node_id": node_id,
                    "var_id": var_id,
                    "obj": 0.0,
                }
            )
            for node_id in range(3)
            for var_id in range(2)
        )
        + "\n"
    )
    graph_src.write_text(
        "\n".join(
            json.dumps(
                {
                    "node_id": node_id,
                    "var_feat": [[0.1] * 9, [0.2] * 9],
                    "con_feat": [[0.0] * 4],
                    "edge_index": [[0, 0], [0, 1]],
                    "sb_score_up": [1.0, 2.0],
                    "chosen_idx": 1,
                }
            )
            for node_id in range(3)
        )
        + "\n"
    )

    candidate_out = tmp_path / "candidates.ndjson.gz"
    graph_out = tmp_path / "graph.pt"
    stats = compact_collection_outputs(
        candidate_src=candidate_src,
        graph_src=graph_src,
        candidate_out=candidate_out,
        graph_out=graph_out,
        max_graph_nodes=2,
        seed=0,
    )

    assert stats.graph_nodes_total == 3
    assert stats.graph_nodes_kept == 2
    assert candidate_out.exists()
    assert graph_out.exists()

    with gzip.open(candidate_out, "rt") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 4
    assert len({row["node_id"] for row in rows}) == 2

    payload = torch.load(graph_out, map_location="cpu")
    assert len(payload["items"]) == 2
