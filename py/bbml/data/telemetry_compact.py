from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


def _compress_target_scores(values: np.ndarray) -> Optional[torch.Tensor]:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    if not finite.all():
        floor = float(np.min(arr[finite]) - 1.0)
        arr = np.where(finite, arr, floor)
    arr = arr - float(np.min(arr))
    arr = np.log1p(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=20.0, neginf=0.0)
    return torch.tensor(arr, dtype=torch.float32)


def _compact_graph_record(obj: dict) -> dict:
    var_feat = torch.tensor(np.asarray(obj["var_feat"], dtype=np.float32), dtype=torch.float16)
    con_feat = torch.tensor(np.asarray(obj.get("con_feat", []), dtype=np.float32), dtype=torch.float16)
    edge_index = torch.tensor(np.asarray(obj.get("edge_index", [[], []]), dtype=np.int32), dtype=torch.int32)
    y_true = None
    if "sb_score_up" in obj or "sb_score_down" in obj:
        up = np.asarray(obj.get("sb_score_up", [0.0] * int(var_feat.size(0))), dtype=np.float32)
        down = np.asarray(obj.get("sb_score_down", []), dtype=np.float32) if "sb_score_down" in obj else None
        tgt = up if down is None or down.size == 0 else np.maximum(up, down)
        y_true = _compress_target_scores(tgt)
    return {
        "node_id": int(obj.get("node_id", 0)),
        "var_feat": var_feat,
        "con_feat": con_feat,
        "edge_index": edge_index,
        "y_true": y_true,
        "chosen": int(obj.get("chosen_idx", 0)),
    }


def _iter_ndjson(path: Path) -> Iterator[dict]:
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as err:
                raise ValueError(f"invalid NDJSON in {path} at line {lineno}: {err.msg}") from err
            if not isinstance(obj, dict):
                raise ValueError(f"invalid NDJSON in {path} at line {lineno}: expected object")
            yield obj


@dataclass
class CompactTelemetryStats:
    graph_nodes_total: int
    graph_nodes_kept: int
    candidate_rows_total: int
    candidate_rows_kept: int


def compact_collection_outputs(
    candidate_src: Path,
    graph_src: Path,
    candidate_out: Path,
    graph_out: Path,
    max_graph_nodes: int,
    seed: int,
) -> CompactTelemetryStats:
    rng = random.Random(seed)
    selected: list[tuple[int, dict]] = []
    graph_nodes_total = 0
    for stream_idx, obj in enumerate(_iter_ndjson(graph_src)):
        graph_nodes_total += 1
        compact = _compact_graph_record(obj)
        if max_graph_nodes <= 0:
            selected.append((stream_idx, compact))
            continue
        if len(selected) < max_graph_nodes:
            selected.append((stream_idx, compact))
            continue
        pick = rng.randint(0, stream_idx)
        if pick < max_graph_nodes:
            selected[pick] = (stream_idx, compact)

    selected.sort(key=lambda item: item[0])
    kept_items = [item for _, item in selected]
    keep_node_ids = {int(item["node_id"]) for item in kept_items}

    candidate_out.parent.mkdir(parents=True, exist_ok=True)
    graph_out.parent.mkdir(parents=True, exist_ok=True)

    candidate_rows_total = 0
    candidate_rows_kept = 0
    with candidate_src.open() as src, gzip.open(candidate_out, "wt") as dst:
        for lineno, line in enumerate(src, start=1):
            raw = line.strip()
            if not raw:
                continue
            candidate_rows_total += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as err:
                raise ValueError(f"invalid NDJSON in {candidate_src} at line {lineno}: {err.msg}") from err
            if int(obj.get("node_id", -1)) not in keep_node_ids:
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            candidate_rows_kept += 1

    payload = {
        "version": 1,
        "graph_nodes_total": graph_nodes_total,
        "graph_nodes_kept": len(kept_items),
        "items": [
            {
                "var_feat": item["var_feat"],
                "con_feat": item["con_feat"],
                "edge_index": item["edge_index"],
                "y_true": item["y_true"],
                "chosen": item["chosen"],
            }
            for item in kept_items
        ],
    }
    torch.save(payload, graph_out)
    return CompactTelemetryStats(
        graph_nodes_total=graph_nodes_total,
        graph_nodes_kept=len(kept_items),
        candidate_rows_total=candidate_rows_total,
        candidate_rows_kept=candidate_rows_kept,
    )
