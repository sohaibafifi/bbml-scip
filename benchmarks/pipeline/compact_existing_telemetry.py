#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path

from bbml.data.telemetry_compact import compact_collection_outputs


def _stem_without_ext(path: Path) -> str:
    name = path.name
    if name.endswith(".ndjson.gz"):
        return name[:-10]
    if name.endswith(".ndjson"):
        return name[:-7]
    if name.endswith(".pt"):
        return name[:-3]
    return path.stem


def _compact_one(candidate_src: Path, graph_src: Path, max_graph_nodes: int, delete_raw: bool) -> str:
    stem = _stem_without_ext(candidate_src)
    candidate_out = candidate_src.with_suffix(candidate_src.suffix + ".gz")
    graph_out = graph_src.with_suffix(".pt")
    if candidate_out.exists() and candidate_out.stat().st_size > 0 and graph_out.exists() and graph_out.stat().st_size > 0:
        return f"[skip] {stem}"
    stats = compact_collection_outputs(
        candidate_src=candidate_src,
        graph_src=graph_src,
        candidate_out=candidate_out,
        graph_out=graph_out,
        max_graph_nodes=max_graph_nodes,
        seed=0,
    )
    if delete_raw:
        try:
            candidate_src.unlink()
        except OSError:
            pass
        try:
            graph_src.unlink()
        except OSError:
            pass
    return f"[ok] {stem} " f"graph={stats.graph_nodes_kept}/{stats.graph_nodes_total} " f"candidate_rows={stats.candidate_rows_kept}/{stats.candidate_rows_total}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact existing raw telemetry logs in place.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--max-graph-nodes", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--delete-raw", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    family = args.family
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    jobs = max(1, int(args.jobs))

    tasks: list[tuple[Path, Path]] = []
    for split in splits:
        candidate_dir = data_dir / "logs" / family / split / "candidates"
        graph_dir = data_dir / "logs" / family / split / "graph"
        if not candidate_dir.exists() or not graph_dir.exists():
            continue
        for candidate_src in sorted(candidate_dir.glob("*.ndjson")):
            stem = _stem_without_ext(candidate_src)
            graph_src = graph_dir / f"{stem}.ndjson"
            if not graph_src.exists():
                continue
            tasks.append((candidate_src, graph_src))

    if not tasks:
        print("no raw telemetry pairs found")
        return 0

    print(f"compacting {len(tasks)} telemetry file pairs with max_graph_nodes={args.max_graph_nodes}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [ex.submit(_compact_one, candidate_src, graph_src, args.max_graph_nodes, args.delete_raw) for candidate_src, graph_src in tasks]
        for fut in concurrent.futures.as_completed(futures):
            print(fut.result(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
