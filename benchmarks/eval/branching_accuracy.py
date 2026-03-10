#!/usr/bin/env python3
"""Compute held-out top-1 branching accuracy from checkpoints and telemetry datasets."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from bbml.train.calibrate import _build_loader, _load_models, _score_group


def _target_index(group: Any) -> int:
    if getattr(group, "y_true", None) is not None:
        return int(torch.argmax(group.y_true).item())
    chosen = getattr(group, "chosen", None)
    if chosen is None:
        raise ValueError("group has neither y_true nor chosen target")
    return int(chosen)


def evaluate_top1(ckpt: str, args: argparse.Namespace) -> dict[str, object]:
    models, cfg = _load_models(ckpt, args.device)
    loader = _build_loader(cfg, args)

    total = 0
    correct = 0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            for group in batch:
                scores = _score_group(models, group, cfg, args.device)
                pred = int(torch.argmax(scores).item())
                target = _target_index(group)
                correct += int(pred == target)
                total += 1
                if hasattr(group, "X"):
                    total_items += int(group.X.size(0))
                else:
                    total_items += int(group.var_feat.size(0))

    return {
        "model": args.name or Path(ckpt.split(",")[0]).stem,
        "ckpt": ckpt,
        "n_groups": total,
        "n_items": total_items,
        "top1_acc": (correct / total) if total else 0.0,
    }


def write_csv(path: Path, record: dict[str, object], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "ckpt", "n_groups", "n_items", "top1_acc"]
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerow(record)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Checkpoint path or comma-separated checkpoint ensemble")
    ap.add_argument("--parquet", required=True, help="Candidate parquet for MLP and var-only GNN models")
    ap.add_argument("--graph-ndjson", default=None, help="Single graph NDJSON file for graph-input checkpoints")
    ap.add_argument("--graph-manifest", default=None, help="Manifest of graph NDJSON files for graph-input checkpoints")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default=None, help="Label written into the output record")
    ap.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    ap.add_argument("--append", action="store_true", help="Append to --out instead of overwriting it")
    args = ap.parse_args()

    if not Path(args.parquet).exists():
        raise FileNotFoundError(args.parquet)

    record = evaluate_top1(args.ckpt, args)
    print(f"Top-1 accuracy: {record['model']} " f"acc={float(record['top1_acc']):.4f} " f"(groups={record['n_groups']}, items={record['n_items']})")
    if args.out is not None:
        write_csv(args.out, record, append=args.append)
        print(f"Written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
