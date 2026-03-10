#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_instances import write_ca, write_cfl, write_mis, write_sc

STAGE_CONFIGS = {
    "pilot": {
        "train": 200,
        "val": 50,
        "easy_test": 5,
        "medium_test": 5,
        "hard_test": 5,
        "bench_seeds": "0",
        "collect_tl": 300,
        "collect_max_nodes": 500,
    },
    "dev": {
        "train": 1000,
        "val": 200,
        "easy_test": 10,
        "medium_test": 10,
        "hard_test": 10,
        "bench_seeds": "0 1",
        "collect_tl": 600,
        "collect_max_nodes": 1000,
    },
    "final": {
        "train": 10000,
        "val": 2000,
        "easy_test": 20,
        "medium_test": 20,
        "hard_test": 20,
        "bench_seeds": "0 1 2 3 4",
        "collect_tl": 1200,
        "collect_max_nodes": 5000,
    },
}

FAMILY_LABELS = {
    "sc": "setcover",
    "ca": "cauctions",
    "cfl": "facilities",
    "mis": "indset",
}

FAMILY_CONFIGS = {
    "sc": {
        "writer": write_sc,
        "easy": {"n_rows": 500, "n_cols": 1000, "density": 0.05},
        "medium": {"n_rows": 1000, "n_cols": 1000, "density": 0.05},
        "hard": {"n_rows": 2000, "n_cols": 1000, "density": 0.05},
    },
    "ca": {
        "writer": write_ca,
        "easy": {"n_items": 100, "n_bids": 500, "max_bundle": 5},
        "medium": {"n_items": 200, "n_bids": 1000, "max_bundle": 5},
        "hard": {"n_items": 300, "n_bids": 1500, "max_bundle": 5},
    },
    "cfl": {
        "writer": write_cfl,
        "easy": {"n_cust": 100, "n_fac": 100},
        "medium": {"n_cust": 200, "n_fac": 100},
        "hard": {"n_cust": 400, "n_fac": 100},
    },
    "mis": {
        "writer": write_mis,
        "easy": {"n_nodes": 500, "m": 4},
        "medium": {"n_nodes": 1000, "m": 4},
        "hard": {"n_nodes": 1500, "m": 4},
    },
}

SEED_OFFSETS = {
    "train": 0,
    "val": 100000,
    "easy_test": 200000,
    "medium_test": 300000,
    "hard_test": 400000,
}


def _generate_block(
    writer,
    writer_kwargs: dict[str, int | float],
    out_dir: Path,
    family: str,
    count: int,
    seed_offset: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(count):
        path = out_dir / f"{family}_{idx:05d}.lp"
        if not path.exists():
            writer(path, seed=seed_offset + idx, **writer_kwargs)
        paths.append(path.resolve())
    return paths


def _write_list(path: Path, entries: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{entry}\n" for entry in entries))


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare staged learn2branch-style instances and list files.")
    ap.add_argument("--stage", choices=sorted(STAGE_CONFIGS), required=True)
    ap.add_argument("--family", choices=sorted(FAMILY_CONFIGS), required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="Stage/family-specific DATA_DIR root")
    ap.add_argument("--instances-dir", type=Path, required=True, help="Directory to write stage list files")
    args = ap.parse_args()

    stage_cfg = STAGE_CONFIGS[args.stage]
    family_cfg = FAMILY_CONFIGS[args.family]
    writer = family_cfg["writer"]

    generated: dict[str, list[Path]] = {}
    generated["train"] = _generate_block(
        writer,
        dict(family_cfg["easy"]),
        args.data_dir / "instances" / args.family / "train",
        args.family,
        int(stage_cfg["train"]),
        SEED_OFFSETS["train"],
    )
    generated["val"] = _generate_block(
        writer,
        dict(family_cfg["easy"]),
        args.data_dir / "instances" / args.family / "val",
        args.family,
        int(stage_cfg["val"]),
        SEED_OFFSETS["val"],
    )
    for difficulty in ("easy", "medium", "hard"):
        split = f"{difficulty}_test"
        generated[split] = _generate_block(
            writer,
            dict(family_cfg[difficulty]),
            args.data_dir / "instances" / args.family / split,
            args.family,
            int(stage_cfg[split]),
            SEED_OFFSETS[split],
        )

    _write_list(args.instances_dir / f"{args.family}_train.txt", generated["train"])
    _write_list(args.instances_dir / f"{args.family}_val.txt", generated["val"])
    for difficulty in ("easy", "medium", "hard"):
        split = f"{difficulty}_test"
        _write_list(args.instances_dir / f"{args.family}_{difficulty}_test.txt", generated[split])

    summary = {
        "stage": args.stage,
        "family": args.family,
        "problem": FAMILY_LABELS[args.family],
        "counts": {key: int(value) if isinstance(value, int) else value for key, value in stage_cfg.items()},
        "instance_lists": {
            "train": str((args.instances_dir / f"{args.family}_train.txt").resolve()),
            "val": str((args.instances_dir / f"{args.family}_val.txt").resolve()),
            "easy_test": str((args.instances_dir / f"{args.family}_easy_test.txt").resolve()),
            "medium_test": str((args.instances_dir / f"{args.family}_medium_test.txt").resolve()),
            "hard_test": str((args.instances_dir / f"{args.family}_hard_test.txt").resolve()),
        },
    }
    (args.instances_dir / "stage_config.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
