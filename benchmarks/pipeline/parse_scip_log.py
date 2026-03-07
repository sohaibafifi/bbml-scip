#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOLVE_TIME_RE = re.compile(r"Solving Time.*?:\s*([0-9.]+)")
NODES_RE = re.compile(r"Solving Nodes.*?:\s*([0-9]+)")
FIRST_INC_RE = re.compile(r"First Solution.*?:\s*([0-9.]+)")
ROOT_TIME_RE = re.compile(r"Root Node.*?LP Time.*?:\s*([0-9.]+)")


def _normalize_status(line: str) -> str:
    low = line.lower()
    if "optimal" in low:
        return "optimal"
    if "infeasible" in low:
        return "infeasible"
    if "time limit" in low:
        return "timelimit"
    if "memory limit" in low:
        return "memlimit"
    if "node limit" in low:
        return "nodelimit"
    if "gap limit" in low:
        return "gaplimit"
    return "unknown"


def parse_log(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "status": "unknown",
        "solve_time": -1.0,
        "n_nodes": -1,
        "time_to_first_inc": -1.0,
        "root_time": -1.0,
    }
    with path.open() as fh:
        for line in fh:
            if "SCIP Status" in line:
                record["status"] = _normalize_status(line)
            m = SOLVE_TIME_RE.search(line)
            if m:
                record["solve_time"] = float(m.group(1))
            m = NODES_RE.search(line)
            if m:
                record["n_nodes"] = int(m.group(1))
            m = FIRST_INC_RE.search(line)
            if m:
                record["time_to_first_inc"] = float(m.group(1))
            m = ROOT_TIME_RE.search(line)
            if m:
                record["root_time"] = float(m.group(1))
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse a SCIP log into a JSON result record.")
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    record = parse_log(args.log)
    record.update({"instance_id": args.instance_id, "solver": args.solver, "seed": args.seed})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
