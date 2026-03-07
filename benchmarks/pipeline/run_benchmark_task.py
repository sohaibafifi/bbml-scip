#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from parse_scip_log import parse_log


def _instance_id(path: str) -> str:
    name = Path(path).name
    if name.endswith(".mps.gz"):
        return name[:-7]
    for suffix in (".lp", ".mps", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one SCIP benchmark task and persist a JSON result.")
    ap.add_argument("--runner-bin", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--time-limit", required=True, type=int)
    ap.add_argument("--result-out", required=True)
    ap.add_argument("--scip-log", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--temperature-file", default="")
    ap.add_argument("--disable-ml", action="store_true")
    ap.add_argument("--alpha-min", type=float, default=None)
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--depth-penalty", type=float, default=None)
    args = ap.parse_args()

    result_out = Path(args.result_out)
    scip_log = Path(args.scip_log)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    scip_log.parent.mkdir(parents=True, exist_ok=True)

    set_path = ""
    with tempfile.NamedTemporaryFile("w", suffix=".set", delete=False) as tmp:
        tmp.write(f"limits/time = {args.time_limit}\n")
        tmp.write(f"randomization/randomseedshift = {args.seed}\n")
        tmp.write("bbml/telemetry = FALSE\n")
        if args.disable_ml:
            tmp.write("bbml/enable = FALSE\n")
        else:
            tmp.write("bbml/enable = TRUE\n")
            if args.model:
                tmp.write(f'bbml/model_path = "{args.model}"\n')
            if args.temperature_file and Path(args.temperature_file).is_file():
                temperature = Path(args.temperature_file).read_text().strip()
                if temperature:
                    tmp.write(f"bbml/temperature = {temperature}\n")
        if args.alpha_min is not None:
            tmp.write(f"bbml/alpha/min = {args.alpha_min}\n")
        if args.alpha_max is not None:
            tmp.write(f"bbml/alpha/max = {args.alpha_max}\n")
        if args.depth_penalty is not None:
            tmp.write(f"bbml/alpha/depth_penalty = {args.depth_penalty}\n")
        if args.solver == "strong-branch":
            tmp.write("branching/fullstrong/priority = 1000000\n")
            tmp.write("bbml/enable = FALSE\n")
        set_path = tmp.name

    cmd = [args.runner_bin, "--problem", args.instance, "--set", set_path]
    try:
        with scip_log.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)
    finally:
        if set_path:
            try:
                os.unlink(set_path)
            except OSError:
                pass
    record = parse_log(scip_log)
    record.update(
        {
            "instance_id": _instance_id(args.instance),
            "solver": args.solver,
            "seed": args.seed,
        }
    )
    result_out.write_text(json.dumps(record) + "\n")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
