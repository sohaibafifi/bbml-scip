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


def _log_looks_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return False
    return ("SCIP Status" in text) or ("BBML_SUMMARY" in text)


def _record_looks_complete(path: Path, solver: str, seed: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("solver") != solver or int(record.get("seed", -1)) != seed:
        return False
    return isinstance(record.get("status"), str) and record.get("status") != "" and "solve_time" in record and "n_nodes" in record


def _load_complete_record(path: Path, solver: str, seed: int) -> dict[str, object] | None:
    if not _record_looks_complete(path, solver, seed):
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _alpha_log_looks_complete(path: Path, record: dict[str, object] | None) -> bool:
    if record is None:
        return False
    try:
        n_nodes = int(record.get("n_nodes", 0))
    except (TypeError, ValueError):
        n_nodes = 0
    if n_nodes <= 1:
        return True
    return path.is_file() and path.stat().st_size > 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one SCIP benchmark task and persist a JSON result.")
    ap.add_argument("--runner-bin", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--instance-set", default="")
    ap.add_argument("--solver", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--time-limit", required=True, type=int)
    ap.add_argument("--result-out", required=True)
    ap.add_argument("--scip-log", required=True)
    ap.add_argument("--alpha-log-out", default="")
    ap.add_argument("--root-cuts-only", action="store_true")
    ap.add_argument("--disable-restarts", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--temperature-file", default="")
    ap.add_argument("--disable-ml", action="store_true")
    ap.add_argument("--alpha-min", type=float, default=None)
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--depth-penalty", type=float, default=None)
    ap.add_argument("--alpha-theta", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--cond-threshold", type=float, default=None)
    ap.add_argument("--disable-confidence-gate", action="store_true")
    args = ap.parse_args()

    result_out = Path(args.result_out)
    scip_log = Path(args.scip_log)
    alpha_log_out = Path(args.alpha_log_out) if args.alpha_log_out else None
    result_out.parent.mkdir(parents=True, exist_ok=True)
    scip_log.parent.mkdir(parents=True, exist_ok=True)
    if alpha_log_out is not None:
        alpha_log_out.parent.mkdir(parents=True, exist_ok=True)

    existing_record = _load_complete_record(result_out, args.solver, args.seed)
    if existing_record is not None and _log_looks_complete(scip_log):
        if alpha_log_out is None or _alpha_log_looks_complete(alpha_log_out, existing_record):
            return 0

    set_path = ""
    tmp_log_path = None
    tmp_alpha_path: Path | None = None
    alpha_path_for_set = ""
    if alpha_log_out is not None and not args.disable_ml:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, dir=alpha_log_out.parent) as tmp_alpha:
            tmp_alpha_path = Path(tmp_alpha.name)
        alpha_path_for_set = str(tmp_alpha_path)
    with tempfile.NamedTemporaryFile("w", suffix=".set", delete=False) as tmp:
        tmp.write(f"limits/time = {args.time_limit}\n")
        tmp.write(f"randomization/randomseedshift = {args.seed}\n")
        tmp.write("bbml/telemetry = FALSE\n")
        if args.disable_ml:
            tmp.write("bbml/enable = FALSE\n")
        else:
            tmp.write("bbml/enable = TRUE\n")
            if alpha_path_for_set:
                tmp.write("bbml/telemetry/alpha = TRUE\n")
                tmp.write(f'bbml/telemetry/alpha_path = "{alpha_path_for_set}"\n')
            if args.model:
                tmp.write(f'bbml/model_path = "{args.model}"\n')
            if args.temperature is not None:
                tmp.write(f"bbml/temperature = {args.temperature}\n")
            elif args.temperature_file and Path(args.temperature_file).is_file():
                temperature = Path(args.temperature_file).read_text().strip()
                if temperature:
                    tmp.write(f"bbml/temperature = {temperature}\n")
            if args.confidence is not None:
                tmp.write(f"bbml/confidence = {args.confidence}\n")
        if args.root_cuts_only:
            tmp.write("separating/maxrounds = 0\n")
        if args.disable_restarts:
            tmp.write("presolving/maxrestarts = 0\n")
        if args.alpha_min is not None:
            tmp.write(f"bbml/alpha/min = {args.alpha_min}\n")
        if args.alpha_max is not None:
            tmp.write(f"bbml/alpha/max = {args.alpha_max}\n")
        if args.depth_penalty is not None:
            tmp.write(f"bbml/alpha/depth_penalty = {args.depth_penalty}\n")
        if args.alpha_theta is not None:
            tmp.write(f"bbml/alpha/theta = {args.alpha_theta}\n")
        if args.disable_confidence_gate:
            tmp.write("bbml/alpha/use_confidence_gate = FALSE\n")
        if args.cond_threshold is not None:
            tmp.write(f"bbml/numerics/cond_threshold = {args.cond_threshold}\n")
        if args.solver == "strong-branch":
            tmp.write("branching/fullstrong/priority = 1000000\n")
            tmp.write("bbml/enable = FALSE\n")
        set_path = tmp.name

    cmd = [args.runner_bin, "--problem", args.instance, "--set", set_path]
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, dir=scip_log.parent) as tmp_log:
            tmp_log_path = Path(tmp_log.name)
        with tmp_log_path.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)
    finally:
        if set_path:
            try:
                os.unlink(set_path)
            except OSError:
                pass
    if tmp_log_path is None:
        return 1
    tmp_log_path.replace(scip_log)
    record = parse_log(scip_log)
    record.update(
        {
            "instance_id": _instance_id(args.instance),
            "instance_path": str(Path(args.instance).resolve()),
            "instance_set": args.instance_set or "unknown",
            "solver": args.solver,
            "seed": args.seed,
        }
    )
    if proc.returncode == 0 and _log_looks_complete(scip_log):
        if alpha_log_out is not None:
            try:
                if tmp_alpha_path is not None and tmp_alpha_path.exists() and tmp_alpha_path.stat().st_size > 0:
                    tmp_alpha_path.replace(alpha_log_out)
                elif alpha_log_out.exists():
                    alpha_log_out.unlink()
            except OSError:
                pass
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=result_out.parent) as tmp_result:
            tmp_result.write(json.dumps(record) + "\n")
            tmp_result_path = Path(tmp_result.name)
        tmp_result_path.replace(result_out)
        return 0
    try:
        if result_out.exists():
            result_out.unlink()
    except OSError:
        pass
    try:
        if tmp_alpha_path is not None and tmp_alpha_path.exists():
            tmp_alpha_path.unlink()
    except OSError:
        pass
    try:
        if alpha_log_out is not None and alpha_log_out.exists():
            alpha_log_out.unlink()
    except OSError:
        pass
    return int(proc.returncode) or 1


if __name__ == "__main__":
    raise SystemExit(main())
