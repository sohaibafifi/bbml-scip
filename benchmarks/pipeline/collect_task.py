#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def _quote(path: str) -> str:
    return '"' + path.replace('"', '\\"') + '"'


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one SCIP telemetry collection task.")
    ap.add_argument("--runner-bin", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--time-limit", type=int, required=True)
    ap.add_argument("--max-nodes", type=int, required=True)
    ap.add_argument("--candidate-out", required=True)
    ap.add_argument("--graph-out", required=True)
    ap.add_argument("--scip-log", required=True)
    args = ap.parse_args()

    candidate_out = Path(args.candidate_out)
    graph_out = Path(args.graph_out)
    scip_log = Path(args.scip_log)
    for path in (candidate_out, graph_out, scip_log):
        path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".ndjson.tmp", dir=str(candidate_out.parent), delete=False) as tmp_candidate:
        candidate_tmp = Path(tmp_candidate.name)
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson.tmp", dir=str(graph_out.parent), delete=False) as tmp_graph:
        graph_tmp = Path(tmp_graph.name)

    set_path = ""
    with tempfile.NamedTemporaryFile("w", suffix=".set", delete=False) as tmp:
        tmp.write(f"limits/time = {args.time_limit}\n")
        tmp.write(f"limits/nodes = {args.max_nodes}\n")
        tmp.write(f"randomization/randomseedshift = {args.seed}\n")
        tmp.write("bbml/enable = TRUE\n")
        tmp.write('bbml/model_path = ""\n')
        tmp.write("bbml/telemetry = TRUE\n")
        tmp.write("bbml/telemetry/append = FALSE\n")
        tmp.write(f"bbml/telemetry/path = {_quote(str(candidate_tmp))}\n")
        tmp.write("bbml/telemetry/strongbranch = TRUE\n")
        tmp.write("bbml/telemetry/graph = TRUE\n")
        tmp.write(f"bbml/telemetry/graph_path = {_quote(str(graph_tmp))}\n")
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
    if proc.returncode != 0:
        _safe_unlink(candidate_tmp)
        _safe_unlink(graph_tmp)
    else:
        for path in (candidate_tmp, graph_tmp):
            if not path.is_file() or path.stat().st_size == 0:
                _safe_unlink(candidate_tmp)
                _safe_unlink(graph_tmp)
                return 1
        candidate_tmp.replace(candidate_out)
        graph_tmp.replace(graph_out)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
