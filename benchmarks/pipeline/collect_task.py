#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
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


def _completed_log(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return "BBML_SUMMARY" in path.read_text(errors="ignore")
    except OSError:
        return False


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
    ap.add_argument("--telemetry-strongbranch", action="store_true")
    ap.add_argument(
        "--telemetry-oracle",
        choices=("none", "strongbranch", "vanillafullstrong"),
        default="vanillafullstrong",
    )
    args = ap.parse_args()

    candidate_out = Path(args.candidate_out)
    graph_out = Path(args.graph_out)
    scip_log = Path(args.scip_log)
    done_marker = candidate_out.with_suffix(candidate_out.suffix + ".done")
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
        tmp.write(f"bbml/telemetry/oracle = {_quote(args.telemetry_oracle)}\n")
        tmp.write(f"bbml/telemetry/strongbranch = {'TRUE' if args.telemetry_strongbranch else 'FALSE'}\n")
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
        _safe_unlink(done_marker)
        print(
            f"collect_task failed exit={proc.returncode} instance={args.instance} " f"seed={args.seed} scip_log={scip_log}",
            file=sys.stderr,
        )
    else:
        candidate_ok = candidate_tmp.is_file() and candidate_tmp.stat().st_size > 0
        graph_ok = graph_tmp.is_file() and graph_tmp.stat().st_size > 0
        if candidate_ok and graph_ok:
            candidate_tmp.replace(candidate_out)
            graph_tmp.replace(graph_out)
            _safe_unlink(done_marker)
        else:
            _safe_unlink(candidate_tmp)
            _safe_unlink(graph_tmp)
            if not _completed_log(scip_log):
                _safe_unlink(done_marker)
                print(
                    f"collect_task failed: missing telemetry output instance={args.instance} " f"seed={args.seed} scip_log={scip_log}",
                    file=sys.stderr,
                )
                return 1
            done_marker.write_text("no_branch_telemetry\n")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
