#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import gzip
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bbml.data.telemetry_compact import compact_collection_outputs, count_graph_samples, trim_compacted_outputs

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


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


def _format_budget_progress(split: str, used: int, budget: int, *, kept: int, current: int) -> str:
    pct = 100.0 if budget <= 0 else (100.0 * used / budget)
    return f"BUDGET_PROGRESS split={split} used={used}/{budget} " f"pct={pct:.1f} kept={kept}/{current}"


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
    ap.add_argument("--telemetry-max-nodes-per-instance", type=int, default=0)
    ap.add_argument("--telemetry-query-expert-prob", type=float, default=1.0)
    ap.add_argument("--sample-budget-state", default=None)
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
        tmp.write(f"bbml/telemetry/query_expert_prob = {args.telemetry_query_expert_prob:.17g}\n")
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
            if graph_out.suffix == ".pt" or args.telemetry_max_nodes_per_instance > 0:
                stats = compact_collection_outputs(
                    candidate_src=candidate_tmp,
                    graph_src=graph_tmp,
                    candidate_out=candidate_out,
                    graph_out=graph_out,
                    max_graph_nodes=args.telemetry_max_nodes_per_instance,
                    seed=args.seed,
                )
                _safe_unlink(candidate_tmp)
                _safe_unlink(graph_tmp)
                print(
                    "collect_task compacted telemetry " f"graph={stats.graph_nodes_kept}/{stats.graph_nodes_total} " f"candidate_rows={stats.candidate_rows_kept}/{stats.candidate_rows_total}",
                    file=sys.stderr,
                )
            elif candidate_out.suffix == ".gz":
                with candidate_tmp.open("rb") as src, gzip.open(candidate_out, "wb") as dst:
                    dst.write(src.read())
                _safe_unlink(candidate_tmp)
                graph_tmp.replace(graph_out)
            else:
                candidate_tmp.replace(candidate_out)
                graph_tmp.replace(graph_out)
            if args.sample_budget_state:
                budget_summary = _apply_sample_budget(
                    candidate_out=candidate_out,
                    graph_out=graph_out,
                    done_marker=done_marker,
                    state_path=Path(args.sample_budget_state),
                )
                if budget_summary:
                    print(budget_summary, file=sys.stderr, flush=True)
            if candidate_out.is_file() and graph_out.is_file():
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


def _apply_sample_budget(candidate_out: Path, graph_out: Path, done_marker: Path, state_path: Path) -> str | None:
    if fcntl is None:
        return None
    if not graph_out.is_file() or graph_out.stat().st_size <= 0:
        return None
    state_path.parent.mkdir(parents=True, exist_ok=True)
    summary = None
    with state_path.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read().strip()
        payload = json.loads(raw) if raw else {}
        budget = int(payload.get("budget", 0))
        used = int(payload.get("used", 0))
        split = str(payload.get("split", state_path.stem))
        report_step = max(1, int(payload.get("report_step_pct", 5)))
        reported_bucket = int(payload.get("reported_bucket", 0))
        current = count_graph_samples(graph_out)
        remaining = max(0, budget - used)
        kept = current
        if budget > 0:
            if remaining <= 0:
                _safe_unlink(candidate_out)
                _safe_unlink(graph_out)
                done_marker.write_text("budget_exhausted\n")
                kept = 0
            elif current > remaining:
                kept = trim_compacted_outputs(candidate_out, graph_out, remaining)
            used += kept
            if used > budget:
                used = budget
            pct = 100.0 * used / budget if budget > 0 else 100.0
            current_bucket = min(100, int(pct)) // report_step
            if current_bucket > reported_bucket or used >= budget:
                payload["reported_bucket"] = current_bucket
                summary = _format_budget_progress(split, used, budget, kept=kept, current=current)
        payload["used"] = used
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
