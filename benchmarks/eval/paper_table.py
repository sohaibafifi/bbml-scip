#!/usr/bin/env python3
"""Generate a learn2branch-style policy evaluation table from raw run JSON.

The output matches the structure of Table 2 in Gasse et al. (2019):
Easy / Medium / Hard blocks with Time / Wins / Nodes columns per difficulty.

Metrics:
- Time: 1-shifted geometric mean of solve times (unsolved clipped to time limit)
- Wins: number of fastest solved attempts over number of solved attempts
- Nodes: arithmetic mean of node counts on attempts solved by all displayed methods
- +/- : average per-instance relative std-dev across seeds (0% when only one seed)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TIME_LIMIT = 3600.0
SHIFT_TIME = 1.0
SOLVED_STATUSES = {"optimal", "infeasible"}
FAMILY_ORDER = ["sc", "ca", "cfl", "mis"]
DIFFICULTY_ORDER = ["easy_test", "medium_test", "hard_test"]
FAMILY_LABELS = {
    "sc": "Set Covering",
    "ca": "Combinatorial Auction",
    "cfl": "Capacitated Facility Location",
    "mis": "Maximum Independent Set",
}
DEFAULT_SOLVERS = ["strong-branch", "scip-default", "bbml-mlp", "bbml-gnn-graph"]
SOLVER_LABELS = {
    "strong-branch": "Strong branching",
    "scip-default": "SCIP default",
    "bbml-mlp": "BBML-MLP",
    "bbml-gnn-graph": "BBML-GNN",
    "bbml-gnn-varonly": "BBML-GNN-varonly",
}


@dataclass
class Cell:
    time_value: float | None
    time_std_pct: float | None
    wins: int
    solved: int
    nodes_value: float | None
    nodes_std_pct: float | None


def shifted_geo_mean(vals: Iterable[float], shift: float) -> float:
    arr = np.asarray(list(vals), dtype=float)
    return math.exp(np.mean(np.log(np.clip(arr + shift, 1e-12, None)))) - shift


def mean_relative_std(df: pd.DataFrame, value_col: str) -> float | None:
    rels: list[float] = []
    for _, group in df.groupby("instance_id", sort=False):
        vals = group[value_col].astype(float).to_numpy()
        if len(vals) <= 1:
            rels.append(0.0)
            continue
        mean = float(np.mean(vals))
        if abs(mean) <= 1e-12:
            continue
        rels.append(float(np.std(vals, ddof=0) / abs(mean) * 100.0))
    if not rels:
        return None
    return float(np.mean(rels))


def load_results(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(results_root.rglob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        if "runs" not in path.parts:
            continue
        try:
            record = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        required = {"solver", "seed", "instance_id", "instance_set", "status", "solve_time", "n_nodes"}
        if not required.issubset(record.keys()):
            continue
        record["_source_path"] = str(path)
        rows.append(record)
    if not rows:
        raise ValueError(f"No run JSON files found under {results_root}")
    df = pd.DataFrame(rows)
    key_cols = ["solver", "seed", "instance_set", "instance_id"]
    if all(col in df.columns for col in key_cols):
        df = df.sort_values(["_source_path"]).drop_duplicates(subset=key_cols, keep="last")
    return df.reset_index(drop=True)


def enrich(df: pd.DataFrame, time_limit: float) -> pd.DataFrame:
    out = df.copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["solve_time"] = pd.to_numeric(out["solve_time"], errors="coerce")
    out["n_nodes"] = pd.to_numeric(out["n_nodes"], errors="coerce")
    out = out.dropna(subset=["seed", "solve_time", "n_nodes", "instance_id", "instance_set", "solver", "status"]).copy()
    out["seed"] = out["seed"].astype(int)
    out["solve_time"] = out["solve_time"].astype(float)
    out["n_nodes"] = out["n_nodes"].astype(float)
    out["status"] = out["status"].astype(str)
    out["solved"] = out["status"].isin(SOLVED_STATUSES)
    out["solve_time_capped"] = np.where(out["solved"], out["solve_time"], time_limit)
    out["family"] = out["instance_set"].astype(str).str.split("_", n=1).str[0]
    out["difficulty"] = out["instance_set"].astype(str).str.split("_", n=1).str[1]
    return out


def compute_block(df: pd.DataFrame, solvers: list[str], family: str, difficulty: str) -> dict[str, Cell]:
    block = df[(df["family"] == family) & (df["difficulty"] == difficulty)].copy()
    if block.empty:
        return {}

    attempts = block[["instance_id", "seed"]].drop_duplicates()
    merged = attempts.copy()
    for solver in solvers:
        solved_col = f"solved__{solver}"
        s = block[block["solver"] == solver][["instance_id", "seed", "solved"]].rename(columns={"solved": solved_col})
        merged = merged.merge(s, on=["instance_id", "seed"], how="left")
        merged[solved_col] = merged[solved_col].fillna(False)

    common_mask = np.ones(len(merged), dtype=bool)
    for solver in solvers:
        common_mask &= merged[f"solved__{solver}"].to_numpy(dtype=bool)
    common_attempts = merged.loc[common_mask, ["instance_id", "seed"]]

    results: dict[str, Cell] = {}
    for solver in solvers:
        s_df = block[block["solver"] == solver].copy()
        if s_df.empty:
            continue
        solved_df = s_df[s_df["solved"]].copy()
        time_value = shifted_geo_mean(s_df["solve_time_capped"].to_numpy(), SHIFT_TIME)
        time_std_pct = mean_relative_std(s_df, "solve_time_capped")

        wins = 0
        solved = int(len(solved_df))
        for _, group in block.groupby(["instance_id", "seed"], sort=False):
            solved_group = group[group["solved"]]
            if solved_group.empty:
                continue
            min_time = float(solved_group["solve_time"].min())
            tol = 1e-9
            winners = solved_group[np.abs(solved_group["solve_time"] - min_time) <= tol]["solver"]
            if solver in winners.to_list():
                wins += 1

        common_df = s_df.merge(common_attempts, on=["instance_id", "seed"], how="inner")
        if common_df.empty:
            nodes_value = None
            nodes_std_pct = None
        else:
            nodes_value = float(common_df["n_nodes"].mean())
            nodes_std_pct = mean_relative_std(common_df, "n_nodes")

        results[solver] = Cell(
            time_value=time_value,
            time_std_pct=time_std_pct,
            wins=wins,
            solved=solved,
            nodes_value=nodes_value,
            nodes_std_pct=nodes_std_pct,
        )
    return results


def fmt_value_std(value: float | None, std_pct: float | None, integer: bool = False) -> str:
    if value is None:
        return r"n/a $\pm$ n/a\%"
    base = f"{value:.0f}" if integer else f"{value:.2f}"
    pct = 0.0 if std_pct is None else std_pct
    return rf"{base} $\pm$ {pct:.1f}\%"


def fmt_wins(cell: Cell) -> str:
    return f"{cell.wins} / {cell.solved}"


def bold(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return rf"\textbf{{{text}}}"


def build_table(df: pd.DataFrame, solvers: list[str], families: list[str], caption: str, label: str) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \multicolumn{3}{c}{\textbf{Easy}} & \multicolumn{3}{c}{\textbf{Medium}} & \multicolumn{3}{c}{\textbf{Hard}} \\")
    lines.append(r" & Time & Wins & Nodes & Time & Wins & Nodes & Time & Wins & Nodes \\")
    lines.append(r"\midrule")

    rendered_any = False
    for family in families:
        blocks = {difficulty: compute_block(df, solvers, family, difficulty) for difficulty in DIFFICULTY_ORDER}
        if not any(blocks.values()):
            continue
        rendered_any = True
        present_solvers = [solver for solver in solvers if any(solver in blocks[d] for d in DIFFICULTY_ORDER)]
        for solver in present_solvers:
            row = SOLVER_LABELS.get(solver, solver)
            for difficulty in DIFFICULTY_ORDER:
                block = blocks[difficulty]
                cell = block.get(solver)
                if cell is None:
                    row += r" & -- & -- & --"
                    continue
                time_best = min(v.time_value for v in block.values() if v.time_value is not None)
                wins_best = max(v.wins for v in block.values())
                node_candidates = [v.nodes_value for v in block.values() if v.nodes_value is not None]
                nodes_best = min(node_candidates) if node_candidates else None
                time_text = fmt_value_std(cell.time_value, cell.time_std_pct, integer=False)
                wins_text = fmt_wins(cell)
                nodes_text = fmt_value_std(cell.nodes_value, cell.nodes_std_pct, integer=True)
                row += " & " + bold(time_text, cell.time_value is not None and abs(cell.time_value - time_best) <= 1e-9)
                row += " & " + bold(wins_text, cell.wins == wins_best and wins_best > 0)
                row += " & " + bold(
                    nodes_text,
                    nodes_best is not None and cell.nodes_value is not None and abs(cell.nodes_value - nodes_best) <= 1e-9,
                )
            lines.append(row + r" \\")
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{10}{c}{\textbf{" + FAMILY_LABELS.get(family, family) + r"}} \\")
        lines.append(r"\midrule")

    if rendered_any and lines[-1] == r"\midrule":
        lines.pop()
    elif not rendered_any:
        lines.append(r"\multicolumn{10}{c}{No matching families/results found.} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def compile_pdf(tex_path: Path, pdf_path: Path) -> None:
    wrapper_name = f"{pdf_path.stem}_standalone.tex"
    wrapper_path = tex_path.parent / wrapper_name
    wrapper_path.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\usepackage[landscape,margin=0.5in]{geometry}",
                r"\usepackage{booktabs}",
                r"\usepackage{array}",
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{lmodern}",
                r"\pagestyle{empty}",
                "",
                r"\begin{document}",
                rf"\input{{{tex_path.name}}}",
                r"\end{document}",
                "",
            ]
        )
    )
    jobname = pdf_path.stem
    for _ in range(2):
        subprocess.run(
            [
                "pdflatex",
                f"-jobname={jobname}",
                "-interaction=nonstopmode",
                "-halt-on-error",
                wrapper_path.name,
            ],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path, help="Root directory containing raw run JSON")
    ap.add_argument("--out", required=True, type=Path, help="Output .tex path")
    ap.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS), help="Comma-separated solver ids to include")
    ap.add_argument("--families", default=",".join(FAMILY_ORDER), help="Comma-separated family ids to include")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--compile-pdf", action="store_true", help="Also compile a standalone PDF next to --out")
    ap.add_argument(
        "--caption",
        default=("Policy evaluation in the style of Gasse et al. (2019). " "Time is a 1-shifted geometric mean in seconds, Wins is fastest solved attempts over solved attempts, " "and Nodes is the mean node count on attempts solved by all displayed methods."),
    )
    ap.add_argument("--label", default="tab:policy_eval_paper")
    args = ap.parse_args()

    df = enrich(load_results(args.results), args.time_limit)
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    requested_families = [f.strip() for f in args.families.split(",") if f.strip()]
    families = [f for f in FAMILY_ORDER if f in requested_families and (df["family"] == f).any()]
    tex = build_table(df, solvers, families, args.caption, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"Wrote {args.out}")
    if args.compile_pdf:
        pdf_path = args.out.with_suffix(".pdf")
        compile_pdf(args.out, pdf_path)
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
