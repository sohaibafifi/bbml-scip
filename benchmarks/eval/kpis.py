#!/usr/bin/env python3
"""Compute benchmark KPIs from SCIP solve logs.

Reads all per-run JSON files under --results, computes per-solver KPIs:
  - Shifted geometric mean (SGM) of solve time and node count
  - % instances solved to optimality
  - Wins / ties / losses vs. a chosen baseline
  - Wilcoxon signed-rank test vs. a chosen baseline

Output: CSV with one row per (solver, instance_set).

Usage:
    uv run python benchmarks/eval/kpis.py \\
        --results results/runs \\
        --instance-sets benchmarks/instances \\
        --baseline scip-default \\
        --out results/kpis.csv
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TIME_LIMIT = 3600.0
SHIFT_TIME = 10.0  # seconds (standard MIPLIB convention)
SHIFT_NODES = 10  # nodes


def shifted_geo_mean(vals: np.ndarray, shift: float) -> float:
    return math.exp(np.mean(np.log(np.clip(vals + shift, 1e-9, None)))) - shift


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.rglob("*.json")) + sorted(results_dir.rglob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise ValueError(f"No result records found in {results_dir}")
    return pd.DataFrame(rows)


def assign_instance_set(df: pd.DataFrame, instances_dir: Path) -> pd.DataFrame:
    """Tag each row with its instance set name using the list files."""
    set_map = {}
    for lst in instances_dir.glob("*.txt"):
        set_name = lst.stem
        with open(lst) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                iid = Path(line).stem
                # Remove common extensions
                for ext in (".lp", ".mps", ".gz"):
                    iid = iid.removesuffix(ext)
                set_map[iid] = set_name
    df = df.copy()
    df["instance_set"] = df["instance_id"].map(set_map).fillna("unknown")
    return df


def compute_kpis(
    df: pd.DataFrame,
    baseline: str = "scip-default",
    wtl_threshold: float = 0.05,
) -> pd.DataFrame:
    df = df.copy()
    # Cap timed-out runs
    df["solve_time"] = df["solve_time"].clip(upper=TIME_LIMIT)
    # For failed/unknown status, treat as timed out
    df.loc[~df["status"].isin(["optimal", "infeasible"]), "solve_time"] = TIME_LIMIT
    df.loc[~df["status"].isin(["optimal", "infeasible"]), "n_nodes"] = df["n_nodes"].max()

    solvers = sorted(df["solver"].unique())
    instance_sets = ["all", *sorted(df["instance_set"].unique())]
    base_df = df[df["solver"] == baseline]

    records = []
    for iset in instance_sets:
        iset_df = df[df["instance_set"] == iset]
        base_iset = base_df[base_df["instance_set"] == iset]
        if iset == "all":
            iset_df = df
            base_iset = base_df

        for solver in solvers:
            s_df = iset_df[iset_df["solver"] == solver]
            if s_df.empty:
                continue

            sgm_t = shifted_geo_mean(s_df["solve_time"].values, SHIFT_TIME)
            sgm_n = shifted_geo_mean(s_df["n_nodes"].values.clip(min=0), SHIFT_NODES)
            solved_pct = (s_df["status"].isin(["optimal", "infeasible"])).mean() * 100

            p_time = p_nodes = 1.0
            wins = ties = losses = 0
            if solver != baseline and not base_iset.empty:
                merged = s_df.merge(
                    base_iset[["instance_id", "seed", "solve_time", "n_nodes"]],
                    on=["instance_id", "seed"],
                    suffixes=("", "_base"),
                )
                if len(merged) > 4:
                    try:
                        _, p_time = wilcoxon(
                            merged["solve_time"],
                            merged["solve_time_base"],
                            alternative="two-sided",
                        )
                        _, p_nodes = wilcoxon(
                            merged["n_nodes"].clip(min=0),
                            merged["n_nodes_base"].clip(min=0),
                            alternative="two-sided",
                        )
                    except Exception:
                        pass
                for _, row in merged.iterrows():
                    base_time = float(row["solve_time_base"])
                    solve_time = float(row["solve_time"])
                    if base_time <= 1e-9:
                        if solve_time <= 1e-9:
                            ties += 1
                        else:
                            losses += 1
                        continue
                    rel = (base_time - solve_time) / base_time
                    if rel > wtl_threshold:
                        wins += 1
                    elif rel < -wtl_threshold:
                        losses += 1
                    else:
                        ties += 1
            elif solver == baseline:
                ties = len(s_df)

            records.append(
                dict(
                    instance_set=iset,
                    solver=solver,
                    n_runs=len(s_df),
                    sgm_time=round(sgm_t, 2),
                    sgm_nodes=round(sgm_n, 1),
                    solved_pct=round(solved_pct, 1),
                    p_time=round(p_time, 4),
                    p_nodes=round(p_nodes, 4),
                    sig_time=p_time < 0.05,
                    sig_nodes=p_nodes < 0.05,
                    wtl_wins=wins,
                    wtl_ties=ties,
                    wtl_losses=losses,
                    wtl=f"{wins}/{ties}/{losses}",
                    wtl_win_pct=round((wins / max(1, wins + ties + losses)) * 100, 1),
                )
            )

    return pd.DataFrame(records).sort_values(["instance_set", "sgm_time"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path, help="Dir with recursive per-run JSON files")
    ap.add_argument("--instance-sets", type=Path, default=None, help="Dir with instance list *.txt files (for set tagging)")
    ap.add_argument("--baseline", default="scip-default", help="Reference solver for Wilcoxon")
    ap.add_argument("--wtl-threshold", type=float, default=0.05, help="Relative threshold for wins/ties/losses vs baseline")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path")
    args = ap.parse_args()

    print(f"Loading results from {args.results} ...")
    df = load_results(args.results)
    print(f"  {len(df)} runs, {df['solver'].nunique()} solvers, {df['instance_id'].nunique()} instances")

    if args.instance_sets and args.instance_sets.exists():
        df = assign_instance_set(df, args.instance_sets)
    else:
        df["instance_set"] = "all"

    kpis = compute_kpis(df, baseline=args.baseline, wtl_threshold=args.wtl_threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kpis.to_csv(args.out, index=False)
    print(f"\nKPIs written to {args.out}")
    print(kpis.to_string(index=False))


if __name__ == "__main__":
    main()
