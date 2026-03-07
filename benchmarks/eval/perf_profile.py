#!/usr/bin/env python3
"""Dolan-Moré performance profiles for BBML benchmarks.

Generates one PDF/PNG per metric (time, nodes) with one curve per solver.
A point (tau, rho) on a curve means the solver solved rho fraction of
instances within a factor tau of the best solver on that instance.

Usage:
    uv run python benchmarks/eval/perf_profile.py \\
        --results results/runs \\
        --out paper/figures/ \\
        --metric time nodes \\
        --solvers scip-default bbml-mlp bbml-gnn-graph
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_LIMIT = 3600.0

SOLVER_STYLES = {
    "scip-default": dict(color="#555555", ls="--", lw=1.5, label="SCIP default"),
    "strong-branch": dict(color="#000000", ls=":", lw=1.5, label="Strong branching (oracle)"),
    "bbml-mlp": dict(color="#D55E00", ls="-", lw=1.5, label="BBML MLP"),
    "bbml-gnn-varonly": dict(color="#0072B2", ls="-.", lw=1.5, label="BBML GNN var-only"),
    "bbml-gnn-graph": dict(color="#009E73", ls="-", lw=2.5, label="BBML GNN graph"),
    "bbml-gnn-graph-fp32": dict(color="#CC79A7", ls="--", lw=1.5, label="BBML GNN FP32"),
    "bbml-gnn-graph-fp16": dict(color="#F0E442", ls="-", lw=1.5, label="BBML GNN FP16"),
}


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
    return pd.DataFrame(rows)


def dolan_more_profile(
    df: pd.DataFrame,
    metric: str,
    solvers: list[str],
    tau_max: float = 10.0,
    n_tau: int = 1000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute Dolan-Moré profile curves.

    Returns {solver: (tau_array, rho_array)}.
    """
    # Pivot: rows=instance+seed, cols=solver
    key = ["instance_id", "seed"]
    pivot = df[df["solver"].isin(solvers)].pivot_table(index=key, columns="solver", values=metric, aggfunc="mean")
    # Best solver per instance (minimum)
    best = pivot.min(axis=1)
    # Ratios
    ratios = pivot.divide(best, axis=0)
    ratios = ratios.replace([np.inf, -np.inf], tau_max + 1)
    ratios = ratios.fillna(tau_max + 1)

    taus = np.logspace(0, np.log10(tau_max), n_tau)
    curves = {}
    for solver in solvers:
        if solver not in ratios.columns:
            continue
        r = ratios[solver].values
        rho = np.array([(r <= t).mean() for t in taus])
        curves[solver] = (taus, rho)
    return curves


def plot_profile(
    curves: dict,
    metric: str,
    out_path: Path,
    solvers: list[str],
):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    tau_end = 10.0
    for solver in solvers:
        if solver not in curves:
            continue
        taus, rho = curves[solver]
        tau_end = float(taus[-1])
        style = SOLVER_STYLES.get(solver, dict(color="gray", ls="-", lw=1.5, label=solver))
        ax.step(taus, rho, where="post", **style)

    ax.set_xscale("log")
    ax.set_xlim(1.0, tau_end)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Performance ratio $\tau$")
    metric_label = "Solve time" if metric == "solve_time" else "Node count"
    ax.set_ylabel(f"Fraction of instances [$\\rho_s(\\tau)$]")
    ax.set_title(f"Performance profile — {metric_label}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Output directory for figures")
    ap.add_argument("--metric", nargs="+", default=["time", "nodes"], choices=["time", "nodes"])
    ap.add_argument("--solvers", nargs="*", default=None, help="Subset of solvers to include (default: all)")
    ap.add_argument("--tau-max", type=float, default=10.0)
    ap.add_argument("--instance-set", default=None, help="Filter to a specific instance set tag")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results} ...")
    df = load_results(args.results)
    df["solve_time"] = df["solve_time"].clip(upper=TIME_LIMIT)
    df["n_nodes"] = df["n_nodes"].clip(lower=0)

    if args.instance_set:
        df = df[df.get("instance_set", pd.Series(["all"] * len(df))) == args.instance_set]

    all_solvers = sorted(set(str(s) for s in df["solver"].tolist()))
    solvers = args.solvers or all_solvers
    print(f"  Solvers: {solvers}")

    metric_map = {"time": "solve_time", "nodes": "n_nodes"}
    for m in args.metric:
        col = metric_map[m]
        if col not in df.columns:
            print(f"  WARNING: column {col} not in data; skipping.")
            continue
        print(f"Computing profile for {col} ...")
        df_main: pd.DataFrame = df  # type: ignore[assignment]
        curves = dolan_more_profile(df_main, col, solvers, tau_max=args.tau_max)
        fname = args.out / f"perf_profile_{m}.pdf"
        plot_profile(curves, col, fname, solvers)

    print("Done.")


if __name__ == "__main__":
    main()
