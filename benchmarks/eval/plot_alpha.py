#!/usr/bin/env python3
"""Experimental alpha-log plotting helper.

Reads alpha log CSVs produced by custom branchrule instrumentation.
The default benchmark pipeline does not emit these logs.
Expected CSV columns: depth, alpha, conf, cond_est, fallback

Usage:
    uv run python benchmarks/eval/plot_alpha.py \\
        --logs results/alpha_logs \\
        --out paper/figures/ \\
        --solver bbml-gnn-graph
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_DEPTH = 60  # clip for display


def load_alpha_logs(logs_dir: Path, solver: str) -> pd.DataFrame:
    frames = []
    pattern = f"*_{solver}_s*.csv"
    for f in sorted(logs_dir.glob(pattern)):
        try:
            df = pd.read_csv(f)
            df["run_file"] = f.stem
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: could not load {f}: {e}")
    if not frames:
        raise FileNotFoundError(f"No alpha log files matching '{pattern}' in {logs_dir}")
    return pd.concat(frames, ignore_index=True)


def plot_alpha_vs_depth(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df["depth"] = df["depth"].clip(upper=MAX_DEPTH)

    stats = df.groupby("depth")["alpha"].agg(["mean", "std"]).reset_index()
    depths = stats["depth"].tolist()
    means = np.array(stats["mean"].tolist())
    stds = np.array(stats["std"].fillna(0).tolist())

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(depths, means, color="#009E73", lw=2.0, label=r"Mean $\alpha(N)$")
    ax.fill_between(
        depths,
        np.clip(means - stds, 0, 1),
        np.clip(means + stds, 0, 1),
        alpha=0.2,
        color="#009E73",
        label=r"$\pm$1 std",
    )
    ax.axhline(0.1, color="gray", ls=":", lw=1.0, label=r"$\alpha_{\min}=0.1$")
    ax.axhline(0.8, color="gray", ls="--", lw=1.0, label=r"$\alpha_{\max}=0.8$")
    ax.set_xlabel("Tree depth")
    ax.set_ylabel(r"$\alpha(N)$")
    ax.set_title(r"Adaptive blend weight vs.\ tree depth")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_fallback_rate(df: pd.DataFrame, out_path: Path):
    """Plot fraction of nodes where hard fallback (cond_est gate) fires, by depth."""
    if "fallback" not in df.columns:
        if "cond_est" in df.columns:
            df = df.copy()
            df["fallback"] = (df["cond_est"] > 1e8).astype(float)
        else:
            print("  WARNING: no 'fallback' or 'cond_est' column; skipping fallback plot.")
            return

    df = df.copy()
    # Bucket depths
    bins = [0, 5, 10, 20, 35, MAX_DEPTH]
    labels = ["0-4", "5-9", "10-19", "20-34", "35+"]
    df["depth_band"] = pd.cut(
        df["depth"].clip(upper=MAX_DEPTH),
        bins=bins,
        labels=labels,
        right=False,
    )
    fallback_np = df["fallback"].to_numpy(dtype=float)
    band_np = df["depth_band"].to_numpy()
    unique_bands = list(dict.fromkeys(band_np))  # ordered unique values
    rate_labels = [str(b) for b in unique_bands]
    rate_vals = [float(fallback_np[band_np == b].mean()) * 100 for b in unique_bands]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(rate_labels, rate_vals, color="#0072B2", alpha=0.8)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax.set_xlabel("Tree depth band")
    ax.set_ylabel("Fallback activation rate (%)")
    ax.set_title(r"Cond.-number fallback by depth")
    ax.set_ylim(0, max(rate_vals) * 1.3 + 0.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confidence_dist(df: pd.DataFrame, out_path: Path):
    if "conf" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(df["conf"].dropna().clip(0, 1), bins=50, color="#009E73", alpha=0.7, edgecolor="white")
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label=r"$c=0.5$ (default confidence)")
    ax.set_xlabel(r"Node confidence $\hat{c}(N)$")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", required=True, type=Path, help="Dir with alpha log CSVs")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for figures")
    ap.add_argument("--solver", default="bbml-gnn-graph", help="Solver tag used in filenames")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading alpha logs for solver='{args.solver}' from {args.logs} ...")
    df = load_alpha_logs(args.logs, args.solver)
    print(f"  {len(df):,} branching decisions loaded.")

    plot_alpha_vs_depth(df, args.out / "alpha_vs_depth.pdf")
    plot_fallback_rate(df, args.out / "fallback_rate.pdf")
    plot_confidence_dist(df, args.out / "confidence_dist.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
