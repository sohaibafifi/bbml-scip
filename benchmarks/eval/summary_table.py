#!/usr/bin/env python3
"""Generate LaTeX result tables from KPI CSV.

Reads the output of kpis.py and produces:
  - Table 1: main comparison (SGM time + nodes, all instance sets)
  - Table 2: ablation results

Usage:
    uv run python benchmarks/eval/summary_table.py \\
        --kpis results/kpis.csv \\
        --out paper/tables/
"""
import argparse
from pathlib import Path

import pandas as pd

# Display order for solvers in Table 1
SOLVER_ORDER = [
    "scip-default",
    "strong-branch",
    "gcnn-gasse19",
    "pure-imitation",
    "alpha-fixed-0.5",
    "tabular-xgb",
    "bbml-nonode",
    "bbml-full",
]

SOLVER_LABELS = {
    "scip-default": r"\texttt{scip-default}",
    "strong-branch": r"\texttt{strong-branch}",
    "gcnn-gasse19": r"\texttt{gcnn-gasse19}~\citep{gasse2019exact}",
    "pure-imitation": r"\texttt{pure-imitation} ($\alpha{=}1$)",
    "alpha-fixed-0.5": r"\texttt{alpha-fixed} ($\alpha{=}0.5$)",
    "tabular-xgb": r"\texttt{tabular-xgb}",
    "bbml-nonode": r"\texttt{bbml-nonode}",
    "bbml-full": r"\textbf{\texttt{bbml-full}} (ours)",
}

ABLATION_ORDER = [
    "a1_gnn_full",
    "a1_gnn_varonly",
    "a1_tabular_xgb",
    "a2_adaptive",
    "a2_fixed_05",
    "a2_fixed_10",
    "a2_fixed_00",
    "a3_with_temp",
    "a3_no_temp",
    "a4_with_cond_gate",
    "a4_no_cond_gate",
    "a5_fp32",
    "a5_fp16",
    "a6_with_nodesel",
    "a6_no_nodesel",
]

ABLATION_LABELS = {
    "a1_gnn_full": r"\textbf{A1} Full bipartite GNN",
    "a1_gnn_varonly": r"\textbf{A1} Var-only GNN (no constraints)",
    "a1_tabular_xgb": r"\textbf{A1} Tabular XGBoost",
    "a2_adaptive": r"\textbf{A2} Adaptive $\alpha$ (ours)",
    "a2_fixed_05": r"\textbf{A2} Fixed $\alpha{=}0.5$",
    "a2_fixed_10": r"\textbf{A2} Fixed $\alpha{=}1.0$ (pure ML)",
    "a2_fixed_00": r"\textbf{A2} Fixed $\alpha{=}0$ (solver only)",
    "a3_with_temp": r"\textbf{A3} With temperature calibration",
    "a3_no_temp": r"\textbf{A3} Without temperature ($T{=}1$)",
    "a4_with_cond_gate": r"\textbf{A4} With cond.-number gate",
    "a4_no_cond_gate": r"\textbf{A4} Without cond.-number gate",
    "a5_fp32": r"\textbf{A5} FP32 ONNX",
    "a5_fp16": r"\textbf{A5} FP16 ONNX",
    "a6_with_nodesel": r"\textbf{A6} With node selector",
    "a6_no_nodesel": r"\textbf{A6} Without node selector",
}

ISET_LABELS = {
    "vrp_train": "VRP-S",
    "vrp_test": "VRP-M/L",
    "miplib_easy": "MIPLIB easy",
    "miplib_medium": "MIPLIB medium",
    "setcover": "Set-cover",
    "all": "All",
}


def sig_marker(row: pd.Series) -> str:
    markers = ""
    if row.get("sig_time", False):
        markers += r"$^\dagger$"
    if row.get("sig_nodes", False):
        markers += r"$^\ddagger$"
    return markers


def format_cell(val: float, best: float, sig: str = "", bold_if_best: bool = True) -> str:
    s = f"{val:.1f}"
    if bold_if_best and abs(val - best) < 0.1:
        s = r"\textbf{" + s + "}"
    return s + sig


def build_main_table(df: pd.DataFrame) -> str:
    present_isets = set(df["instance_set"].to_numpy().tolist())
    present_solvers_main = set(df["solver"].to_numpy().tolist())
    isets = [c for c in ["vrp_train", "vrp_test", "miplib_easy", "miplib_medium"] if c in present_isets]
    solvers = [s for s in SOLVER_ORDER if s in present_solvers_main]
    n_isets = len(isets)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"  Shifted geometric mean (SGM, shift=10) of solve time (s) and B\&B node count.")
    lines.append(r"  Lower is better. Bold: best per column.")
    lines.append(r"  $\dagger$/$\ddagger$: Wilcoxon $p < 0.05$ vs.\ \texttt{scip-default} for time/nodes.")
    lines.append(r"}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")

    col_spec = "l" + "cc" * n_isets
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1
    hdr1 = r"\textbf{Method}"
    for iset in isets:
        label = ISET_LABELS.get(iset, iset)
        hdr1 += r" & \multicolumn{2}{c}{\textbf{" + label + r"}}"
    lines.append(hdr1 + r" \\")

    # Header row 2: cmidrules + column labels
    for iset in isets:
        col_start = 2 + isets.index(iset) * 2
        lines.append(r"\cmidrule(lr){" + str(col_start) + "-" + str(col_start + 1) + "}")
    lines.append(r"\textbf{} " + r" & Time & Nodes" * n_isets + r" \\")
    lines.append(r"\midrule")

    # Compute per-column bests
    best_time = {}
    best_nodes = {}
    for iset in isets:
        sub: pd.DataFrame = df[df["instance_set"] == iset]  # type: ignore[assignment]
        best_time[iset] = float(sub["sgm_time"].to_numpy().min())
        best_nodes[iset] = float(sub["sgm_nodes"].to_numpy().min())

    for solver in solvers:
        label = SOLVER_LABELS.get(solver, r"\texttt{" + solver + "}")
        row_str = label
        for iset in isets:
            cell_sub: pd.DataFrame = df[(df["instance_set"] == iset) & (df["solver"] == solver)]  # type: ignore[assignment]
            cell_records = cell_sub.to_dict("records")  # type: ignore[arg-type]
            if not cell_records:
                row_str += r" & -- & --"
            else:
                row = cell_records[0]
                sig = sig_marker(pd.Series(row))
                t_cell = format_cell(float(row["sgm_time"]), best_time[iset], sig)
                n_cell = format_cell(float(row["sgm_nodes"]), best_nodes[iset])
                row_str += f" & {t_cell} & {n_cell}"
        lines.append(row_str + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_ablation_table(df: pd.DataFrame, iset: str = "vrp_test") -> str:
    sub: pd.DataFrame = df[df["instance_set"] == iset]  # type: ignore[assignment]
    present_solvers = set(sub["solver"].to_numpy().tolist())
    ablations = [a for a in ABLATION_ORDER if a in present_solvers]
    if not ablations:
        return "% No ablation data found.\n"

    best_t = float(sub["sgm_time"].to_numpy().min())
    best_n = float(sub["sgm_nodes"].to_numpy().min())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation results on VRP-MIP test instances. SGM(10). Lower is better.}")
    lines.append(r"\label{tab:ablations}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{SGM Time} & \textbf{SGM Nodes} \\")
    lines.append(r"\midrule")

    current_group = None
    for abl in ablations:
        group = abl.split("_")[0].upper()
        if group != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            current_group = group
        label = ABLATION_LABELS.get(abl, abl)
        abl_sub: pd.DataFrame = sub[sub["solver"] == abl]  # type: ignore[assignment]
        records = abl_sub.to_dict("records")  # type: ignore[arg-type]
        if not records:
            lines.append(f"{label} & -- & -- \\\\")
        else:
            r = records[0]
            t_cell = format_cell(float(r["sgm_time"]), best_t)
            n_cell = format_cell(float(r["sgm_nodes"]), best_n)
            lines.append(f"{label} & {t_cell} & {n_cell} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kpis", required=True, type=Path, help="KPI CSV from kpis.py")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for .tex files")
    ap.add_argument("--ablation-set", default="vrp_test", help="Instance set to use for ablation table")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.kpis)
    print(f"Loaded {len(df)} KPI rows from {args.kpis}")

    # Table 1: main comparison
    main_tex = build_main_table(df)
    out1 = args.out / "table_main.tex"
    out1.write_text(main_tex)
    print(f"  Written: {out1}")

    # Table 2: ablations
    abl_tex = build_ablation_table(df, iset=args.ablation_set)
    out2 = args.out / "table_ablations.tex"
    out2.write_text(abl_tex)
    print(f"  Written: {out2}")

    print("Done.")


if __name__ == "__main__":
    main()
