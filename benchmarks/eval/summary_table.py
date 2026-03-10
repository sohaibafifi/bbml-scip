#!/usr/bin/env python3
"""Generate LaTeX result tables from KPI CSV.

Reads the output of kpis.py and produces:
  - Table 1: main comparison (SGM time + nodes, all instance sets)
  - Table 2: ablation results
  - Table 3: wins / ties / losses vs. baseline
  - Table 4: optional branching top-1 accuracy

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
    "bbml-mlp",
    "bbml-gnn-varonly",
    "bbml-gnn-graph",
    "bbml-gnn-graph-fp32",
    "bbml-gnn-graph-fp16",
]

SOLVER_LABELS = {
    "scip-default": r"\texttt{scip-default}",
    "strong-branch": r"\texttt{strong-branch}",
    "bbml-mlp": r"\texttt{bbml-mlp}",
    "bbml-gnn-varonly": r"\texttt{bbml-gnn-varonly}",
    "bbml-gnn-graph": r"\textbf{\texttt{bbml-gnn-graph}}",
    "bbml-gnn-graph-fp32": r"\texttt{bbml-gnn-graph-fp32}",
    "bbml-gnn-graph-fp16": r"\texttt{bbml-gnn-graph-fp16}",
}

ABLATION_ORDER = [
    "bbml-gnn-graph",
    "alpha-fixed-0.5",
    "pure-imitation",
    "solver-only-alpha0",
    "no-temperature",
    "no-cond-gate",
    "no-conf-gate",
    "bbml-mlp",
    "bbml-gnn-varonly",
    "bbml-gnn-graph-fp32",
    "bbml-gnn-graph-fp16",
]

ABLATION_LABELS = {
    "bbml-gnn-graph": r"Graph GNN",
    "alpha-fixed-0.5": r"Fixed $\alpha = 0.5$",
    "pure-imitation": r"Pure imitation ($\alpha = 1.0$)",
    "solver-only-alpha0": r"Solver-only blend ($\alpha = 0.0$)",
    "no-temperature": r"No temperature calibration",
    "no-cond-gate": r"No condition gate",
    "no-conf-gate": r"No confidence gate",
    "bbml-mlp": r"MLP ranker",
    "bbml-gnn-varonly": r"Var-only GNN",
    "bbml-gnn-graph-fp32": r"Graph GNN FP32",
    "bbml-gnn-graph-fp16": r"Graph GNN FP16",
}

ISET_LABELS = {
    "sc_test": "Set Cover",
    "ca_test": "Comb. Auction",
    "cfl_test": "Facility Loc.",
    "mis_test": "MIS",
    "all": "All",
}

FAMILY_LABELS = {
    "sc": "Set Cover",
    "ca": "Comb. Auction",
    "cfl": "Facility Loc.",
    "mis": "MIS",
}

DIFFICULTY_LABELS = {
    "easy_test": "Easy",
    "medium_test": "Medium",
    "hard_test": "Hard",
    "test": "",
}


def iset_label(name: str) -> str:
    if name in ISET_LABELS:
        return ISET_LABELS[name]
    if name == "all":
        return "All"
    if "_" not in name:
        return name
    family, suffix = name.split("_", 1)
    family_label = FAMILY_LABELS.get(family)
    difficulty = DIFFICULTY_LABELS.get(suffix)
    if family_label and difficulty is not None:
        if difficulty:
            return f"{family_label} ({difficulty})"
        return family_label
    return name


def iset_sort_key(name: str) -> tuple[int, int, int, str]:
    family_order = {"sc": 0, "ca": 1, "cfl": 2, "mis": 3}
    difficulty_order = {"test": 0, "easy_test": 1, "medium_test": 2, "hard_test": 3}
    if name == "all":
        return (99, 99, 0, name)
    if "_" not in name:
        return (98, 98, 0, name)
    family, suffix = name.split("_", 1)
    return (
        family_order.get(family, 97),
        difficulty_order.get(suffix, 97),
        0,
        name,
    )


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
    preferred_isets = [
        "sc_test",
        "ca_test",
        "cfl_test",
        "mis_test",
        "sc_easy_test",
        "sc_medium_test",
        "sc_hard_test",
        "ca_easy_test",
        "ca_medium_test",
        "ca_hard_test",
        "cfl_easy_test",
        "cfl_medium_test",
        "cfl_hard_test",
        "mis_easy_test",
        "mis_medium_test",
        "mis_hard_test",
    ]
    isets = [c for c in preferred_isets if c in present_isets]
    if not isets:
        isets = sorted(present_isets, key=iset_sort_key)
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
        label = iset_label(iset)
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


def build_ablation_table(df: pd.DataFrame, iset: str = "sc_test") -> str:
    sub: pd.DataFrame = df[df["instance_set"] == iset]  # type: ignore[assignment]
    if sub.empty:
        test_isets = [name for name in df["instance_set"].to_numpy().tolist() if name != "all"]
        if test_isets:
            iset = sorted(set(test_isets), key=iset_sort_key)[0]
            sub = df[df["instance_set"] == iset]  # type: ignore[assignment]
    present_solvers = set(sub["solver"].to_numpy().tolist())
    ablations = [a for a in ABLATION_ORDER if a in present_solvers]
    if not ablations:
        return "% No ablation data found.\n"

    best_t = float(sub["sgm_time"].to_numpy().min())
    best_n = float(sub["sgm_nodes"].to_numpy().min())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Low-cost ablations on " + iset_label(iset) + r". SGM(10). Lower is better.}")
    lines.append(r"\label{tab:ablations}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{SGM Time} & \textbf{SGM Nodes} \\")
    lines.append(r"\midrule")

    current_group = None
    for abl in ablations:
        group = "GRAPH"
        if abl in {"bbml-mlp", "bbml-gnn-varonly"}:
            group = "ENCODER"
        elif abl in {"bbml-gnn-graph-fp32", "bbml-gnn-graph-fp16"}:
            group = "PRECISION"
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


def build_wtl_table(df: pd.DataFrame, iset: str = "all") -> str:
    sub: pd.DataFrame = df[df["instance_set"] == iset]  # type: ignore[assignment]
    if sub.empty or "wtl" not in sub.columns:
        return "% No W/T/L data found.\n"

    preferred = [
        "scip-default",
        "strong-branch",
        "bbml-mlp",
        "bbml-gnn-varonly",
        "bbml-gnn-graph",
        "alpha-fixed-0.5",
        "pure-imitation",
        "solver-only-alpha0",
        "no-temperature",
        "no-cond-gate",
        "no-conf-gate",
    ]
    present = set(sub["solver"].to_numpy().tolist())
    solvers = [solver for solver in preferred if solver in present]
    if not solvers:
        solvers = sorted(present)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Wins / ties / losses versus \texttt{scip-default}. A win is a $>5\%$ reduction in solve time on the matched instance-seed run.}")
    lines.append(r"\label{tab:wtl}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{W/T/L} & \textbf{Win \%} \\")
    lines.append(r"\midrule")
    for solver in solvers:
        solver_sub: pd.DataFrame = sub[sub["solver"] == solver]  # type: ignore[assignment]
        if solver_sub.empty:
            continue
        row = solver_sub.iloc[0]
        label = SOLVER_LABELS.get(solver, ABLATION_LABELS.get(solver, r"\texttt{" + solver + "}"))
        lines.append(f"{label} & {row['wtl']} & {float(row['wtl_win_pct']):.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_accuracy_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "% No accuracy data found.\n"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Held-out branching top-1 accuracy.}")
    lines.append(r"\label{tab:top1acc}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Top-1 Acc.} & \textbf{Branch Nodes} \\")
    lines.append(r"\midrule")
    best = float(df["top1_acc"].max())
    for _, row in df.sort_values("top1_acc", ascending=False).iterrows():
        label = SOLVER_LABELS.get(str(row["model"]), ABLATION_LABELS.get(str(row["model"]), r"\texttt{" + str(row["model"]) + "}"))
        acc = float(row["top1_acc"])
        acc_str = f"{acc:.3f}"
        if abs(acc - best) < 1e-9:
            acc_str = r"\textbf{" + acc_str + "}"
        lines.append(f"{label} & {acc_str} & {int(row['n_groups'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kpis", required=True, type=Path, help="KPI CSV from kpis.py")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for .tex files")
    ap.add_argument("--ablation-set", default="sc_test", help="Instance set to use for the secondary comparison table")
    ap.add_argument("--wtl-set", default="all", help="Instance set to use for the W/T/L table")
    ap.add_argument("--accuracy", type=Path, default=None, help="Optional branching accuracy CSV")
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

    wtl_tex = build_wtl_table(df, iset=args.wtl_set)
    out3 = args.out / "table_wtl.tex"
    out3.write_text(wtl_tex)
    print(f"  Written: {out3}")

    if args.accuracy and args.accuracy.exists():
        acc_df = pd.read_csv(args.accuracy)
        acc_tex = build_accuracy_table(acc_df)
        out4 = args.out / "table_branching_accuracy.tex"
        out4.write_text(acc_tex)
        print(f"  Written: {out4}")

    print("Done.")


if __name__ == "__main__":
    main()
