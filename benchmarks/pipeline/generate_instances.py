"""
Standalone benchmark instance generators following Gasse et al. 2019 (NeurIPS).

Families:
  sc   -- Set Covering
  ca   -- Combinatorial Auctions (Leyton-Brown CATS)
  cfl  -- Capacitated Facility Location
  mis  -- Maximum Independent Set

Output: LP files compatible with SCIP.
No ecole or SCIP dependency required -- only numpy and networkx.
"""

import argparse
import pathlib

import networkx as nx  # type: ignore[import-untyped]
import numpy as np


# ---------------------------------------------------------------------------
# LP writer helpers
# ---------------------------------------------------------------------------


def _write_lp(path: pathlib.Path, obj_sense: str, obj: dict[str, float], constrs: list[tuple[str, dict[str, float], str, float]], bounds: dict[str, tuple[float | None, float | None]] | None = None, generals: list[str] | None = None, binary: list[str] | None = None) -> None:
    """Write a minimal LP file.

    constrs: list of (name, {var: coef}, sense, rhs)
    """
    lines = [f"\\Problem name: {path.stem}\n"]
    # Objective
    obj_str = " + ".join(f"{c:.6g} {v}" if c >= 0 else f"{c:.6g} {v}" for v, c in obj.items())
    lines.append(f"{obj_sense}\n  obj: {obj_str}\n")
    # Constraints
    lines.append("subject to\n")
    for name, lhs, sense, rhs in constrs:
        lhs_str = " + ".join(f"{c:.6g} {v}" if c >= 0 else f"- {abs(c):.6g} {v}" for v, c in lhs.items())
        lines.append(f"  {name}: {lhs_str} {sense} {rhs:.6g}\n")
    # Bounds
    if bounds:
        lines.append("bounds\n")
        for v, (lb, ub) in bounds.items():
            lb_str = str(lb) if lb is not None else "-inf"
            ub_str = str(ub) if ub is not None else "+inf"
            lines.append(f"  {lb_str} <= {v} <= {ub_str}\n")
    # Generals (integer non-binary)
    if generals:
        lines.append("generals\n  " + " ".join(generals) + "\n")
    # Binary
    if binary:
        per_line = 10
        lines.append("binary\n")
        for i in range(0, len(binary), per_line):
            lines.append("  " + " ".join(binary[i : i + per_line]) + "\n")
    lines.append("end\n")
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Set Covering
# ---------------------------------------------------------------------------


def gen_sc(rng: np.random.Generator, n_rows: int = 1000, n_cols: int = 500, density: float = 0.05) -> tuple[dict, list, list[str] | None, list[str]]:
    """Generate a set-covering LP (Gasse 2019, Appendix A.1)."""
    # Coverage matrix
    A = rng.random((n_rows, n_cols)) < density
    # Ensure every row is covered by at least one column
    uncovered = np.where(A.sum(axis=1) == 0)[0]
    for r in uncovered:
        A[r, rng.integers(n_cols)] = True
    costs = rng.uniform(1, 100, size=n_cols)
    vars_ = [f"x{j}" for j in range(n_cols)]
    obj = {v: float(c) for v, c in zip(vars_, costs)}
    constrs = []
    for i in range(n_rows):
        lhs = {vars_[j]: 1.0 for j in range(n_cols) if A[i, j]}
        constrs.append((f"c{i}", lhs, ">=", 1.0))
    return obj, constrs, None, vars_


def write_sc(
    path: pathlib.Path,
    seed: int,
    *,
    n_rows: int = 1000,
    n_cols: int = 500,
    density: float = 0.05,
) -> None:
    rng = np.random.default_rng(seed)
    obj, constrs, _, binary = gen_sc(rng, n_rows=n_rows, n_cols=n_cols, density=density)
    _write_lp(path, "minimize", obj, constrs, binary=binary)


# ---------------------------------------------------------------------------
# Combinatorial Auctions  (Leyton-Brown CATS arbitrary distribution)
# ---------------------------------------------------------------------------


def gen_ca(rng: np.random.Generator, n_items: int = 100, n_bids: int = 500, max_bundle: int = 5) -> tuple[dict, list, None, list[str]]:
    """Generate a combinatorial auction MIP (Gasse 2019, Appendix A.2)."""
    # Generate bids: each bid covers a random subset of items
    bids: list[tuple[float, list[int]]] = []
    for _ in range(n_bids):
        k = rng.integers(1, max_bundle + 1)
        items = rng.choice(n_items, size=k, replace=False).tolist()
        # Price proportional to bundle size + super-additive noise
        price = float(rng.uniform(1, 10) * k + rng.exponential(5))
        bids.append((price, items))

    vars_ = [f"x{j}" for j in range(n_bids)]
    obj = {f"x{j}": -bid[0] for j, bid in enumerate(bids)}  # maximise -> negate
    # Each item can be assigned to at most one bid
    constrs = []
    for i in range(n_items):
        lhs = {f"x{j}": 1.0 for j, (_, items) in enumerate(bids) if i in items}
        if lhs:
            constrs.append((f"item{i}", lhs, "<=", 1.0))
    return obj, constrs, None, vars_


def write_ca(
    path: pathlib.Path,
    seed: int,
    *,
    n_items: int = 100,
    n_bids: int = 500,
    max_bundle: int = 5,
) -> None:
    rng = np.random.default_rng(seed)
    obj, constrs, _, binary = gen_ca(rng, n_items=n_items, n_bids=n_bids, max_bundle=max_bundle)
    _write_lp(path, "minimize", obj, constrs, binary=binary)


# ---------------------------------------------------------------------------
# Capacitated Facility Location
# ---------------------------------------------------------------------------


def gen_cfl(rng: np.random.Generator, n_cust: int = 100, n_fac: int = 100) -> tuple[dict, list, dict, list[str]]:
    """Generate a capacitated facility location MIP (Gasse 2019, Appendix A.3)."""
    demand = rng.uniform(5, 35, size=n_cust)
    capacity = rng.uniform(10, 160, size=n_fac)
    fixed_cost = rng.uniform(100, 110, size=n_fac) * np.sqrt(n_fac)
    transport = rng.uniform(0, 90, size=(n_cust, n_fac)) * demand[:, None]

    # Variables: y[j] = open facility j (binary), x[i,j] = fraction from j to i
    y_vars = [f"y{j}" for j in range(n_fac)]
    x_vars = [f"x{i}_{j}" for i in range(n_cust) for j in range(n_fac)]

    obj: dict[str, float] = {}
    for j in range(n_fac):
        obj[y_vars[j]] = float(fixed_cost[j])
    for i in range(n_cust):
        for j in range(n_fac):
            obj[f"x{i}_{j}"] = float(transport[i, j])

    constrs: list[tuple[str, dict[str, float], str, float]] = []
    # Each customer fully served
    for i in range(n_cust):
        lhs = {f"x{i}_{j}": 1.0 for j in range(n_fac)}
        constrs.append((f"dem{i}", lhs, ">=", 1.0))
    # Capacity constraints
    for j in range(n_fac):
        lhs = {f"x{i}_{j}": float(demand[i]) for i in range(n_cust)}
        lhs[y_vars[j]] = -float(capacity[j])
        constrs.append((f"cap{j}", lhs, "<=", 0.0))

    bounds: dict[str, tuple[float | None, float | None]] = {v: (0.0, 1.0) for v in x_vars}
    return obj, constrs, bounds, y_vars


def write_cfl(
    path: pathlib.Path,
    seed: int,
    *,
    n_cust: int = 100,
    n_fac: int = 100,
) -> None:
    rng = np.random.default_rng(seed)
    obj, constrs, bounds, y_vars = gen_cfl(rng, n_cust=n_cust, n_fac=n_fac)
    _write_lp(path, "minimize", obj, constrs, bounds=bounds, binary=y_vars)


# ---------------------------------------------------------------------------
# Maximum Independent Set  (Barabasi-Albert graph)
# ---------------------------------------------------------------------------


def gen_mis(rng: np.random.Generator, n_nodes: int = 500, m: int = 4) -> tuple[dict, list, None, list[str]]:
    """Generate a maximum independent set MIP (Gasse 2019, Appendix A.4).

    Uses Barabasi-Albert random graph with m attachment edges.
    """
    # NetworkX BA graph with a numpy-seeded random state
    seed_int = int(rng.integers(0, 2**31))
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed_int)
    vars_ = [f"x{v}" for v in G.nodes()]
    obj = {v: -1.0 for v in vars_}  # maximise independent set size
    constrs: list[tuple[str, dict[str, float], str, float]] = []
    for idx, (u, v) in enumerate(G.edges()):
        constrs.append((f"e{idx}", {f"x{u}": 1.0, f"x{v}": 1.0}, "<=", 1.0))
    return obj, constrs, None, vars_


def write_mis(
    path: pathlib.Path,
    seed: int,
    *,
    n_nodes: int = 500,
    m: int = 4,
) -> None:
    rng = np.random.default_rng(seed)
    obj, constrs, _, binary = gen_mis(rng, n_nodes=n_nodes, m=m)
    _write_lp(path, "minimize", obj, constrs, binary=binary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

WRITERS = {"sc": write_sc, "ca": write_ca, "cfl": write_cfl, "mis": write_mis}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark instances (Gasse 2019)")
    parser.add_argument("family", choices=list(WRITERS))
    parser.add_argument("out_dir", type=pathlib.Path, help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="First instance index")
    parser.add_argument("--count", type=int, required=True, help="Number of instances to generate")
    parser.add_argument("--seed-offset", type=int, default=0, help="Added to instance index to get the RNG seed")
    parser.add_argument("--sc-rows", type=int, default=1000, help="Set covering row count")
    parser.add_argument("--sc-cols", type=int, default=500, help="Set covering column count")
    parser.add_argument("--sc-density", type=float, default=0.05, help="Set covering density")
    parser.add_argument("--ca-items", type=int, default=100, help="Combinatorial auction item count")
    parser.add_argument("--ca-bids", type=int, default=500, help="Combinatorial auction bid count")
    parser.add_argument("--ca-max-bundle", type=int, default=5, help="Combinatorial auction max bundle size")
    parser.add_argument("--cfl-customers", type=int, default=100, help="Facility location customer count")
    parser.add_argument("--cfl-facilities", type=int, default=100, help="Facility location facility count")
    parser.add_argument("--mis-nodes", type=int, default=500, help="MIS node count")
    parser.add_argument("--mis-affinity", type=int, default=4, help="MIS Barabasi-Albert attachment factor")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = WRITERS[args.family]
    writer_kwargs: dict[str, int | float] = {}
    if args.family == "sc":
        writer_kwargs = {"n_rows": args.sc_rows, "n_cols": args.sc_cols, "density": args.sc_density}
    elif args.family == "ca":
        writer_kwargs = {"n_items": args.ca_items, "n_bids": args.ca_bids, "max_bundle": args.ca_max_bundle}
    elif args.family == "cfl":
        writer_kwargs = {"n_cust": args.cfl_customers, "n_fac": args.cfl_facilities}
    elif args.family == "mis":
        writer_kwargs = {"n_nodes": args.mis_nodes, "m": args.mis_affinity}

    for i in range(args.start, args.start + args.count):
        path = args.out_dir / f"{args.family}_{i:05d}.lp"
        if path.exists():
            continue
        writer(path, seed=i + args.seed_offset, **writer_kwargs)
        if (i - args.start + 1) % 500 == 0 or i == args.start:
            print(f"  {args.family}: {i - args.start + 1}/{args.count}", flush=True)

    print(f"Done: {args.count} {args.family} instances in {args.out_dir}")


if __name__ == "__main__":
    main()
