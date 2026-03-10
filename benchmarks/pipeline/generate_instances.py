"""
Standalone benchmark instance generators following Gasse et al. 2019 (NeurIPS).

Families:
  sc   -- Set Covering
  ca   -- Combinatorial Auctions (Leyton-Brown CATS)
  cfl  -- Capacitated Facility Location
  mis  -- Maximum Independent Set

Output: LP files compatible with SCIP.
No ecole or SCIP dependency required -- only numpy and scipy.
"""

import argparse
import pathlib
from itertools import combinations

import numpy as np
import scipy.sparse


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


class Graph:
    """Minimal Barabasi-Albert graph container ported from learn2branch."""

    def __init__(
        self,
        number_of_nodes: int,
        edges: set[tuple[int, int]],
        degrees: np.ndarray,
        neighbors: dict[int, set[int]],
    ) -> None:
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self) -> int:
        return self.number_of_nodes

    def greedy_clique_partition(self) -> list[set[int]]:
        cliques: list[set[int]] = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {int(clique_center)}
            neighbors = self.neighbors[int(clique_center)].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda node: -self.degrees[node])
            for neighbor in densest_neighbors:
                if all(neighbor in self.neighbors[clique_node] for clique_node in clique):
                    clique.add(int(neighbor))
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def barabasi_albert(number_of_nodes: int, affinity: int, rng: np.random.Generator) -> "Graph":
        assert affinity >= 1 and affinity < number_of_nodes

        edges: set[tuple[int, int]] = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}

        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = rng.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edge = (int(node), int(new_node))
                edges.add(edge)
                degrees[int(node)] += 1
                degrees[int(new_node)] += 1
                neighbors[int(node)].add(int(new_node))
                neighbors[int(new_node)].add(int(node))

        return Graph(number_of_nodes, edges, degrees, neighbors)


# ---------------------------------------------------------------------------
# Set Covering
# ---------------------------------------------------------------------------


def gen_sc(rng: np.random.Generator, n_rows: int = 1000, n_cols: int = 500, density: float = 0.05) -> tuple[dict, list, list[str] | None, list[str]]:
    """Generate a set-cover instance using the learn2branch construction."""
    max_coef = 100
    nnzrs = int(n_rows * n_cols * density)

    assert nnzrs >= n_rows
    assert nnzrs >= 2 * n_cols

    indices = rng.choice(n_cols, size=nnzrs)
    indices[: 2 * n_cols] = np.repeat(np.arange(n_cols), 2)
    _, col_nrows = np.unique(indices, return_counts=True)

    indices[:n_rows] = rng.permutation(n_rows)
    i = 0
    indptr = [0]
    for n in col_nrows:
        if i >= n_rows:
            indices[i : i + n] = rng.choice(n_rows, size=n, replace=False)
        elif i + n > n_rows:
            remaining_rows = np.setdiff1d(np.arange(n_rows), indices[i:n_rows], assume_unique=True)
            indices[n_rows : i + n] = rng.choice(remaining_rows, size=i + n - n_rows, replace=False)
        i += n
        indptr.append(i)

    coefs = rng.integers(max_coef, size=n_cols) + 1
    A = scipy.sparse.csc_matrix(
        (np.ones(len(indices), dtype=int), indices, indptr),
        shape=(n_rows, n_cols),
    ).tocsr()

    vars_ = [f"x{j}" for j in range(n_cols)]
    obj = {v: float(c) for v, c in zip(vars_, coefs)}
    constrs = []
    for row_idx in range(n_rows):
        row_cols = A.indices[A.indptr[row_idx] : A.indptr[row_idx + 1]]
        lhs = {vars_[int(col_idx)]: 1.0 for col_idx in row_cols}
        constrs.append((f"c{row_idx}", lhs, ">=", 1.0))
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


def gen_ca(
    rng: np.random.Generator,
    n_items: int = 100,
    n_bids: int = 500,
    max_bundle: int = 5,
) -> tuple[dict, list, None, list[str]]:
    """Generate a combinatorial auction MIP using learn2branch's arbitrary scheme.

    This ports the generator from ds4dm/learn2branch so that the CA staged protocol
    matches the original instance distribution instead of the previous random-bundle
    placeholder. The ``max_bundle`` argument is kept only for CLI compatibility with
    older repo scripts and is ignored by the arbitrary-scheme generator.
    """

    del max_bundle

    def choose_next_item(
        bundle_mask: np.ndarray,
        interests: np.ndarray,
        compats: np.ndarray,
    ) -> int:
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return int(rng.choice(n_items, p=prob))

    min_value = 1.0
    max_value = 100.0
    value_deviation = 0.5
    add_item_prob = 0.7
    max_n_sub_bids = 5
    additivity = 0.2
    budget_factor = 1.5
    resale_factor = 0.5

    values = min_value + (max_value - min_value) * rng.random(n_items)

    compats = np.triu(rng.random((n_items, n_items)), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(axis=1, keepdims=True)

    bids: list[tuple[list[int], float]] = []
    n_dummy_items = 0

    while len(bids) < n_bids:
        private_interests = rng.random(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        bidder_bids: dict[frozenset[int], float] = {}

        prob = private_interests / private_interests.sum()
        item = int(rng.choice(n_items, p=prob))
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        while rng.random() < add_item_prob:
            if int(bundle_mask.sum()) == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]
        price = float(private_values[bundle].sum() + np.power(len(bundle), 1 + additivity))
        if price < 0:
            continue
        bidder_bids[frozenset(int(i) for i in bundle)] = price

        sub_candidates: list[tuple[np.ndarray, float]] = []
        for item in bundle:
            sub_bundle_mask = np.full(n_items, 0)
            sub_bundle_mask[item] = 1

            while int(sub_bundle_mask.sum()) < len(bundle):
                next_item = choose_next_item(sub_bundle_mask, private_interests, compats)
                sub_bundle_mask[next_item] = 1

            sub_bundle = np.nonzero(sub_bundle_mask)[0]
            sub_price = float(private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity))
            sub_candidates.append((sub_bundle, sub_price))

        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        ranked_candidates = [sub_candidates[i] for i in np.argsort([-candidate_price for _, candidate_price in sub_candidates])]
        for sub_bundle, sub_price in ranked_candidates:
            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break
            if sub_price < 0:
                continue
            if sub_price > budget:
                continue
            if values[sub_bundle].sum() < min_resale_value:
                continue
            bundle_key = frozenset(int(i) for i in sub_bundle)
            if bundle_key in bidder_bids:
                continue
            bidder_bids[bundle_key] = sub_price

        dummy_item = [n_items + n_dummy_items] if len(bidder_bids) > 2 else []
        if dummy_item:
            n_dummy_items += 1

        for bidder_bundle, bidder_price in bidder_bids.items():
            bids.append((list(bidder_bundle) + dummy_item, bidder_price))

    vars_ = [f"x{j}" for j in range(len(bids))]
    obj = {var: -float(price) for var, (_, price) in zip(vars_, bids)}

    bids_per_item: list[list[int]] = [[] for _ in range(n_items + n_dummy_items)]
    for bid_idx, (bundle_items, _) in enumerate(bids):
        for item in bundle_items:
            bids_per_item[item].append(bid_idx)

    constrs = []
    for item_idx, item_bids in enumerate(bids_per_item):
        if not item_bids:
            continue
        lhs = {vars_[bid_idx]: 1.0 for bid_idx in item_bids}
        constrs.append((f"item{item_idx}", lhs, "<=", 1.0))

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


def gen_cfl(
    rng: np.random.Generator,
    n_cust: int = 100,
    n_fac: int = 100,
    ratio: float = 5.0,
) -> tuple[dict, list, dict, list[str]]:
    """Generate a CFL instance using the learn2branch geometric formulation."""
    c_x = rng.random(n_cust)
    c_y = rng.random(n_cust)
    f_x = rng.random(n_fac)
    f_y = rng.random(n_fac)

    demands = rng.integers(5, 36, size=n_cust)
    capacities = rng.integers(10, 161, size=n_fac)
    fixed_costs = (rng.integers(100, 111, size=n_fac) * np.sqrt(capacities) + rng.integers(91, size=n_fac)).astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()
    capacities = (capacities * ratio * total_demand / total_capacity).astype(int)
    total_capacity = capacities.sum()

    trans_costs = np.sqrt((c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    y_vars = [f"y_{j + 1}" for j in range(n_fac)]
    x_vars = [f"x_{i + 1}_{j + 1}" for i in range(n_cust) for j in range(n_fac)]

    obj: dict[str, float] = {}
    for i in range(n_cust):
        for j in range(n_fac):
            obj[f"x_{i + 1}_{j + 1}"] = float(trans_costs[i, j])
    for j in range(n_fac):
        obj[y_vars[j]] = float(fixed_costs[j])

    constrs: list[tuple[str, dict[str, float], str, float]] = []
    for i in range(n_cust):
        lhs = {f"x_{i + 1}_{j + 1}": -1.0 for j in range(n_fac)}
        constrs.append((f"demand_{i + 1}", lhs, "<=", -1.0))
    for j in range(n_fac):
        lhs = {f"x_{i + 1}_{j + 1}": float(demands[i]) for i in range(n_cust)}
        lhs[y_vars[j]] = -float(capacities[j])
        constrs.append((f"capacity_{j + 1}", lhs, "<=", 0.0))

    lhs = {y_vars[j]: -float(capacities[j]) for j in range(n_fac)}
    constrs.append(("total_capacity", lhs, "<=", -float(total_demand)))
    for i in range(n_cust):
        for j in range(n_fac):
            constrs.append(
                (
                    f"affectation_{i + 1}_{j + 1}",
                    {f"x_{i + 1}_{j + 1}": 1.0, y_vars[j]: -1.0},
                    "<=",
                    0.0,
                )
            )

    bounds: dict[str, tuple[float | None, float | None]] = {v: (0.0, 1.0) for v in x_vars}
    return obj, constrs, bounds, y_vars


def write_cfl(
    path: pathlib.Path,
    seed: int,
    *,
    n_cust: int = 100,
    n_fac: int = 100,
    ratio: float = 5.0,
) -> None:
    rng = np.random.default_rng(seed)
    obj, constrs, bounds, y_vars = gen_cfl(rng, n_cust=n_cust, n_fac=n_fac, ratio=ratio)
    _write_lp(path, "minimize", obj, constrs, bounds=bounds, binary=y_vars)


# ---------------------------------------------------------------------------
# Maximum Independent Set  (Barabasi-Albert graph)
# ---------------------------------------------------------------------------


def gen_mis(rng: np.random.Generator, n_nodes: int = 500, m: int = 4) -> tuple[dict, list, None, list[str]]:
    """Generate MIS using the learn2branch BA graph and clique formulation."""
    graph = Graph.barabasi_albert(n_nodes, m, rng)
    cliques = graph.greedy_clique_partition()
    inequalities: set[tuple[int, ...]] = set(graph.edges)
    for clique in cliques:
        clique_tuple = tuple(sorted(clique))
        for edge in combinations(clique_tuple, 2):
            inequalities.discard(edge)
        if len(clique_tuple) > 1:
            inequalities.add(clique_tuple)

    used_nodes: set[int] = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(min(10, n_nodes)):
        if node not in used_nodes:
            inequalities.add((node,))

    vars_ = [f"x{node + 1}" for node in range(n_nodes)]
    obj = {var: -1.0 for var in vars_}
    constrs: list[tuple[str, dict[str, float], str, float]] = []
    for idx, group in enumerate(sorted(inequalities)):
        lhs = {f"x{node + 1}": 1.0 for node in group}
        constrs.append((f"c{idx + 1}", lhs, "<=", 1.0))
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
    parser.add_argument(
        "--ca-max-bundle",
        type=int,
        default=5,
        help="Deprecated compatibility knob; ignored by the learn2branch CA generator",
    )
    parser.add_argument("--cfl-customers", type=int, default=100, help="Facility location customer count")
    parser.add_argument("--cfl-facilities", type=int, default=100, help="Facility location facility count")
    parser.add_argument("--cfl-ratio", type=float, default=5.0, help="Facility location capacity/demand ratio")
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
        writer_kwargs = {"n_cust": args.cfl_customers, "n_fac": args.cfl_facilities, "ratio": args.cfl_ratio}
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
