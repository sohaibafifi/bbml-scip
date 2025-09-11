import pandas as pd
from typing import List, Dict, Any


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_ndjson(path: str) -> pd.DataFrame:
    """Load newline-delimited JSON produced by the C++ telemetry logger."""
    return pd.read_json(path, lines=True)


def group_candidates_by_node(df: pd.DataFrame) -> Dict[Any, pd.DataFrame]:
    """Return a dict node_id -> candidate DataFrame for that node."""
    if "node_id" not in df.columns:
        raise ValueError("expected column 'node_id'")
    return {k: g for k, g in df.groupby("node_id")}


def reconstruct_queue(df: pd.DataFrame) -> List[Any]:
    """Best-effort reconstruction of node processing order.

    Uses (depth, lp_time) as a stable sort key if timestamps are absent.
    """
    keycols = [c for c in ["depth", "lp_time"] if c in df.columns]
    if not keycols:
        # fallback to arbitrary grouping order
        return list(df.groupby("node_id").groups.keys())
    return list(df.sort_values(keycols).groupby("node_id").groups.keys())
