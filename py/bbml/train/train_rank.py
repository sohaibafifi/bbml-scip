import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from bbml.models.graph_ranker import GraphRanker

# -----------------------------
# Feature → tensor pipeline
# -----------------------------
DEFAULT_FEATS = [
    "obj",
    "reduced_cost",
    "fracval",
    "domain_width",
    "is_binary",
    "is_integer",
    "pseudocost_up",
    "pseudocost_down",
    "pc_obs_up",
    "pc_obs_down",
]

GRAPH_VAR_FEATS = [
    *DEFAULT_FEATS,
    "at_lb",
    "at_ub",
    "col_nnz",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class NodeGroup:
    X: torch.Tensor  # [m, d]
    y_true: Optional[torch.Tensor]  # [m] (ListNet targets) or None
    chosen: Optional[int]  # argmax index (if no y_true)


class NodeDataset(Dataset):
    """Reads a Parquet of candidate-level rows and groups them per node_id.
    Expects at least columns: node_id, var_id, and DEFAULT_FEATS.
    Optional: sb_score_up/sb_score_down (to build listwise targets) or a group-level chosen index.
    """

    def __init__(self, parquet_path: str, feature_cols: Optional[List[str]] = None):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")
        _log(f"[data] loading parquet candidates from {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        self.feature_cols = feature_cols or DEFAULT_FEATS
        missing = [c for c in self.feature_cols + ["node_id"] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # Optional targets present?
        self.has_sb = ("sb_score_up" in self.df.columns) or ("sb_score" in self.df.columns)
        # Precompute group indices
        self.groups: List[NodeGroup] = []
        self._build_groups()
        _log(f"[data] loaded {len(self.df)} candidate rows into {len(self.groups)} node groups " f"(strongbranch_targets={'yes' if self.has_sb else 'no'})")

    def _build_groups(self):
        df = self.df
        # Try to detect group-level chosen index if provided in a separate frame
        chosen_by_node: Dict[Any, int] = {}
        if "chosen_idx" in df.columns:
            # If repeated per row, take first
            chosen_by_node = df.groupby("node_id")["chosen_idx"].first().to_dict()
        # Build groups
        for node_id, g in df.groupby("node_id"):
            X = torch.tensor(g[self.feature_cols].astype(float).values, dtype=torch.float32)
            y_true = None
            chosen = None
            if self.has_sb:
                # Prefer a single score column if available, else combine up/down
                if "sb_score" in g.columns:
                    y_true = torch.tensor(g["sb_score"].astype(float).values, dtype=torch.float32)
                else:
                    # Use max of up/down child bound improvements as proxy target
                    up = g["sb_score_up"].astype(float).values
                    if "sb_score_down" in g.columns:
                        down = g["sb_score_down"].astype(float).values
                        tgt = torch.tensor([max(u, d) for u, d in zip(up, down)], dtype=torch.float32)
                    else:
                        tgt = torch.tensor(up, dtype=torch.float32)
                    y_true = tgt
                # If chosen is missing, derive from y_true
                chosen = int(torch.argmax(y_true).item())
            else:
                # Fall back to a provided chosen index or a heuristic
                if node_id in chosen_by_node:
                    chosen = int(chosen_by_node[node_id])
                elif "chosen" in g.columns:
                    chosen = int(g["chosen"].astype(int).iloc[0])
                elif {"pseudocost_up", "pseudocost_down"}.issubset(g.columns):
                    up = torch.tensor(g["pseudocost_up"].astype(float).values, dtype=torch.float32)
                    down = torch.tensor(g["pseudocost_down"].astype(float).values, dtype=torch.float32)
                    chosen = int(torch.argmax(torch.maximum(up, down)).item())
                else:
                    # Last-resort heuristic: pick argmax reduced_cost magnitude.
                    chosen = int(torch.argmax(torch.abs(X[:, self.feature_cols.index("reduced_cost")])).item())
            self.groups.append(NodeGroup(X=X, y_true=y_true, chosen=chosen))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int) -> NodeGroup:
        return self.groups[idx]


# -----------------------------
# Synthetic dataset (sanity test)
# -----------------------------
class SyntheticNodeDataset(Dataset):
    def __init__(self, n_nodes=512, d=8, min_c=8, max_c=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.groups: List[NodeGroup] = []
        w = torch.randn(d, generator=g)
        for _ in range(n_nodes):
            m = int(torch.randint(min_c, max_c + 1, (1,), generator=g).item())
            X = torch.randn(m, d, generator=g)
            noise = 0.1 * torch.randn(m, generator=g)
            scores = X @ w + noise
            chosen = int(torch.argmax(scores).item())
            self.groups.append(NodeGroup(X=X, y_true=scores, chosen=chosen))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int) -> NodeGroup:
        return self.groups[idx]


# -----------------------------
# Model & losses
# -----------------------------
class ScoreMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        p = float(dropout) if dropout and dropout > 0 else 0.0
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [m, d] → scores[m]
        return self.net(X).squeeze(-1)


def listnet_loss(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
    """ListNet: cross-entropy between softmax(true) and softmax(pred)."""
    p = F.softmax(true_scores, dim=0)
    log_q = F.log_softmax(pred_scores, dim=0)
    return -(p * log_q).sum()


# -----------------------------
# Training loop
# -----------------------------


def collate_groups(batch: List[NodeGroup]) -> List[NodeGroup]:
    # Keep variable-length groups; iterate inside the step
    return batch


@dataclass
class TrainCfg:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 0.0
    device: str = "cpu"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str = "cpu",
):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        optim.zero_grad()
        batch_loss = 0.0
        for grp in batch:
            X = grp.X.to(device)
            s = model(X)
            if grp.y_true is not None:
                y = grp.y_true.to(device)
                loss_g = listnet_loss(s, y)
                target = int(torch.argmax(y).item())
            else:
                # Fallback: multiclass CE on chosen index
                target = grp.chosen
                loss_g = F.cross_entropy(s.unsqueeze(0), torch.tensor([target], device=device))
            batch_loss = batch_loss + loss_g
            pred = int(torch.argmax(s).item())
            correct += int(pred == target)
            n += 1
        batch_loss.backward()
        optim.step()
        total_loss += float(batch_loss.item())
    acc = correct / max(1, n)
    return total_loss / max(1, len(loader)), acc


# -----------------------------
# Synthetic bipartite dataset for GNN
# -----------------------------


@dataclass
class GraphNodeGroup:
    var_feat: torch.Tensor  # [n_var, d_var]
    con_feat: torch.Tensor  # [n_con, d_con]
    edge_index: torch.Tensor  # [2, E]
    y_true: Optional[torch.Tensor]
    chosen: Optional[int]


class GraphSyntheticNodeDataset(Dataset):
    def __init__(
        self,
        n_nodes=256,
        d_var=8,
        d_con=8,
        min_v=8,
        max_v=24,
        min_c=4,
        max_c=12,
        seed=0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.groups: List[GraphNodeGroup] = []
        wv = torch.randn(d_var, generator=g)
        for _ in range(n_nodes):
            nv = int(torch.randint(min_v, max_v + 1, (1,), generator=g).item())
            nc = int(torch.randint(min_c, max_c + 1, (1,), generator=g).item())
            v = torch.randn(nv, d_var, generator=g)
            c = torch.randn(nc, d_con, generator=g)
            E = int(max(nv, 1) * 2)
            rows = torch.randint(0, nc, (E,), generator=g)
            cols = torch.randint(0, nv, (E,), generator=g)
            edge_index = torch.stack([rows, cols], dim=0)
            scores = v @ wv
            chosen = int(torch.argmax(scores).item())
            self.groups.append(
                GraphNodeGroup(
                    var_feat=v,
                    con_feat=c,
                    edge_index=edge_index,
                    y_true=scores,
                    chosen=chosen,
                )
            )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int) -> GraphNodeGroup:
        return self.groups[idx]


def collate_graph_groups(batch: List[GraphNodeGroup]) -> List[GraphNodeGroup]:
    return batch


def train_epoch_gnn(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str = "cpu",
):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        optim.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for grp in batch:
            # Support both GraphNodeGroup (with var_feat/con_feat/edge_index)
            # and the tabular NodeGroup (with X as var features).
            v = getattr(grp, "var_feat", None)
            if v is None:
                # NodeGroup: X holds variable features
                v = grp.X
            v = v.to(device)

            c = getattr(grp, "con_feat", None)
            if c is not None:
                c = c.to(device)

            ei = getattr(grp, "edge_index", None)
            if ei is not None:
                ei = ei.to(device)
            else:
                ei = None
            s = model(v, c, ei)
            if grp.y_true is not None:
                y = grp.y_true.to(device)
                loss_g = listnet_loss(s, y)
                target = int(torch.argmax(y).item())
            else:
                target = grp.chosen
                loss_g = F.cross_entropy(s.unsqueeze(0), torch.tensor([target], device=device))
            batch_loss = batch_loss + loss_g
            pred = int(torch.argmax(s).item())
            correct += int(pred == target)
            n += 1
        batch_loss.backward()
        optim.step()
        total_loss += float(batch_loss.item())
    acc = correct / max(1, n)
    return total_loss / max(1, len(loader)), acc


# -----------------------------
# Graph NDJSON dataset (from C++ logger)
# -----------------------------


class GraphJsonNodeDataset(Dataset):
    def __init__(self, ndjson_path: Optional[str] = None, manifest_path: Optional[str] = None):
        t0 = time.time()
        self.paths = self._resolve_paths(ndjson_path, manifest_path)
        _log(f"[data] indexing graph telemetry from {len(self.paths)} file(s)")
        # Build byte offsets for each line to support random access without loading all
        self._offsets: List[tuple[str, int]] = []
        for idx, path in enumerate(self.paths, start=1):
            with open(path, "rb") as f:
                off = 0
                for line in f:
                    ln = len(line)
                    if ln > 1:
                        self._offsets.append((path, off))
                    off += ln
            if idx == len(self.paths) or idx % 50 == 0:
                _log(f"[data] indexed {idx}/{len(self.paths)} files ({len(self._offsets)} graph nodes)")
        # Peek first line to infer dims
        self.d_var = 0
        self.d_con = 0
        if len(self._offsets) > 0:
            g0 = self._read_item(0)
            self.d_var = int(g0.var_feat.size(1))
            self.d_con = int(g0.con_feat.size(1)) if g0.con_feat is not None else 0
        _log(f"[data] graph dataset ready: {len(self._offsets)} node groups, " f"d_var={self.d_var}, d_con={self.d_con}, took {time.time() - t0:.1f}s")

    @staticmethod
    def _resolve_paths(ndjson_path: Optional[str], manifest_path: Optional[str]) -> List[str]:
        paths: List[str] = []
        if ndjson_path:
            if not os.path.exists(ndjson_path):
                raise FileNotFoundError(ndjson_path)
            paths.append(ndjson_path)
        if manifest_path:
            manifest = Path(manifest_path)
            if not manifest.exists():
                raise FileNotFoundError(manifest_path)
            for line in manifest.read_text().splitlines():
                path = line.strip()
                if not path:
                    continue
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                paths.append(path)
        if not paths:
            raise ValueError("expected --graph_ndjson or --graph_manifest")
        return paths

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_item(self, idx: int) -> "GraphNodeGroup":
        path, off = self._offsets[idx]
        with open(path, "rb") as f:
            f.seek(off)
            line = f.readline()
        obj = json.loads(line)
        var_feat = torch.tensor(obj["var_feat"], dtype=torch.float32)
        con_feat = torch.tensor(obj["con_feat"], dtype=torch.float32) if "con_feat" in obj else None
        ei = obj.get("edge_index", None)
        if ei is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(ei, dtype=torch.long)
        # Targets: prefer SB if present; else chosen only
        y_true = None
        if "sb_score_up" in obj or "sb_score_down" in obj:
            up = obj.get("sb_score_up", [0.0] * var_feat.size(0))
            down = obj.get("sb_score_down", None)
            if down is None:
                y = torch.tensor(up, dtype=torch.float32)
            else:
                y = torch.tensor([max(u, d) for u, d in zip(up, down)], dtype=torch.float32)
            y_true = y
        chosen = int(obj.get("chosen_idx", 0))
        return GraphNodeGroup(var_feat=var_feat, con_feat=con_feat, edge_index=edge_index, y_true=y_true, chosen=chosen)

    def __getitem__(self, idx: int) -> "GraphNodeGroup":
        return self._read_item(idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Path to training parquet; if omitted, use synthetic",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "gnn"])
    parser.add_argument("--graph_ndjson", type=str, default=None, help="Path to graph NDJSON logged by C++ (var_feat, con_feat, edge_index)")
    parser.add_argument("--graph_manifest", type=str, default=None, help="Manifest of graph NDJSON files, one path per line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--synthetic_nodes", type=int, default=512)
    parser.add_argument("--d", type=int, default=len(DEFAULT_FEATS))
    parser.add_argument("--min_c", type=int, default=8)
    parser.add_argument("--max_c", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="score_mlp.pt")
    parser.add_argument(
        "--ckpt_best",
        type=str,
        default=None,
        help="Optional path to save the best model. If not set, uses --ckpt",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        choices=["loss", "acc"],
        help="Metric to select best model (min loss or max acc)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    _log(f"[train] seed={args.seed}")

    # Track model config for checkpoint metadata
    model_cfg: Dict[str, Any]

    if args.model == "mlp":
        if args.parquet is None:
            ds = SyntheticNodeDataset(
                n_nodes=args.synthetic_nodes,
                d=args.d,
                min_c=args.min_c,
                max_c=args.max_c,
                seed=args.seed,
            )
            d_in = args.d
            _log(f"[data] using synthetic MLP dataset with {len(ds)} node groups and d_in={d_in}")
        else:
            ds = NodeDataset(args.parquet, feature_cols=DEFAULT_FEATS)
            d_in = len(DEFAULT_FEATS)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_groups)
        model = ScoreMLP(d_in=d_in, hidden=args.hidden, dropout=args.dropout)
        model_cfg = {
            "model": "mlp",
            "d_in": d_in,
            "hidden": args.hidden,
            "dropout": args.dropout,
        }
    else:
        if args.graph_ndjson is not None or args.graph_manifest is not None:
            # Load graph snapshots from NDJSON (constraints ↔ candidate vars)
            ds_g = GraphJsonNodeDataset(args.graph_ndjson, args.graph_manifest)
            loader = DataLoader(
                ds_g,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_graph_groups,
            )
            d_var, d_con = ds_g.d_var, ds_g.d_con
            layers_used, graph_inputs = 3, True
            _log(f"[data] using graph telemetry dataset with {len(ds_g)} node groups " f"(batch_size={args.batch_size})")
            model = GraphRanker(
                d_var=d_var,
                d_con=max(1, d_con),
                hidden=args.hidden,
                layers=layers_used,
                dropout=args.dropout,
            )
        elif args.parquet is None:
            ds_g = GraphSyntheticNodeDataset(n_nodes=args.synthetic_nodes, d_var=args.d, d_con=args.d, seed=args.seed)
            loader = DataLoader(
                ds_g,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_graph_groups,
            )
            d_var, d_con, layers_used, graph_inputs = args.d, args.d, 3, True
            _log(f"[data] using synthetic GNN dataset with {len(ds_g)} node groups")
            model = GraphRanker(
                d_var=d_var,
                d_con=d_con,
                hidden=args.hidden,
                layers=layers_used,
                dropout=args.dropout,
            )
        else:
            # Fallback: load tabular dataset and use var-only path (no edges)
            ds = NodeDataset(args.parquet, feature_cols=DEFAULT_FEATS)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_groups)
            d_var, d_con, layers_used, graph_inputs = (
                len(DEFAULT_FEATS),
                len(DEFAULT_FEATS),
                1,
                False,
            )
            _log(f"[data] using var-only GNN dataset with {len(ds)} node groups " f"(batch_size={args.batch_size})")
            model = GraphRanker(
                d_var=d_var,
                d_con=d_con,
                hidden=args.hidden,
                layers=layers_used,
                dropout=args.dropout,
            )
        model_cfg = {
            "model": "gnn",
            "d_var": d_var,
            "d_con": d_con,
            "hidden": args.hidden,
            "layers": layers_used,
            "dropout": args.dropout,
            "graph_inputs": graph_inputs,
        }
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _log(f"[train] model={model_cfg['model']} device={device} hidden={args.hidden} dropout={args.dropout}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Best-model tracking
    select_higher = args.metric == "acc"
    best_value = -float("inf") if select_higher else float("inf")
    best_epoch = 0
    ckpt_best_path = args.ckpt_best or args.ckpt
    _log(f"[train] starting optimization for {args.epochs} epoch(s)")

    for epoch in range(1, args.epochs + 1):
        if args.model == "mlp":
            loss, acc = train_epoch(model, loader, optim, device=device)
        else:
            loss, acc = train_epoch_gnn(model, loader, optim, device=device)
        _log(f"[epoch {epoch:03d}] loss={loss:.4f} acc@1={acc:.3f}")
        # Check for improvement and save best
        val = acc if select_higher else loss
        improved = val > best_value if select_higher else val < best_value
        if improved and ckpt_best_path:
            best_value = val
            best_epoch = epoch
            torch.save(
                {"model": model_cfg["model"], "cfg": model_cfg, "state_dict": model.state_dict()},
                ckpt_best_path,
            )
            _log(f"Saved new best model to {ckpt_best_path} (epoch {epoch}, {args.metric}={val:.4f})")
    # Note: no final 'last' checkpoint; we keep the best per user's request
    if best_epoch == 0 and ckpt_best_path:
        # Edge case: zero epochs or no improvement criterion triggered
        torch.save(
            {"model": model_cfg["model"], "cfg": model_cfg, "state_dict": model.state_dict()},
            ckpt_best_path,
        )
        _log(f"Saved model checkpoint to {ckpt_best_path}")
    # quick sanity: on synthetic, expect >0.8 acc@1 within a few epochs


if __name__ == "__main__":
    main()
