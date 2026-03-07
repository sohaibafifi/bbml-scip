import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbml.models.graph_ranker import GraphRanker
from bbml.train.train_rank import (
    DEFAULT_FEATS,
    GraphJsonNodeDataset,
    NodeDataset,
    ScoreMLP,
    collate_graph_groups,
    collate_groups,
)


@dataclass
class TempScaleResult:
    temperature: float
    n_groups: int
    n_items: int


def listnet_nll(scores: torch.Tensor, targets: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    scaled = scores / temperature.clamp_min(1e-3)
    p = F.softmax(targets, dim=0)
    log_q = F.log_softmax(scaled, dim=0)
    return -(p * log_q).sum()


def _build_model(cfg: dict) -> nn.Module:
    model_type = cfg.get("model")
    if model_type == "mlp":
        return ScoreMLP(
            d_in=int(cfg["d_in"]),
            hidden=int(cfg.get("hidden", 64)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if model_type == "gnn":
        return GraphRanker(
            d_var=int(cfg["d_var"]),
            d_con=int(cfg["d_con"]),
            hidden=int(cfg.get("hidden", 64)),
            layers=int(cfg.get("layers", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported checkpoint model type: {model_type!r}")


def _load_model(ckpt_path: str, device: str) -> Tuple[nn.Module, dict]:
    obj = torch.load(ckpt_path, map_location=device)
    if not isinstance(obj, dict) or "state_dict" not in obj or "cfg" not in obj:
        raise ValueError(f"Checkpoint {ckpt_path} does not contain cfg/state_dict metadata")
    cfg = dict(obj["cfg"])
    model = _build_model(cfg)
    model.load_state_dict(obj["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def _build_loader(cfg: dict, args: argparse.Namespace) -> DataLoader:
    if cfg["model"] == "mlp":
        dataset = NodeDataset(args.parquet, feature_cols=DEFAULT_FEATS)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_groups)
    if cfg.get("graph_inputs", False):
        dataset = GraphJsonNodeDataset(ndjson_path=args.graph_ndjson, manifest_path=args.graph_manifest)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graph_groups)
    dataset = NodeDataset(args.parquet, feature_cols=DEFAULT_FEATS)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_groups)


def _score_group(model: nn.Module, group, cfg: dict, device: str) -> torch.Tensor:
    if cfg["model"] == "mlp":
        return model(group.X.to(device))
    if cfg.get("graph_inputs", False):
        con_feat = group.con_feat.to(device) if group.con_feat is not None else None
        edge_index = group.edge_index.to(device) if group.edge_index is not None else None
        return model(group.var_feat.to(device), con_feat, edge_index)
    return model(group.X.to(device), None, None)


def fit_temperature_listwise(model: nn.Module, loader: DataLoader, cfg: dict, device: str = "cpu") -> TempScaleResult:
    model.eval()
    temperature_param = nn.Parameter(torch.tensor([0.0], device=device))
    optimizer = torch.optim.LBFGS([temperature_param], lr=0.1, max_iter=50)

    def closure():  # type: ignore
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros([], device=device)
        for batch in loader:
            for group in batch:
                if group.y_true is None:
                    continue
                scores = _score_group(model, group, cfg, device)
                total_loss = total_loss + listnet_nll(scores, group.y_true.to(device), temperature_param.exp())
        total_loss.backward()
        return total_loss

    optimizer.step(closure)

    n_groups = 0
    n_items = 0
    for batch in loader:
        for group in batch:
            if group.y_true is None:
                continue
            n_groups += 1
            if hasattr(group, "X"):
                n_items += int(group.X.size(0))
            else:
                n_items += int(group.var_feat.size(0))

    return TempScaleResult(temperature=float(temperature_param.exp().item()), n_groups=n_groups, n_items=n_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint with saved cfg/state_dict metadata")
    ap.add_argument("--parquet", type=str, required=True, help="Validation parquet for MLP and var-only models")
    ap.add_argument("--graph_ndjson", type=str, default=None, help="Single graph NDJSON validation file")
    ap.add_argument("--graph_manifest", type=str, default=None, help="Manifest of graph NDJSON validation files")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None, help="Optional file to write the fitted temperature into")
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(args.ckpt)
    if not Path(args.parquet).exists():
        raise FileNotFoundError(args.parquet)

    model, cfg = _load_model(args.ckpt, args.device)
    if cfg.get("graph_inputs", False) and not (args.graph_ndjson or args.graph_manifest):
        raise ValueError("graph-input checkpoints require --graph_ndjson or --graph_manifest for calibration")

    loader = _build_loader(cfg, args)
    res = fit_temperature_listwise(model, loader, cfg, device=args.device)
    if args.out:
        Path(args.out).write_text(f"{res.temperature:.6f}\n")
    print(f"Fitted temperature: T={res.temperature:.6f} (groups={res.n_groups}, items={res.n_items})")


if __name__ == "__main__":
    main()
