import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
    _auto_num_workers,
    _auto_pin_memory,
    _build_loader_kwargs,
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


def _parse_ckpt_paths(ckpt_spec: str) -> List[str]:
    return [part.strip() for part in ckpt_spec.split(",") if part.strip()]


def _cfg_signature(cfg: dict) -> Tuple:
    return (
        cfg.get("model"),
        int(cfg.get("d_in", -1)),
        int(cfg.get("d_var", -1)),
        int(cfg.get("d_con", -1)),
        int(cfg.get("hidden", 64)),
        int(cfg.get("layers", 1)),
        float(cfg.get("dropout", 0.0)),
        bool(cfg.get("graph_inputs", False)),
    )


def _load_models(ckpt_spec: str, device: str) -> Tuple[List[nn.Module], dict]:
    models: List[nn.Module] = []
    cfg_ref = None
    sig_ref = None
    for ckpt_path in _parse_ckpt_paths(ckpt_spec):
        obj = torch.load(ckpt_path, map_location=device)
        if not isinstance(obj, dict) or "state_dict" not in obj or "cfg" not in obj:
            raise ValueError(f"Checkpoint {ckpt_path} does not contain cfg/state_dict metadata")
        cfg = dict(obj["cfg"])
        sig = _cfg_signature(cfg)
        if cfg_ref is None:
            cfg_ref = cfg
            sig_ref = sig
        elif sig != sig_ref:
            raise ValueError("All ensemble checkpoints must share the same model configuration")
        model = _build_model(cfg)
        model.load_state_dict(obj["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    if not models or cfg_ref is None:
        raise ValueError(f"No checkpoints resolved from: {ckpt_spec}")
    return models, cfg_ref


def _build_loader(cfg: dict, args: argparse.Namespace) -> DataLoader:
    graph_inputs = bool(cfg.get("graph_inputs", False))
    if args.num_workers < 0:
        args.num_workers = _auto_num_workers(device=str(args.device), graph_inputs=graph_inputs)
    if args.pin_memory < 0:
        args.pin_memory = int(_auto_pin_memory(device=str(args.device), num_workers=args.num_workers, graph_inputs=graph_inputs))
    pin_memory = bool(args.pin_memory)
    if cfg["model"] == "mlp":
        dataset = NodeDataset(args.parquet, feature_cols=DEFAULT_FEATS)
        return DataLoader(
            dataset,
            **_build_loader_kwargs(
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_groups,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            ),
        )
    if cfg.get("graph_inputs", False):
        dataset = GraphJsonNodeDataset(ndjson_path=args.graph_ndjson, manifest_path=args.graph_manifest)
        return DataLoader(
            dataset,
            **_build_loader_kwargs(
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_graph_groups,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            ),
        )
    dataset = NodeDataset(args.parquet, feature_cols=DEFAULT_FEATS)
    return DataLoader(
        dataset,
        **_build_loader_kwargs(
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_groups,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
    )


def _score_group(models: List[nn.Module], group, cfg: dict, device: str) -> torch.Tensor:
    scores = []
    for model in models:
        if cfg["model"] == "mlp":
            scores.append(model(group.X.to(device)))
            continue
        if cfg.get("graph_inputs", False):
            con_feat = group.con_feat.to(device) if group.con_feat is not None else None
            edge_index = group.edge_index.to(device) if group.edge_index is not None else None
            scores.append(model(group.var_feat.to(device), con_feat, edge_index))
            continue
        scores.append(model(group.X.to(device), None, None))
    if len(scores) == 1:
        return scores[0]
    return torch.stack(scores, dim=0).mean(dim=0)


def fit_temperature_listwise(models: List[nn.Module], loader: DataLoader, cfg: dict, device: str = "cpu") -> TempScaleResult:
    for model in models:
        model.eval()
    temperature_param = nn.Parameter(torch.tensor([0.0], device=device))
    optimizer = torch.optim.LBFGS([temperature_param], lr=0.1, max_iter=50)

    def closure():  # type: ignore
        with torch.enable_grad():
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros([], device=device)
            used_groups = 0
            for batch in loader:
                for group in batch:
                    if group.y_true is None:
                        continue
                    scores = _score_group(models, group, cfg, device)
                    total_loss = total_loss + listnet_nll(scores, group.y_true.to(device), temperature_param.exp())
                    used_groups += 1
            if used_groups == 0:
                # Keep the optimization graph valid while leaving T at 1.0.
                total_loss = temperature_param.sum() * 0.0
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
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers. -1 selects an automatic default.")
    ap.add_argument("--pin_memory", type=int, default=-1, help="Pin DataLoader memory. -1 selects automatically.")
    ap.add_argument("--out", type=str, default=None, help="Optional file to write the fitted temperature into")
    args = ap.parse_args()

    ckpt_paths = _parse_ckpt_paths(args.ckpt)
    if not ckpt_paths:
        raise FileNotFoundError(args.ckpt)
    for ckpt in ckpt_paths:
        if not Path(ckpt).exists():
            raise FileNotFoundError(ckpt)
    if not Path(args.parquet).exists():
        raise FileNotFoundError(args.parquet)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    models, cfg = _load_models(args.ckpt, args.device)
    if cfg.get("graph_inputs", False) and not (args.graph_ndjson or args.graph_manifest):
        raise ValueError("graph-input checkpoints require --graph_ndjson or --graph_manifest for calibration")

    loader = _build_loader(cfg, args)
    res = fit_temperature_listwise(models, loader, cfg, device=args.device)
    if args.out:
        Path(args.out).write_text(f"{res.temperature:.6f}\n")
    print(f"Fitted temperature: T={res.temperature:.6f} (groups={res.n_groups}, items={res.n_items})")


if __name__ == "__main__":
    main()
