import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbml.train.train_rank import (
    NodeDataset,
    DEFAULT_FEATS,
    collate_groups,
    ScoreMLP,
)
import argparse
import os


@dataclass
class TempScaleResult:
    temperature: float
    n_groups: int
    n_items: int


def listnet_nll(scores: torch.Tensor, targets: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    s = scores / T.clamp_min(1e-3)
    p = F.softmax(targets, dim=0)
    log_q = F.log_softmax(s, dim=0)
    return -(p * log_q).sum()


def fit_temperature_listwise(model: nn.Module, loader: DataLoader, device: str = "cpu") -> TempScaleResult:
    """Optimize temperature T > 0 to minimize sum ListNet NLL on a loader of NodeGroup.

    Expects each batch to be a list of NodeGroup with fields X and y_true.
    """
    model.eval()
    T_param = nn.Parameter(torch.tensor([1.0], device=device))
    opt = torch.optim.LBFGS([T_param], lr=0.1, max_iter=50)

    n_groups = 0
    n_items = 0

    def closure():  # type: ignore
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros([], device=device)
        for batch in loader:
            for grp in batch:
                if grp.y_true is None:
                    continue
                X = grp.X.to(device)
                y = grp.y_true.to(device)
                s = model(X)
                loss = loss + listnet_nll(s, y, T_param.exp())
        loss.backward()
        return loss

    opt.step(closure)

    # Compute summary stats
    for batch in loader:
        for grp in batch:
            if grp.y_true is None:
                continue
            n_groups += 1
            n_items += int(grp.X.size(0))

    return TempScaleResult(temperature=float(T_param.exp().item()), n_groups=n_groups, n_items=n_items)


def mc_dropout_scores(model: nn.Module, X: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MC-dropout by enabling dropout at eval time.

    Returns (mean_scores[m], std_scores[m]).
    """
    was_training = model.training
    model.eval()

    # Enable dropout modules during eval
    def _enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    scores = []
    with torch.no_grad():
        for _ in range(max(1, n_samples)):
            scores.append(model(X))
    S = torch.stack(scores, dim=0)
    mean = S.mean(dim=0)
    std = S.std(dim=0)
    if was_training:
        model.train()
    else:
        model.eval()
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True, help="Training parquet for calibration")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if not os.path.exists(args.parquet):
        raise FileNotFoundError(args.parquet)

    ds = NodeDataset(args.parquet, feature_cols=DEFAULT_FEATS)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_groups)
    model = ScoreMLP(d_in=len(DEFAULT_FEATS), hidden=args.hidden, dropout=args.dropout).to(args.device)
    res = fit_temperature_listwise(model, loader, device=args.device)
    print(f"Fitted temperature: T={res.temperature:.3f} (groups={res.n_groups}, items={res.n_items})")


if __name__ == "__main__":
    main()
