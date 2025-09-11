import torch
from torch import nn


class RankHead(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, var_emb, global_emb=None):
        return self.mlp(var_emb).squeeze(-1)
