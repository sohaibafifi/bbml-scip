import torch
from torch import nn
from typing import Optional

try:  # Optional acceleration via PyG
    from torch_geometric.nn import SAGEConv

    _PYG = True
except Exception:  # pragma: no cover - optional dep
    SAGEConv = None
    _PYG = False


class BipartiteEncoder(nn.Module):
    """Lightweight bipartite encoder (variables ↔ constraints).

    Inputs
    - var_feat: [n_var, d_var]
    - con_feat: [n_con, d_con] (optional; if None, returns var-only MLP embeddings)
    - edge_index: LongTensor [2, E] with edges from constraints (row 0) to variables (row 1)

    Returns
    - var embeddings [n_var, d_hidden]
    """

    def __init__(
        self,
        d_var: int = 32,
        d_con: int = 32,
        d_hidden: int = 64,
        L: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h = d_hidden
        self.L = L
        self.var_mlp = nn.Sequential(nn.Linear(d_var, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden))
        self.con_mlp = nn.Sequential(nn.Linear(d_con, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.act = nn.ReLU()

        if _PYG:
            # Two-direction message passing per layer
            self.c2v = nn.ModuleList([SAGEConv((d_hidden, d_hidden), d_hidden) for _ in range(L)])
            self.v2c = nn.ModuleList([SAGEConv((d_hidden, d_hidden), d_hidden) for _ in range(L)])
        else:
            print("Warning: PyG not found; using slower bipartite encoder.")
            # Fallback: simple residual + mean-aggregation without PyG
            self.var_lin = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(L)])
            self.con_lin = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(L)])

    def _mean_aggregate(self, src: torch.Tensor, dst_size: int, edge_index: torch.Tensor) -> torch.Tensor:
        """Mean aggregate messages from src nodes to dst nodes via edge_index.
        edge_index[0]: src indices in [0, n_src), edge_index[1]: dst indices in [0, n_dst)
        src: [n_src, h] → out: [n_dst, h]
        """
        if edge_index is None or src.numel() == 0:
            return torch.zeros((dst_size, src.size(-1)), device=src.device, dtype=src.dtype)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        m = src[src_idx]
        out = torch.zeros((dst_size, src.size(-1)), device=src.device, dtype=src.dtype)
        out.index_add_(0, dst_idx, m)
        deg = torch.bincount(dst_idx, minlength=dst_size).clamp(min=1).to(out.dtype).unsqueeze(-1)
        return out / deg

    def forward(
        self,
        var_feat: torch.Tensor,
        con_feat: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v = self.var_mlp(var_feat)
        if con_feat is None or edge_index is None:
            # No graph provided; return var-only features
            return v

        c = self.con_mlp(con_feat)

        if _PYG:
            for l in range(self.L):
                # constraints → variables, then variables → constraints
                v = self.act(v + self.c2v[l]((c, v), edge_index))
                v = self.dropout(v)
                c = self.act(c + self.v2c[l]((v, c), edge_index.flip(0)))
                c = self.dropout(c)
        else:
            n_v = v.size(0)
            n_c = c.size(0)
            for l in range(self.L):
                # c → v
                agg_cv = self._mean_aggregate(c, n_v, edge_index)
                v = self.act(self.var_lin[l](v) + agg_cv)
                v = self.dropout(v)
                # v → c (use reversed edges: variables are sources, constraints are dest)
                rev_edge = torch.stack((edge_index[1], edge_index[0]), dim=0)
                agg_vc = self._mean_aggregate(v, n_c, rev_edge)
                c = self.act(self.con_lin[l](c) + agg_vc)
                c = self.dropout(c)

        return v
