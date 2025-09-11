from typing import Optional
import torch
from torch import nn

from bbml.models.gnn import BipartiteEncoder
from bbml.models.heads import RankHead


class GraphRanker(nn.Module):
    """Bipartite encoder + ranking head.

    forward(var_feat [n_var,dv], con_feat [n_con,dc]|None, edge_index [2,E]|None)
    returns scores [n_var].
    """

    def __init__(
        self,
        d_var: int,
        d_con: int,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.enc = BipartiteEncoder(d_var=d_var, d_con=d_con, d_hidden=hidden, L=layers, dropout=dropout)
        self.head = RankHead(d=hidden)

    def forward(
        self,
        var_feat: torch.Tensor,
        con_feat: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v = self.enc(var_feat, con_feat=con_feat, edge_index=edge_index)
        return self.head(v)
