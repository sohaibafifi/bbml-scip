import torch
from bbml.models.gnn import BipartiteEncoder


def test_bipartite_encoder_shapes_no_graph():
    enc = BipartiteEncoder(d_var=8, d_con=4, d_hidden=16, L=2)
    v = torch.randn(10, 8)
    out = enc(v)
    assert out.shape == (10, 16)


def test_bipartite_encoder_shapes_with_graph():
    enc = BipartiteEncoder(d_var=8, d_con=4, d_hidden=16, L=2)
    v = torch.randn(5, 8)
    c = torch.randn(3, 4)
    # build a simple bipartite edge list: constraints (0..2) to variables (0..4)
    edges = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 3, 1, 4]], dtype=torch.long)
    out = enc(v, c, edges)
    assert out.shape == (5, 16)
