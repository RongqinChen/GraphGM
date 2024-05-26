import torch
from torch import nn
from torch_geometric.graphgym.register import register_node_encoder, register_edge_encoder
import warnings


class OuterAndLinearEncoder(nn.Module):
    def __init__(
        self, name, emb_dim, out_dim, batchnorm=False, layernorm=False,
    ):
        super().__init__()
        self.name = name
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.lin = nn.Linear((self.emb_dim**2 - self.emb_dim) // 2 + self.emb_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        # self.emb_dim = (2 + max_poly_order) * (max_poly_order + 1) // 2
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info "
            )
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, node_x):
        node_h = _upper_triangular_part_of_outer_multiplication(node_x)
        node_h = self.lin(node_h)
        if self.batchnorm:
            node_h = self.bn(node_h)
        if self.layernorm:
            node_h = self.ln(node_h)
        return node_h


@torch.no_grad()
def _upper_triangular_part_of_outer_multiplication(X):
    n = X.size(1)
    M = X.unsqueeze(2)
    idx = torch.triu_indices(n, n, offset=1)
    B = torch.bmm(M, M.transpose(1, 2))
    C = B[:, idx[0], idx[1]]
    D = torch.cat((X, C), 1)
    return D


@register_node_encoder("outmul_linear_poly")
class OutMulLinearNodeEncoder(nn.Module):
    def __init__(
        self, name, emb_dim, out_dim, batchnorm=False, layernorm=False,
    ):
        super().__init__()
        self.encoder = OuterAndLinearEncoder(name, emb_dim, out_dim, batchnorm, layernorm,)

    def forward(self, node_x):
        return self.encoder(node_x)


@register_edge_encoder("outmul_linear_poly")
class OutMulLinearEdgeEncoder(nn.Module):
    def __init__(
        self, name, emb_dim, out_dim, batchnorm=False, layernorm=False,
    ):
        super().__init__()
        self.encoder = OuterAndLinearEncoder(name, emb_dim, out_dim, batchnorm, layernorm,)

    def forward(self, node_x):
        return self.encoder(node_x)
