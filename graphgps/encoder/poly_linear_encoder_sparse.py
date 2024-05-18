"""
    The Generalized Metrics encoder
"""

import torch
from torch import nn
from torch_geometric.graphgym.register import register_node_encoder, register_edge_encoder
import warnings


@register_node_encoder("poly_sparse")
class LinearNodeEncoder(torch.nn.Module):
    def __init__(
        self, name, emb_dim, out_dim, use_bias=False, batchnorm=False, layernorm=False,
    ):
        super().__init__()
        self.name = name
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.emb_dim = emb_dim
        self.fc = nn.Linear(self.emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, node_h):
        node_h = self.fc(node_h)
        if self.batchnorm:
            node_h = self.bn(node_h)
        if self.layernorm:
            node_h = self.ln(node_h)
        return node_h


@register_edge_encoder("poly_sparse")
class SparseLinearEdgeEncoder(torch.nn.Module):
    def __init__(
        self, name, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
    ):
        super().__init__()
        self.name = name
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(self.emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
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

    def forward(self, poly_val):
        poly_val = self.fc(poly_val)
        if self.batchnorm:
            poly_val = self.bn(poly_val)
        if self.layernorm:
            poly_val = self.ln(poly_val)
        return poly_val

    def __repr__(self):
        return (f"{self.__class__.__name__}("f"{self.fc.__repr__()})")
