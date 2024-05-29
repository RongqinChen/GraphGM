import torch
from torch import nn
from torch_geometric.graphgym.register import register_layer

import warnings


@register_layer('simple_linear')
class Linear(nn.Module):
    def __init__(
        self, in_dim, out_dim, batchnorm=False, layernorm=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim, False)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info "
            )
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, node_x):
        node_h = self.lin(node_x)
        if self.batchnorm:
            node_h = self.bn(node_h)
        if self.layernorm:
            node_h = self.ln(node_h)
        return node_h
