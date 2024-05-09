"""
    The Generalized Metrics encoder
"""

import torch
from torch import nn
import torch_sparse
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.data import Data
from torch_scatter import scatter
import warnings


def full_edge_index(batch: torch.Tensor):
    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    Ns = num_nodes.tolist()
    full_index_list = [
        torch.ones(
            (Ns[idx], Ns[idx]), dtype=torch.short, device=batch.device
        ).nonzero(as_tuple=False).t().contiguous() + cum_nodes[idx]
        for idx in range(batch_size)
    ]
    batch_index_full = torch.cat(full_index_list, dim=1).contiguous()
    return batch_index_full


@register_node_encoder("gm_linear")
class GMLinearNodeEncoder(torch.nn.Module):
    """
    FC_1(GM) + FC_2 (Node-attr)
    note: FC_2 is given by the Typedict encoder of node-attr in some cases
    Parameters:
    num_classes - the number of classes for the embedding mapping to learn
    """

    def __init__(
        self,
        emb_dim,
        out_dim,
        use_bias=False,
        batchnorm=False,
        layernorm=False,
        pe_name="gm",
    ):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        gm = batch[f"{self.name}"]
        gm = self.fc(gm)
        if self.batchnorm:
            gm = self.bn(gm)
        if self.layernorm:
            gm = self.ln(gm)
        if "x" in batch:
            batch.x = batch.x + gm
        else:
            batch.x = gm
        return batch


@register_edge_encoder("gm_linear")
class GMLinearEdgeEncoder(torch.nn.Module):
    """
    Merge GM with given edge-attr and Zero-padding to all pairs of node
    FC_1(GM) + FC_2(edge-attr)
    - FC_2 given by the TypedictEncoder in same cases
    - Zero-padding for non-existing edges in fully-connected graph
    - (optional) add node-attr as the E_{i,i}'s attr
        note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    """

    def __init__(
        self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False, fill_value=0.0,
    ):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info "
            )

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fill_value = 0.0
        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch: Data):
        gm_idx = batch.gm_index
        gm_val = batch.gm_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        gm_val = self.fc(gm_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), gm_val.size(1))
            # zero padding for non-existing edges

        full_index_full = full_edge_index(batch.batch)
        full_attr_pad = self.padding.repeat(full_index_full.size(1), 1)
        out_idx, out_val = torch_sparse.coalesce(
            torch.cat([edge_index, gm_idx, full_index_full], dim=1),
            torch.cat([edge_attr, gm_val, full_attr_pad], dim=0),
            batch.num_nodes, batch.num_nodes, op="add",
        )
        if self.batchnorm:
            out_val = self.bn(out_val)
        if self.layernorm:
            out_val = self.ln(out_val)
        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"fill_value={self.fill_value},"
            f"{self.fc.__repr__()})"
        )
