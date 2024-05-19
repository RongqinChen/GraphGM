# """
#     The Bern encoder
# """

# import torch
# from torch import nn
# # from torch.nn import functional as F
# # from ogb.utils.features import get_bond_feature_dims
# import torch_sparse

# # import torch_geometric as pyg
# from torch_geometric.graphgym.register import (
#     register_edge_encoder,
#     register_node_encoder,
# )
# from torch_geometric.data import Data
# # from torch_geometric.utils import (
# #     remove_self_loops,
# #     add_remaining_self_loops,
# #     add_self_loops,
# # )
# from torch_scatter import scatter
# import warnings


# def compute_full_index(batch: torch.Tensor):
#     batch_size = batch.max().item() + 1
#     one = batch.new_ones(batch.size(0))
#     num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
#     cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
#     Ns = num_nodes.tolist()
#     full_index_list = [
#         torch.ones(
#             (Ns[idx], Ns[idx]), dtype=torch.short, device=batch.device
#         ).nonzero(as_tuple=False).t().contiguous() + cum_nodes[idx]
#         for idx in range(batch_size)
#     ]
#     batch_index_full = torch.cat(full_index_list, dim=1).contiguous()
#     return batch_index_full


# @register_node_encoder("bern_linear")
# class BernLinearNodeEncoder(torch.nn.Module):
#     """
#     FC_1(Bern) + FC_2 (Node-attr)
#     note: FC_2 is given by the Typedict encoder of node-attr in some cases
#     Parameters:
#     num_classes - the number of classes for the embedding mapping to learn
#     """

#     def __init__(
#         self,
#         emb_dim,
#         out_dim,
#         use_bias=False,
#         batchnorm=False,
#         layernorm=False,
#         pe_name="bern",
#     ):
#         super().__init__()
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         self.name = pe_name

#         self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc.weight)

#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)

#     def forward(self, batch):
#         # Encode just the first dimension if more exist
#         bern = batch[f"{self.name}"]
#         bern = self.fc(bern)

#         if self.batchnorm:
#             bern = self.bn(bern)

#         if self.layernorm:
#             bern = self.ln(bern)

#         if "x" in batch:
#             batch.x = batch.x + bern
#         else:
#             batch.x = bern

#         return batch


# @register_edge_encoder("bern_linear")
# class BernLinearEdgeEncoder(torch.nn.Module):
#     """
#     Merge Bern with given edge-attr and Zero-padding to all pairs of node
#     FC_1(Bern) + FC_2(edge-attr)
#     - FC_2 given by the TypedictEncoder in same cases
#     - Zero-padding for non-existing edges in fully-connected graph
#     - (optional) add node-attr as the E_{i,i}'s attr
#         note: assuming  node-attr and edge-attr is with the same dimension after Encoders
#     """

#     def __init__(
#         self,
#         emb_dim,
#         out_dim,
#         batchnorm=False,
#         layernorm=False,
#         use_bias=False,
#         fill_value=0.0,
#     ):
#         super().__init__()
#         # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         if self.batchnorm or self.layernorm:
#             warnings.warn(
#                 "batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info "
#             )

#         self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#         self.fill_value = 0.0
#         padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
#         self.register_buffer("padding", padding)

#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)

#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)

#     def forward(self, batch: Data):
#         bern_idx = batch.bern_index
#         bern_val = batch.bern_val
#         edge_index = batch.edge_index
#         edge_attr = batch.edge_attr
#         bern_val = self.fc(bern_val)

#         if edge_attr is None:
#             edge_attr = edge_index.new_zeros(edge_index.size(1), bern_val.size(1))
#             # zero padding for non-existing edges

#         if self.batchnorm:
#             bern_val = self.bn(bern_val)

#         if self.layernorm:
#             bern_val = self.ln(bern_val)

#         full_index = compute_full_index(batch.batch)
#         full_pad = self.padding.repeat(full_index.size(1), 1)
#         out_idx, out_val = torch_sparse.coalesce(
#             torch.cat([edge_index, bern_idx, full_index], dim=1),
#             torch.cat([edge_attr, bern_val, full_pad], dim=0),
#             batch.num_nodes, batch.num_nodes, op="add",
#         )

#         batch.edge_index, batch.edge_attr = out_idx, out_val
#         return batch

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}("
#             f"fill_value={self.fill_value},"
#             f"{self.fc.__repr__()})"
#         )
