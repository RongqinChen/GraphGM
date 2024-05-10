from typing import Any, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, scatter
from torch_sparse import SparseTensor
from scipy.special import comb


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if "x" in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


@torch.no_grad()
def add_full_gm2(
    data: Data,
    poly_order=8,
    attr_name_abs="gm",  # name: 'gm'
    attr_name_rel="gm",  # name: ('gm_idx', 'gm_val')
):
    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.size(1), device=device)
    # edge_index, _ = remove_self_loops(edge_index)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # A1 = I - A_norm.
    index_1, weight_1 = add_self_loops(
        edge_index, -edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    base1 = SparseTensor.from_edge_index(
        index_1, weight_1, sparse_sizes=(num_nodes, num_nodes)
    )
    base1 = base1.to_dense()
    # A2 = I + A_norm.
    index_2, weight_2 = add_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    base2 = SparseTensor.from_edge_index(
        index_2, weight_2, sparse_sizes=(num_nodes, num_nodes)
    )
    base2 = base2.to_dense()

    base1_list = [1, base1] + [None] * poly_order
    base2_list = [1, base2] + [None] * poly_order
    for k in range(2, poly_order + 1):
        ldx, rdx = (k) // 2, (k + 1) // 2
        base1_list[k] = base1_list[ldx] @ base1_list[rdx]
        base2_list[k] = base2_list[ldx] @ base2_list[rdx]

    poly_list = [base2_list[poly_order]]
    poly_list += [
        base1_list[k] @ base2_list[poly_order - k]
        for k in range(1, poly_order)
    ]
    poly_list += [base1_list[poly_order]]
    poly_list = [
        poly * (2 ** (-poly_order)) * comb(poly_order, k)
        for k, poly in enumerate(poly_list)
    ]
    gm_list = poly_list + [
        poly_list[idx] * poly_list[jdx]
        for idx in range(poly_order + 1)
        for jdx in range(idx, poly_order + 1)
    ]
    gms = torch.stack(gm_list, dim=-1)  # n x n x (K+(K-1)*K/2)
    loops = gms.diagonal().transpose(0, 1)  # n x (K+(K-1)*K/2)
    gm_adj = SparseTensor.from_dense(gms, has_value=True)
    gm_row, gm_col, gm_val = gm_adj.coo()
    gm_idx = torch.stack([gm_row, gm_col], dim=0)
    data = add_node_attr(data, loops, attr_name=attr_name_abs)
    data = add_node_attr(data, gm_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, gm_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1).unsqueeze_(1)
    return data