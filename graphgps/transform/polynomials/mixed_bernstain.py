from typing import Any, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, scatter
from torch_sparse import SparseTensor
from scipy.special import comb


@torch.no_grad()
def compute_mixed_bernstain_polynomials(
    data: Data,
    order=8,
    add_full_edge_index: bool = False
):
    assert order > 2
    assert order % 2 == 0, "Parameter `order` should be an even number."
    method_name = "mixed_bern"
    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.size(1), device=device)

    adj = SparseTensor.from_edge_index(
        edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes)
    )
    adj = adj.to_dense()

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # Adj1 = I + A_norm.
    index_1, weight_1 = add_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    adj1 = SparseTensor.from_edge_index(
        index_1, weight_1, sparse_sizes=(num_nodes, num_nodes)
    )
    adj1 = adj1.to_dense()

    # Adj2 = I - A_norm.
    index_2, weight_2 = add_self_loops(
        edge_index, -edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    adj2 = SparseTensor.from_edge_index(
        index_2, weight_2, sparse_sizes=(num_nodes, num_nodes)
    )
    adj2 = adj2.to_dense()

    K = order
    base_list = [adj1 @ adj1, adj1 @ adj2, adj2 @ adj2]
    base_dict = {2: base_list}
    for k in range(4, K + 1, 2):
        a_idx = ((k // 2) + 1) // 2 * 2
        b_idx = k - a_idx
        base_list = [
            base_dict[a_idx][1] @ base_dict[b_idx][0],
            base_dict[a_idx][1] @ base_dict[b_idx][1],
            base_dict[a_idx][1] @ base_dict[b_idx][2],
        ]
        base_dict[k] = base_list

    polys = [adj1]
    for k in range(2, K + 1, 2):
        base_list = base_dict[k]
        polys.append(
            base_dict[k][0] * ((2 ** -k) * comb(k, k // 2 - 1))
        )
        polys.append(
            base_dict[k][2] * ((2 ** -k) * comb(k, k // 2 + 1))
        )
    polys.append(base_dict[K][1] * ((2 ** -K) * comb(K, K // 2)))

    polys = torch.stack(polys, dim=-1)  # n x n x (K+2)
    loop = polys.diagonal().transpose(0, 1)  # n x (K+2)
    poly_adj = SparseTensor.from_dense(polys, has_value=True)
    poly_row, poly_col, poly_val = poly_adj.coo()
    poly_idx = torch.stack([poly_row, poly_col], dim=0)
    data[f"{method_name}_loop"] = loop
    data[f"{method_name}_index"] = poly_idx
    data[f"{method_name}_conn"] = poly_val
    data.log_deg = torch.log(deg + 1).unsqueeze_(1)

    if add_full_edge_index:
        if num_nodes ** 2 == poly_row.size(0):
            full_index = poly_idx
        else:
            full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
        data["full_index"] = full_index
    return data


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
