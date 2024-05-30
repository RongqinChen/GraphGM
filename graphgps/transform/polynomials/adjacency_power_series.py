import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter


@torch.no_grad()
def compute_adjacency_power_series(
    data: Data, method, power=8,
    add_full_edge_index: bool = False
):
    assert power > 2
    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.size(1), device=device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    adj = SparseTensor.from_edge_index(
        edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes)
    )
    adj = adj.to_dense()

    base_list = [torch.eye(num_nodes, dtype=torch.float), adj]
    while (k := len(base_list)) <= power:
        base = base_list[k // 2] @ base_list[k - k // 2]
        base_list.append(base)

    polys = torch.stack(base_list, dim=-1)  # n x n x (K+1)
    loop = polys.diagonal().transpose(0, 1)  # n x (K+1)

    poly_adj = SparseTensor.from_dense(polys, has_value=True)
    poly_row, poly_col, poly_val = poly_adj.coo()
    poly_idx = torch.stack([poly_row, poly_col], dim=0)
    data[f"{method}_loop"] = loop
    data[f"{method}_index"] = poly_idx
    data[f"{method}_conn"] = poly_val
    data["log_deg"] = torch.log(deg + 1).unsqueeze_(1)

    if add_full_edge_index:
        if num_nodes ** 2 == poly_row.size(0):
            full_index = poly_idx
        else:
            full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
        data["full_index"] = full_index

    return data
