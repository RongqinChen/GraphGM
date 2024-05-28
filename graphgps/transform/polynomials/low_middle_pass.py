import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, scatter
from torch_sparse import SparseTensor
# from scipy.special import comb


@torch.no_grad()
def compute_low_middle_pass_polynomials(
    data: Data, method='low_middle_pass', power=8,
    add_full_edge_index: bool = False
):
    assert power > 2
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

    base_list = [adj1, adj1 @ adj2]

    while True:
        D = len(base_list)
        for d in range(D):
            if len(base_list) == power:
                break
            base = base_list[d] @ base_list[D - 1]
            base_list.append(base)

        if len(base_list) == power:
            break

    # base_list = [
    #     base * ((2 ** -k) * comb(k, k // 2))
    #     for k, base in zip(range(1, power + 1), base_list)
    # ]

    polys = torch.stack(base_list, dim=-1)  # n x n x (K)
    loop = polys.diagonal().transpose(0, 1)  # n x (K)
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
