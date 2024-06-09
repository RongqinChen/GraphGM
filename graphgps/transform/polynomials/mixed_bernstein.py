import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, scatter
from torch_sparse import SparseTensor
from scipy.special import comb


@torch.no_grad()
def compute_mixed_bernstein_polynomials(
    data: Data, method, power=8,
    add_full_edge_index: bool = False
):
    assert power > 2
    assert power % 2 == 0, "Parameter `power` should be an even number."
    K = power

    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.size(1))
    size_tuple = (num_nodes, num_nodes)

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, size_tuple, True, True)
    adj = adj.to_dense()

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # Adj1 = I + A_norm.
    index_1, weight_1 = add_self_loops(edge_index, edge_weight, 1.0, num_nodes)
    adj1 = SparseTensor.from_edge_index(index_1, weight_1, size_tuple, True, True)
    adj1 = adj1.to_dense()

    # Adj2 = I - A_norm.
    index_2, weight_2 = add_self_loops(edge_index, -edge_weight, 1.0, num_nodes)
    adj2 = SparseTensor.from_edge_index(index_2, weight_2, size_tuple, True, True)
    adj2 = adj2.to_dense()

    identity = torch.eye(num_nodes)
    base1_list = [identity, adj1] + [None] * (K - 1)
    base2_list = [identity, adj2] + [None] * (K - 1)
    for k in range(2, K + 1):
        lidx, ridx = k // 2, k - k // 2
        base1_list[k] = base1_list[lidx] @ base1_list[ridx]
        base2_list[k] = base2_list[lidx] @ base2_list[ridx]

    basis1 = [base / (2 ** k) for k, base in enumerate(base1_list[:-1])]

    bp_base_list = [base1_list[K - k] @ base2_list[k] for k in range(K + 1)]
    bp_coef_list = [2 ** (-K) * comb(K, k) for k in range(K + 1)]
    basis2 = [bp_base_list[k] * bp_coef_list[k] for k in range(K + 1)]

    polys = torch.stack(basis1 + basis2, dim=-1)  # n x n x (K+2)
    loop = polys.diagonal().transpose(0, 1)  # n x (K+2)
    poly_adj = SparseTensor.from_dense(polys, has_value=True)
    poly_row, poly_col, poly_val = poly_adj.coo()
    poly_idx = torch.stack([poly_row, poly_col], dim=0)
    data[f"{method}_loop"] = loop
    data[f"{method}_index"] = poly_idx
    data[f"{method}_conn"] = poly_val
    data["sqrt_deg"] = deg.pow(0.5).unsqueeze_(1)

    if add_full_edge_index:
        if num_nodes ** 2 == poly_row.size(0):
            full_index = poly_idx
        else:
            full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
        data["full_index"] = full_index

    return data
