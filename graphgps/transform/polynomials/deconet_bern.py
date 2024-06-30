import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, scatter
from torch_sparse import SparseTensor
from scipy.special import comb


def extract_sparse(mat: torch.Tensor):
    if mat.dim() > 2:
        index = mat.abs().sum([i for i in range(2, mat.dim())]).nonzero()
    else:
        index = mat.nonzero()
    index = index.t()
    row = index[0]
    col = index[1]
    value = mat[row, col]
    return index, value


@torch.no_grad()
def compute_deconet_bern(
    data: Data, method, power=16, add_full_edge_index: bool = False
):
    assert power >= 2
    assert power % 2 == 0, "Parameter `power` should be an even number."
    K = power

    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.size(1))
    size_tuple = (num_nodes, num_nodes)

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

    K = power
    base_dict = {
        1: [adj1, adj2],
        2: [adj1 @ adj1, adj1 @ adj2, adj2 @ adj2],
    }
    k = 2
    while k < K:
        base_list = [
            base_dict[k][1] @ base_dict[k][0],
            base_dict[k][1] @ base_dict[k][1],
            base_dict[k][1] @ base_dict[k][2],
        ]
        k *= 2
        base_dict[k] = base_list

    k = 2
    while k <= K:
        base_list = [
            base_dict[k][0] * ((2 ** -k) * comb(k, k // 2 - 1)),
            base_dict[k][1] * ((2 ** -k) * comb(k, k // 2 + 0)),
            base_dict[k][2] * ((2 ** -k) * comb(k, k // 2 + 1)),
        ]
        base_dict[k] = base_list
        k *= 2

    polys = sum(base_dict.values(), list())
    polys = torch.stack(polys, dim=-1)  # n x n x ((math.log2(cfg.posenc_Poly.power) + 1) * 3 - 1)
    loop = polys.diagonal().transpose(0, 1)  # n x ((math.log2(cfg.posenc_Poly.power) + 1) * 3 - 1)
    poly_idx, poly_val = extract_sparse(polys)
    data[f"{method}_loop"] = loop
    data[f"{method}_index"] = poly_idx
    data[f"{method}_conn"] = poly_val
    data["sqrt_deg"] = deg.pow(0.5).unsqueeze_(1)

    if add_full_edge_index:
        if num_nodes ** 2 == poly_val.size(0):
            full_index = poly_idx
        else:
            full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
        data["full_index"] = full_index

    return data
