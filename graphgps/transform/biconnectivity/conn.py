import torch
from torch_geometric.data import Data
from collections import defaultdict
from torch_sparse import SparseTensor
from .bcc import Graph4BCC

@torch.no_grad()
def biconnectivity(
    graph: Data,
    method_name='biconn',
    truncated_len=8,
    extension_len=16,
    add_full_edge_index: bool = False,
    edge_weight = 1.
):
    if isinstance(edge_weight, (float, int)):
        weight_mat = torch.ones((N, N)) * edge_weight
    else:
        raise NotImplementedError(edge_weight)

    edge_list = graph.edge_index.T.tolist()
    bcc_list = Graph4BCC(edge_list).BCC()
    bcc_list = [len(bcc) > 2 for bcc in bcc_list]

    N = graph.num_nodes
    conn = torch.zeros((N, N, extension_len), dtype=torch.float32)

    conn_via_truncated_epath_tree(edge_list, conn, truncated_len, weight_mat)
    
    k = truncated_len
    for idx in range(extension_len):
        c = conn[:, :, k-1] @ conn[:, :, idx-k]
        conn[:, :, idx] = c

    loop = conn.diagonal().transpose(0, 1)
    conn_adj = SparseTensor.from_dense(conn, has_value=True)
    conn_row, conn_col, conn_val = conn_adj.coo()
    conn_idx = torch.stack([conn_row, conn_col], dim=0)
    graph[f"{method_name}_loop"] = loop
    graph[f"{method_name}_index"] = conn_idx
    graph[f"{method_name}_conn"] = conn_val

    if add_full_edge_index:
        if N ** 2 == conn_row.size(0):
            full_index = conn_idx
        else:
            full_mat = torch.ones((N, N), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
        graph["full_index"] = full_index
    return graph

def conn_via_truncated_epath_tree(edge_list, conn, truncated_len, weight_mat):
    bcc = defaultdict(set)
    for edge in edge_list:
        bcc[edge[0]].add(edge[1]) 
        bcc[edge[1]].add(edge[0])

    nodes = list(bcc.keys())

    def dfs(path, cumprod):
        if len(path) == truncated_len:
            return
        if len(path) > 2 and path[0] == path[1]:
            # closed path
            return

        relay = path[-1]
        for v in bcc[relay]:
            if v not in path or (len(path) > 1 and path[0]==v):
                # forming a simple path or a closed path
                new_path = path + [v]
                new_cumprod = cumprod * weight_mat[relay, v]
                conn[path[0], v] += new_cumprod
                dfs(new_path, new_cumprod)

    for node in nodes:
        path = [node]
        dfs(path, 1)
    