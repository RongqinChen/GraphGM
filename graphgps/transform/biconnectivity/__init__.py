from .conn import biconnectivity
method_dict = {
    'biconn': biconnectivity,
}


def compute_biconnectivity(data, method, truncated_len, extension_len, add_full_edge_index, edge_weight):
    if method in method_dict:
        data = method_dict[method](data, truncated_len, extension_len, add_full_edge_index, edge_weight)
    else:
        raise NotImplementedError

    return data
