from .mixed_bernstein import compute_mixed_bernstein_polynomials

method_dict = {
    'mixed_bern': compute_mixed_bernstein_polynomials,
}


def compute_polynomials(data, method, power, add_full_edge_index):
    if method in method_dict:
        data = method_dict[method](data, method, power, add_full_edge_index)
    else:
        raise NotImplementedError

    return data
