from .adjacency_power_series import compute_adjacency_power_series
from .bernstein import compute_bernstein_polynomials
from .mixed_bernstein import compute_mixed_bernstein_polynomials

method_dict = {
    'adj_powers': compute_adjacency_power_series,
    'bern': compute_bernstein_polynomials,
    'mixed_bern': compute_mixed_bernstein_polynomials,
}


def compute_polynomials(data, method, power, add_full_edge_index):
    if method in method_dict:
        data = method_dict[method](data, method, power, add_full_edge_index)
    else:
        raise NotImplementedError

    return data
