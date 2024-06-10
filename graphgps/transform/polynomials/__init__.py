from .adjacency_power_series import compute_adjacency_power_series
from .bernstein import compute_bernstein_polynomials
from .mixed_sym_bernstein import compute_mixed_sym_bernstein_polynomials
from .mixed_middle_bernstein import compute_mixed_middle_bernstein_polynomials
from .mixed_low_bernstein import compute_low_bernstein_polynomials

method_dict = {
    'adj_powers': compute_adjacency_power_series,
    'bern': compute_bernstein_polynomials,
    'mixed_sym_bern': compute_mixed_sym_bernstein_polynomials,
    'mixed_middle_bern': compute_mixed_middle_bernstein_polynomials,
    'mixed_low_bern': compute_low_bernstein_polynomials,
}


def compute_polynomials(data, method, power, add_full_edge_index):
    if method in method_dict:
        data = method_dict[method](data, method, power, add_full_edge_index)
    else:
        raise NotImplementedError

    return data
