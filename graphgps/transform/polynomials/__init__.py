from .adjacency_power_series import compute_adjacency_power_series
from .bernstein import compute_bernstein_polynomials
# from .mixed_sym_bernstein import compute_mixed_sym_bernstein_polynomials
from .mixed_low_bernstein import compute_low_bernstein_polynomials
from .deco_bernstein import compute_deco_bernstein_polynomials
from .spect_adj_powers import compute_spect_adj_powers
from .bern_lp import compute_bernstein_landing_probility


method_dict = {
    'adj_powers': compute_adjacency_power_series,
    'bern': compute_bernstein_polynomials,
    # 'mixed_sym_bern': compute_mixed_sym_bernstein_polynomials,
    'mixed_low_bern': compute_low_bernstein_polynomials,
    'deco_bern': compute_deco_bernstein_polynomials,
    'spect_adj_powers': compute_spect_adj_powers,
    'bern_lp': compute_bernstein_landing_probility
}


def compute_polynomials(data, method, power, add_full_edge_index):
    if method in method_dict:
        data = method_dict[method](data, method, power, add_full_edge_index)
    else:
        raise NotImplementedError

    return data
