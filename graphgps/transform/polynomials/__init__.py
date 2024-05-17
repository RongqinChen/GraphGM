from .combined_bernstain import compute_combined_bernstain_polynomials
from .mixed_bernstain import compute_mixed_bernstain_polynomials
method_dict = {
    'comb_bern': compute_combined_bernstain_polynomials,
    'mixed_bern': compute_mixed_bernstain_polynomials,
}


def compute_polynomials(data, method, order, add_full_edge_index):
    if method in method_dict:
        data = method_dict[method](data, order, add_full_edge_index)
    else:
        raise NotImplementedError

    return data
