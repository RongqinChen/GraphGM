# from .mixed_bernstain import compute_mixed_bernstain_polynomials
from .low_middle_pass import compute_low_middle_pass_polynomials

method_dict = {
    # 'mixed_bern': compute_mixed_bernstain_polynomials,
    'low_middle_pass': compute_low_middle_pass_polynomials,
}


def compute_polynomials(data, method, order, add_full_edge_index):
    if method in method_dict:
        data = method_dict[method](data, order, method, add_full_edge_index)
    else:
        raise NotImplementedError

    return data
