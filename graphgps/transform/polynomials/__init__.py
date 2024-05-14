from .rrw_bernstain import compute_rrw_bernstain_polynomials
method_dict = {
    'rrw_bern': compute_rrw_bernstain_polynomials,
}


def compute_polynomials(data, method, order, attr_name_abs, attr_name_rel):
    if method in method_dict:
        data = method_dict[method](data, order, attr_name_abs, attr_name_rel)
    else:
        raise NotImplementedError

    return data
