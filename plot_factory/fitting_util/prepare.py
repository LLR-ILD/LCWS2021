import numpy as np


def prepare_data(m, data):
    if isinstance(m, dict):
        m_dict = m
    else:
        m_dict = {"MIGRAD": m}
    param_names = set().union(*(set(v.parameters) for v in m_dict.values()))
    try:
        param_names = sorted(param_names, key=str.lower)
    except:
        print("param_names could not be sorted.")
    br_idx = np.arange(len(param_names))
    new_X0_ids = [data.x_names.index(p) for p in param_names]
    ordered_X0 = [data.X0[i] for i in new_X0_ids]
    return m_dict, br_idx, param_names, ordered_X0


def get_val_and_err(mm, param_names, m_name=""):
    val_or_None = lambda k: mm.values[k] if k in mm.parameters else np.NAN
    err_or_None = lambda k: mm.errors[k] if k in mm.parameters else np.NAN
    y = np.array(list(map(val_or_None, param_names)))
    y_err = np.array(list(map(err_or_None, param_names)))
    if sum(y) > 1:
        if "oisson" not in m_name:
            print("WARNING: This should only happen for "
                f"Poisson-type likelihoods! {m_name=}, {sum(y)=}")
        y_err = y_err / sum(y)
        y = y / sum(y)
    return y, y_err


def shift_x(i, old_x, n_instances):
    return old_x + 0.4 * (.5 + i - n_instances / 2) / n_instances