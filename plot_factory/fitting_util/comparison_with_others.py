import matplotlib.pyplot as plt
import numpy as np

from .ild import ild_tag
from .prepare import prepare_data, get_val_and_err, shift_x


global_fit_ilc_250 = {  # Table 1 in J.Tian & K. Fujii. https://www.sciencedirect.com/science/article/pii/S2405601415006161
    "H→cc":     0.068,
    "H→bb":     0.053,
    "H→ττ":     0.057,
    "H→gluons": 0.064,
    "H→γγ":     0.18,
    "H→ZZ*":    0.08,
    "H→WW":     0.048,
    # "ΓH":       0.11,
    # "H→inv.":   0.0095,
}
for k, v in global_fit_ilc_250.items():
    # Scale errors to the H20 ILC250 scenario that I am considering.
    global_fit_ilc_250[k] = v * (250 / 2000)**.5

# SFitter: 3000 ifb of 14 TeV LHC,
#           250 ifb of 250 GeV ILC.
# Table I in SFitter 2013: https://inspirehep.net/literature/1209590
sfitter_hl_lhc = {
    "H→bb":     0.17,
    "H→ττ":     0.09,
    "H→gluons": 0.105,
    "H→γγ":     0.08,
    "H→ZZ*":    0.08,
    "H→WW":     0.075,
}
sfitter_hl_lhc_improved = {
    "H→bb":     0.145,
    "H→ττ":     0.07,
    "H→gluons": 0.095,
    "H→γγ":     0.06,
    "H→ZZ*":    0.07,
    "H→WW":     0.06,
}
sfitter_ilc = {
    "H→bb":     0.145,
    "H→cc":     0.095,
    "H→ττ":     0.08,
    "H→gluons": 0.08,
    "H→γγ":     0.155,
    "H→ZZ*":    0.015,
    "H→WW":     0.055,
}
sfitter_ilc_scaled = {}
for k, v in sfitter_ilc.items():
    # Scale errors to the H20 ILC250 scenario that I am considering.
    sfitter_ilc_scaled[k] = v * (250 / 2000)**.5
sfitter_lhc_ilc = {
    "H→bb":     0.045,
    "H→cc":     0.06,
    "H→ττ":     0.05,
    "H→gluons": 0.06,
    "H→γγ":     0.06,
    "H→ZZ*":    0.0095,
    "H→WW":     0.04,
}

peskin_ilc_zh = {  # https://arxiv.org/abs/1207.2516
    # "H→ZZ*":    0.19,  # σ(ZH)BR(ZZ)
    "H→ZZ*":    0.025,  # σ(ZH)
    # "H→bb":     0.105,  # σ(WW)BR(bb)
    "H→bb":     0.011,  # σ(ZH)BR(bb)
    "H→cc":     0.074,  # σ(ZH)BR(cc)
    "H→ττ":     0.042,  # σ(ZH)BR(ττ)
    "H→gluons": 0.06,  # σ(ZH)BR(bb)
    "H→γγ":     0.38,  # σ(ZH)BR(γγ)
    "H→WW":     0.064,  # σ(ZH)BR(WW)
    # "H→inv.":   0.005,  # σ(ZH)BR(inv.)
}
for k, v in peskin_ilc_zh.items():
    # Scale errors to the H20 ILC250 scenario that I am considering.
    peskin_ilc_zh[k] = v * (250 / 2000)**.5


coupling_projections = {}
coupling_projections["ILC 250 global coupling fit [1]"] = global_fit_ilc_250
# coupling_projections["SFitter HL-LHC"] = sfitter_hl_lhc
coupling_projections["SFitter HL-LHC improved [2]"] = sfitter_hl_lhc_improved
# coupling_projections["SFitter ILC250 250 ifb"] = sfitter_ilc
# coupling_projections["SFitter ILC250 scaled Lumi"] = sfitter_ilc_scaled
# coupling_projections["SFitter LHC+(ILC250 250 ifb)"] = sfitter_lhc_ilc
# coupling_projections["Peskin 2012 through σ(ZH)"] = peskin_ilc_zh


def get_deltas_from_couplings(couplings, param_names):
    xs, Deltas = [], []
    for i, pn in enumerate(param_names):
        if pn in couplings:
            xs.append(i)
            Deltas.append(couplings[pn])
    return np.array(xs), np.array(Deltas)


def comparison_with_others(m, data, draw_lines=True,
        compare_to_global_couplings=True,
    ):
    m_dict, br_idx, param_names, ordered_X0 = prepare_data(m, data)
    fig, ax = plt.subplots(figsize=(4, 3))
    ild_tag(ax)
    for i, (m_name, mm) in enumerate(m_dict.items()):
        x = shift_x(i, br_idx, len(m_dict))
        y, y_err = get_val_and_err(mm, param_names, m_name)
        Deltas = y_err / y / 2 # Factor 2 to go from BR or CS to coupling.
        ax.scatter(x, Deltas, marker="*", color=f"C{i+1}",  label=m_name)
    if draw_lines: ax.plot(x, Deltas, color=f"C{i+1}", alpha=.3)
    if compare_to_global_couplings:
        if compare_to_global_couplings == True:
            compare_to_global_couplings = coupling_projections
        for j, c_name in enumerate(compare_to_global_couplings):
            couplings = coupling_projections[c_name]
            x, Deltas = get_deltas_from_couplings(couplings, param_names)
            x = x + 0.01
            ax.scatter(x, Deltas, marker="_", color=f"C{i+j+2}",  label=c_name)
            if draw_lines: ax.plot(x, Deltas, color=f"C{i+j+2}", alpha=.3)
    ax.set_ylabel("$\Delta_X = g_X / g_X^{SM} - 1$")
    ax.set_xticks(br_idx)
    ax.set_xticklabels(param_names, rotation=90)
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig