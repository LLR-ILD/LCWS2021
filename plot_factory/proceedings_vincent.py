"""Create the plot type that Vincent prefers.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from changed_brs import changed_brs
import fitting_util.fit_setups as fs
from fitting_util.ild import ild_tag
from fitting_util.prepare import prepare_data, get_val_and_err, shift_x
from fit_plots import br_estimates_plot
from load_data import load_data, default_brs
from paths import FIG_PATH, N_DATA
from toys import uncertainty_toy_study


def vincent_plot(m, data, n_data):
    m_dict, br_idx, param_names, ordered_X0 = prepare_data(m, data)
    fig, ax = plt.subplots(figsize=(4, 3))
    ild_tag(ax)
    for i, (m_name, mm) in enumerate(m_dict.items()):
        x = shift_x(i, br_idx, len(m_dict))
        y, y_err = get_val_and_err(mm, param_names, m_name)
        ax.errorbar(x, y / ordered_X0, yerr=y_err / ordered_X0, xerr=.3,
            barsabove=True,
            fmt=".", label=m_name, color=f"C{i+1}")
    ax.axhline(1, color="grey", ls=":")
    ax.set_ylabel("reconstructed / input")
    ax.set_xticks(br_idx)
    ax.set_xticklabels(param_names, rotation=90)
    ax.set_ylim((1 - 0.4, 1 + 0.4))
    fig.tight_layout()
    return fig


def ylim_adjusted_br_estimates_plot(m, data, n_data):
    fig = br_estimates_plot(m, data, n_data)
    fig.get_axes()[0].set_ylim((None, 0.6))
    fig.get_axes()[1].set_ylim((None, 0.007))
    return fig


def main():
    folder = FIG_PATH.parents[1] / "img_proceedings"
    folder.mkdir(exist_ok=True)
    for brs, prefix in [(None, ""), (changed_brs, "changed_")]:
        data = load_data(
            data_str=str(Path(__file__).parent / "data/overlay_free_higher_stats"),
            brs=brs,
        )
        m = {"Multinomial": fs.binomial_minimization_with_limits(data, N_DATA)[0]}
        fig = ylim_adjusted_br_estimates_plot(m, data, N_DATA)
        fig.savefig(folder / f"{prefix}br_estimates.png")

        fig = vincent_plot(m, data, N_DATA)
        fig.savefig(folder / f"{prefix}vincent_plot.png")

    toy_dict = vars(data)
    rng = np.random.default_rng(seed=0)
    toy_dict["Y"] = rng.multinomial(N_DATA, data.Y / sum(data.Y))
    toy_data = SimpleNamespace(**toy_dict)
    m = {"Multinomial": fs.binomial_minimization_with_limits(data, N_DATA)[0]}
    fig = vincent_plot(m, toy_data, N_DATA)
    fig.get_axes()[0].set_ylim((0.4, 1.6))
    fig.savefig(folder / f"vincent_toy_plot.png")





if __name__ == "__main__":
    main()