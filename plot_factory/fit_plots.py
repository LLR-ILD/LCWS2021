import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from fitting_util.comparison_with_others import comparison_with_others
from fitting_util.prepare import prepare_data, get_val_and_err, shift_x
import fitting_util.fit_setups as fs
from load_data import default_brs, load_data
from paths import FIG_PATH, N_DATA


def br_estimates_plot(m, data, n_data):
    m_dict, br_idx, param_names, ordered_X0 = prepare_data(m, data)
    fig, axs = plt.subplots(nrows=2, figsize=(4, 6), sharex=True)
    for i, (m_name, mm) in enumerate(m_dict.items()):
        x = shift_x(i, br_idx, len(m_dict))
        y, y_err = get_val_and_err(mm, param_names, m_name)
        axs[0].errorbar(x, y, yerr=y_err, xerr=.3, fmt="o", label=m_name,
                        color=f"C{i+1}")
        axs[1].scatter(x, y_err, marker="*", color=f"C{i+1}")
    axs[0].bar(br_idx, ordered_X0, alpha=.75, label="Input BRs", color="C0")
    axs[0].set_ylabel("BR estimate after MIGRAD")
    axs[0].legend()
    axs[1].set_ylabel("HESSE 67% CL interval")
    axs[1].set_xticks(br_idx)
    axs[1].set_xticklabels(param_names, rotation=90)
    axs[0].set_title(f"n_data={n_data}")
    fig.tight_layout()
    return fig


def relative_error_plot(m, data, n_data, no_legend=False):
    """ Just a convenience wrapper.
    """
    fig = comparison_with_others(m, data,
        compare_to_global_couplings=False, draw_lines=False)
    ax = fig.get_axes()[0]
    ax.set_title(f"n_data={n_data}")
    ax.get_legend().remove()
    fig.tight_layout()
    return fig


def correlations_plot(minimizer):
    labels = minimizer.covariance.to_table()[-1]
    cov = np.array(minimizer.covariance.tolist())
    corr = (cov / cov.diagonal()**.5).T / cov.diagonal()**.5
    corr = corr - np.eye(corr.shape[0]) # We do not want to color the diagonal.
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xticks(np.arange(cov.shape[1]))
    ax.set_yticks(np.arange(cov.shape[0]))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    for text_y, row in enumerate(corr):
        for text_x, val in enumerate(row):
            if text_x == text_y: continue
            color = "black" # if val > .3*cov.max() else "white"
            plt.text(text_x, text_y,
                f"{val:4.3f}",
                ha="center", va="center", color=color)
    ax.imshow(corr, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
    ax.set_title("Correlations")
    fig.tight_layout()
    return fig


def make_bias_table(min_result, fit_starting_values, table_name="bias_table"):
    def h_to_latex(h_str):
        return "$"+(h_str
            ).replace("→μμ", "\to \mu\mu"
            ).replace("→ττ", "\to \tau\tau"
            ).replace("→Zγ", "\to Z\gamma"
            ).replace("→ZZ*", "\to ZZ^*"
            ).replace("→γγ", "\to \gamma\gamma"
            ).replace("→", "\to "
            )+"$"
    param_names = min_result.parameters
    if "H→ZZ*" not in param_names:
        param_names = [p for p in param_names] + ["H→ZZ*"]
    br, br_err = get_val_and_err(min_result, param_names, "bias_table")

    pd.DataFrame({
            "SM BR": [100*fit_starting_values[br] for br in param_names],
            "minimum": 100*br,
            "$\sigma$": 100*br_err,
            }, index=list(map(h_to_latex, param_names)),
        ).to_latex(buf=(FIG_PATH / f"{table_name}.tex"),
                   float_format="%0.3f", escape=False)


minimization_choices = dict((
    ("Gaussian LSQ", fs.gaussian_minimization),
    ("Gaussian LSQ - (0,1) limits", fs.gaussian_minimization_with_limits),
    ("Gaussian LSQ w binomial unc.", fs.gaussian_minimization_binomial_error),
    ("Gaussian LSQ w binomial unc. - (0,1) limits", fs.gaussian_minimization_with_limits_binomial_error),
    ("Multinomial", fs.binomial_minimization),
    ("Multinomial - (0,1) limits", fs.binomial_minimization_with_limits),
    ("Poisson", fs.poisson_minimization),
))

def many_minimizer_plot(data, prefix=""):
    m, bin_counts_models = {}, {}
    for k, do_minimization in minimization_choices.items():
        m[k], bin_counts_models[k] = do_minimization(data, N_DATA)

    fig = br_estimates_plot(m, data, N_DATA)
    fig.savefig(FIG_PATH / f"{prefix}many_br_estimates.png")

    fig = relative_error_plot(m, data, N_DATA, no_legend=True)
    fig.savefig(FIG_PATH / f"{prefix}many_br_relative_error.png")


def multinomial_minimizer_plot(data, prefix=""):
    m = {"Multinomial": fs.binomial_minimization(data, N_DATA)[0]}

    fig = br_estimates_plot(m, data, N_DATA)
    fig.savefig(FIG_PATH / f"{prefix}br_estimates.png")

    fig = relative_error_plot(m, data, N_DATA)
    fig.savefig(FIG_PATH / f"{prefix}br_relative_error.png")

    fig = correlations_plot(m["Multinomial"])
    fig.savefig(FIG_PATH / f"{prefix}default_correlations.png")

    fig = comparison_with_others(m, data,
        draw_lines=True, compare_to_global_couplings=True)
    fig.savefig(FIG_PATH / f"{prefix}comparison_with_others.png")

    make_bias_table(m["Multinomial"], default_brs)

    smaller_data = load_data(data_str=str(Path(__file__).parent / "data/overlay_free"))
    m_smaller = {"Multinomial": fs.binomial_minimization(smaller_data, N_DATA)[0]}
    make_bias_table(m_smaller["Multinomial"], default_brs,
        table_name="bias_table_original_stats")



def highly_correlated_fit():
    highly_corr_data = load_data(
        data_str=str(Path(__file__).parent / "data/highly_correlated"))


    m, bin_counts_models = {}, {}
    for k, do_minimization in minimization_choices.items():
        m[k], bin_counts_models[k] = do_minimization(highly_corr_data, N_DATA)

    fig = correlations_plot(m["Multinomial"])
    fig.savefig(FIG_PATH / "highly_correlated.png")

    fig = br_estimates_plot(m, highly_corr_data, N_DATA)
    fig.set_size_inches(5, 6)
    fig.savefig(FIG_PATH / f"highly_correlated_many_br_estimates.png")


def main():
    data = load_data(data_str=str(Path(__file__).parent / "data/overlay_free_higher_stats"))
    many_minimizer_plot(data)
    multinomial_minimizer_plot(data)

    highly_correlated_fit()


if __name__ == "__main__":
    main()