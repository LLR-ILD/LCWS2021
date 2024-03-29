from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import tqdm

from fitting_util.ild import ild_tag
from fitting_util.prepare import get_val_and_err
import fitting_util.fit_setups as fs
from load_data import load_data, default_brs
from paths import FIG_PATH, N_DATA


fit_starting_values = {
    "H→cc":     0.02718,
    "H→bb":     0.57720,
    "H→μμ":     0.00030,
    "H→ττ":     0.06198,
    "H→Zγ":     0.00170,
    "H→gg":     0.08516+0.00034,
    "H→γγ":     0.00242,
    "H→ZZ*":    0.02616,
    "H→WW":     0.21756,
}
fit_starting_values = default_brs


def get_toy_minima(data, n_data, n_toys, minimization_procedure):
    toys = np.random.multinomial(n_data, data.Y / sum(data.Y), size=n_toys)
    toy_dict = vars(data)
    min_vals = []
    for toy in tqdm.tqdm(toys, total=len(toys), unit=" toys minimizations"):
        toy_dict["Y"] = toy
        minimum, _ = minimization_procedure(SimpleNamespace(**toy_dict), n_data)
        min_vals.append(minimum.values)
    return np.array(min_vals)


def gauss(x, mu, sigma):
    return (2*np.pi*sigma**2)**-.5 * np.exp(-0.5 * (x - mu)**2 / sigma**2)


def uncertainty_toy_study(data, minimization_procedure, toy_dir,
        n_toys = 10_000):
    print(f"Run {toy_dir.name} toys.")
    single_min, _ = minimization_procedure(data, N_DATA)
    toy_minima = get_toy_minima(data, N_DATA, n_toys,
        minimization_procedure)
    param_names = single_min.parameters
    br, br_err = get_val_and_err(single_min, param_names, toy_dir.name)

    # if "Poisson" in toy_dir.name:
    #     br = br / br.sum()
    #     br_err = br_err / br.sum()
    #     toy_minima = (toy_minima.T / toy_minima.sum(axis=1)).T

    def toy_hist(br_name, toy_min_i, br_i, br_err_i):
        fig, ax = plt.subplots(figsize=(4, 4))
        ild_tag(ax)
        ax.set_title(br_name)
        # ax.axvline(data.X0[data.x_names.index(br_name)], color="grey",
        ax.axvline(fit_starting_values[br_name], color="grey",
            linestyle=":", linewidth=2, zorder=3,
            label=f"{fit_starting_values[br_name]:0.5f} SM BR")
        bins = 20
        _, edges, _ = ax.hist(toy_min_i, bins,
            label="\n".join([f"Minima from {n_toys} fits",
                              "on toy counts",
                              "(MC2, Multinomial draws)",]),
            alpha=.8, color="C1", density=True)
        ax.axvline(br_i, color="black",
            label=f"{br_i:0.5f} EECF Minimum")
        x = np.linspace(edges[0], edges[-1], 1000)
        ax.plot(x, gauss(x, br_i, br_err_i),
            label=f"{br_err_i:0.5f} σ from EECF")
        plt.legend(title="\n".join([
                "EECF: Minuit fit on",
                "expected event counts (MC2)",]))
        fig.savefig(toy_dir / f"{br_name.replace('→', '_')}.png")

    for i, br_name in enumerate(param_names):
        toy_min_i = toy_minima[:,i]
        br_i, br_err_i = br[i], br_err[i]
        toy_hist(br_name, toy_min_i, br_i, br_err_i)

    # Add toy plot for H→ZZ*.
    has_dependent_br = len(param_names) == len(fit_starting_values) - 1
    if has_dependent_br:
        dependent_br = next(p for p in fit_starting_values if p not in param_names)
        toy_min_i = 1 - toy_minima.sum(axis=1)
        br_i = 1 - br.sum()
        cov = np.array(single_min.covariance.tolist())
        br_err_i = cov.sum()**.5
        toy_hist(dependent_br, toy_min_i, br_i, br_err_i)


def main():
    for postfix, stat_size_tag, cheat_train_test in [
        ("", "data/overlay_free_higher_stats", False),
        ("_cheat_train_test", "data/overlay_free_higher_stats", True),
        ("_original_stats", "data/overlay_free", False),
        ]:
        data = load_data(
            data_str=str(Path(__file__).parent / stat_size_tag),
            cheat_train_test=cheat_train_test,
        )
        minimization_procedure = fs.binomial_minimization_with_limits
        toy_dir = FIG_PATH / f"toys_multinomial{postfix}"
        toy_dir.mkdir(exist_ok=True, parents=True)
        uncertainty_toy_study(data, minimization_procedure, toy_dir)


if __name__ == "__main__":
    main()