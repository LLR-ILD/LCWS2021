import numpy as np
from pathlib import Path

import fitting_util.fit_setups as fs
from fit_plots import br_estimates_plot
from load_data import load_data, default_brs
from paths import FIG_PATH, N_DATA
from toys import uncertainty_toy_study


changed_brs = np.array(list({
    "H→cc":     0.02718,
    "H→bb":     0.57720-.15,
    "H→μμ":     0.00030,
    "H→ττ":     0.06198,
    "H→Zγ":     0.00170,
    "H→gg":     0.08516+0.00034,
    "H→γγ":     0.00242,
    "H→ZZ*":    0.02616,
    "H→WW":     0.21756+.15,
}.values()))


def main():
    data = load_data(data_str=str(Path(__file__).parent / "data/overlay_free_higher_stats"),
        brs=changed_brs,
    )
    minimization_procedure = fs.binomial_minimization_with_limits
    toy_dir = FIG_PATH / "toys_multinomial_changed"
    toy_dir.mkdir(exist_ok=True, parents=True)
    uncertainty_toy_study(data, minimization_procedure, toy_dir)

    m = {"Multinomial": fs.binomial_minimization_with_limits(data, N_DATA)[0]}
    fig = br_estimates_plot(m, data, N_DATA)
    fig.savefig(FIG_PATH / "changed_br_estimates.png")


if __name__ == "__main__":
    main()