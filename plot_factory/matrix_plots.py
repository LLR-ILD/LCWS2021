import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from fitting_util.ild import ild_tag
from load_data import load_data
from paths import FIG_PATH, N_DATA

def plot_matrix(data, ax=None, omit_zero=True):
    M = data.M
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    ild_tag(ax)

    def set_labels(ax):
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_xticks(np.arange(M.shape[1]))
        try:
            ax.set_ylabel("class", fontsize=14)
            ax.set_yticklabels(data.y_names, fontsize=12)
        except:
            pass
        try:
            ax.set_xlabel("BR", fontsize=14)
            ax.set_xticklabels(data.x_names, rotation=90, fontsize=12)
        except:
            pass

    def set_numbers(ax):
        for text_y, row in enumerate(M):
            for text_x, prob_class_given_br in enumerate(row):
                color = "black" if prob_class_given_br > .3*M.max() else "white"
                if omit_zero and prob_class_given_br == 0:
                    continue
                plt.text(text_x, text_y,
                    f"{prob_class_given_br:4.2f}",
                    ha="center", va="center", color=color)

    ax.imshow(M)
    set_labels(ax)
    set_numbers(ax)
    return ax


def probability_and_expected(data, prefix):
    def probability_matrix():
        fig, ax = plt.subplots(figsize=(8, 10))
        ild_tag(ax)
        ax.set_title("Matrix entries P(Class|BR) [%]", fontsize=14)
        percent_matrix = copy.deepcopy(data)
        percent_matrix.M *= 100
        plot_matrix(percent_matrix, ax)
        fig.tight_layout()
        fig.savefig(FIG_PATH / f"{prefix}_probability_matrix.png")

    def expected_counts_matrix():
        n_events = N_DATA
        fig, ax = plt.subplots(figsize=(8, 10))
        ild_tag(ax)
        ax.set_title((f"Expected counts for {n_events:,} Higgs events and SM BRs."), fontsize=14)
        expected_matrix = copy.deepcopy(data)
        expected_matrix.M = n_events * expected_matrix.M * expected_matrix.X0
        plot_matrix(expected_matrix, ax)
        fig.tight_layout()
        fig.savefig(FIG_PATH / f"{prefix}_expected_counts.png")


    probability_matrix()
    expected_counts_matrix()


def main():
    data_dir = Path(__file__).parent / "data"
    for scenario in ["default", "highly_correlated", "overlay_free"]:
        data = load_data(data_str=str(data_dir / scenario))
        probability_and_expected(data, scenario)


if __name__ == "__main__":
    main()
