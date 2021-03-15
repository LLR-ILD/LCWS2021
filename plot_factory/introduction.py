import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from fitting_util.ild import ild_tag
from load_data import load_data
from paths import FIG_PATH, N_DATA


def intro_sample_construction(data):
    expected_matrix = copy.deepcopy(data)
    expected_matrix.M = N_DATA * expected_matrix.M * expected_matrix.X0
    x_names = data.x_names
    y_names = data.y_names

    def sample_counts():
        fig, ax = plt.subplots(figsize=(4, 4))
        ild_tag(ax)
        left = [0]
        for i, br_count in enumerate(expected_matrix.M.sum(axis=0)):
            ax.barh(np.arange(len(left)), br_count, left=left, label=x_names[i])
            left += br_count
        ax.legend(title="(truth) Higgs decay mode", loc="center right")
        ax.set_xlabel("expected signal counts")
        plt.yticks([0], ["in sample"])
        fig.tight_layout()
        fig.savefig(FIG_PATH / "intro_sample_counts.png")

    def category_counts():
        fig_expected, ax_expected = plt.subplots(figsize=(4, 4))
        ild_tag(ax_expected)
        left = np.zeros(len(y_names))
        for i, br_count in enumerate(expected_matrix.M.T):
            ax_expected.barh(np.arange(len(left)), br_count,
                             left=left, label=x_names[i])
            left += br_count
        ax_expected.legend(title="(truth) Higgs decay mode", loc="center right")
        ax_expected.set_xlabel("expected signal counts")
        plt.yticks(np.arange(len(left)), y_names)
        fig_expected.tight_layout()
        fig_expected.savefig(FIG_PATH / "intro_category_counts.png")

    def signal_composition_per_category():
        fig, ax = plt.subplots(figsize=(4, 4))
        ild_tag(ax)
        left = np.zeros(len(y_names))

        normed_categories = expected_matrix.M.T / expected_matrix.M.sum(axis=1)
        for i, br_count in enumerate(normed_categories):
            ax.barh(np.arange(len(left)), br_count, left=left, label=x_names[i])
            left += br_count
        ax.legend(title="(truth) Higgs decay mode", loc="center right")
        ax.set_xlabel("expected signal composition")
        plt.yticks(np.arange(len(left)), y_names)
        fig.tight_layout()
        fig.savefig(FIG_PATH / "intro_signal_composition_per_category.png")

    sample_counts()
    category_counts()
    signal_composition_per_category()


def main():
    data = load_data(data_str=str(Path(__file__).parent / "data/overlay_free_higher_stats"))
    intro_sample_construction(data)


if __name__ == "__main__":
    main()