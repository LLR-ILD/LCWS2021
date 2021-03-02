import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numexpr
import numpy as np

from paths import FIG_PATH, get_ev, higgs_decay_id


mcs = list(mcolors.TABLEAU_COLORS.values())
h_color = {h: mcs[i%len(mcs)] for i, h in enumerate(higgs_decay_id)}


def create_pie(df, ax):
    per_decay = df.groupby("hDecay")["weight"].sum()
    per_decay = per_decay / per_decay.sum()
    if set(per_decay.index).issubset(set(higgs_decay_id.values())):
        per_decay = per_decay.rename(index={v: k
                                    for k, v in higgs_decay_id.items()})
    try:
        names2numbers = lambda idx: [higgs_decay_id[x] for x in idx]
        per_decay = per_decay.sort_index(key=names2numbers)
    except KeyError:
        per_decay = per_decay.sort_index()
    colors = [h_color[n] for n in per_decay.index]
    patches, txts = ax.pie(per_decay, labels=per_decay.index,
        colors=colors, normalize=True)
    labels = [f"{100*per_decay[n]:6.2f}% {n}" for n in per_decay.index]
    ax.legend(patches, labels, bbox_to_anchor=(1.1, 1),
        title=f"Total count: {df.weight.sum():,.1f}")


def build_up_category_cc(ev):
    cat_cc = ev
    h_dec = 4
    sel_vars_cc = {
        "n_pfos": " > 20",
        "c_tag1": " > 0.5",
        # "n_iso_leptons": " == 0",
        "m_h": " > 100",
        "c_tag2": " > 0.5",
    }
    cut_expr = "(n_iso_leptons == 0)"

    nr = (len(sel_vars_cc) - 1) // 2 + 1
    fig, axs = plt.subplots(ncols=2, nrows=max(2, nr), figsize=(8, 3*nr))
    for i, (col, cut_val) in enumerate(sel_vars_cc.items()):
        ax = axs[i % nr][i // nr]
        if np.issubdtype(cat_cc[col].dtype, np.integer):
            bins = np.arange(cat_cc[col].min() - .5, cat_cc[col].max() + 1)
        else:
            bins = 100
        _, bins, _ = ax.hist(cat_cc[col], bins=bins, weights=cat_cc.weight, label="H→any")
        cat_cc[cat_cc.hDecay == h_dec][col].plot
        ax.hist(    cat_cc[cat_cc.hDecay == h_dec][col], bins=bins,
            weights=cat_cc[cat_cc.hDecay == h_dec]["weight"], label="H→cc")
        ax.axvline(float(cut_val.split(" ")[-1]), color="black", linewidth=2)
        ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_title(cut_expr, wrap=True)
        new_expr = f"({col}{cut_val})"
        cut_expr = new_expr if cut_expr == "" else cut_expr + " & " + new_expr
        cat_cc = cat_cc[numexpr.evaluate(cut_expr, cat_cc)]
    axs[0][0].legend()
    fig.tight_layout()
    fig.savefig(FIG_PATH / "cc_category_build_up.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(cut_expr, wrap=True)
    create_pie(cat_cc, ax)
    fig.tight_layout()
    fig.savefig(FIG_PATH / "cc_category_pie.png")


def main():
    ev = get_ev()
    build_up_category_cc(ev)


if __name__ == "__main__":
    main()