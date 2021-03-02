import matplotlib.pyplot as plt
from pathlib import Path

import sys

FIG_PATH = Path(__file__).parent.parent / "img/plot_factory"
N_DATA = 40_000

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.facecolor"] = "None"

def get_ev():
    HIGGS_BR_CLASSES = Path().home() / "iLCSoft/projects/Higgs-BR-classes"
    version = "v05"
    njets_file = "simple_event_vector_njets_997k.root"

    if str(HIGGS_BR_CLASSES) not in sys.path:
        sys.path.append(str(HIGGS_BR_CLASSES))
    import helper
    ev = helper.get_data(version=version, file_name=njets_file)
    return ev

higgs_decay_id = {
        "H→ss": 3, "H→cc": 4, "H→bb": 5, "H→μμ": 13, "H→ττ": 15,
        "H→Zγ": 20, "H→gg": 21, "H→γγ": 22, "H→ZZ*": 23, "H→WW": 24,
    }