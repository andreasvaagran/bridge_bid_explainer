"""What every figure and table of the report shares: the colours, matplotlib's settings, saving and loading."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Okabe-Ito (colour-blind safe)
OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#999999"]
CAT_COLOR = {"pass / double": OI[7], "strain": OI[8], "opening": OI[4], "convention": OI[5], "competitive": OI[0],
             "fit / support": OI[2], "length / strain": OI[1], "strength": OI[6], "auction structure": OI[3]}
MODE_COLOR = {"joint": OI[0], "independent": OI[4], "sequential": OI[2]}
STATED, RANDOM, TOP = OI[4], OI[8], OI[2]   # the explanation's concepts, random ones, the top ones by contribution
MAIN = ["SL joint", "SL independent", "SL sequential", "PPO joint", "PPO independent", "PPO sequential",
        "PPO joint unanchored", "PPO independent unanchored", "PPO sequential unanchored"]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def mode_color(name: str) -> str:
    """The colour of a model of this work by its concept mode ("PPO joint unanchored" -> joint); grey for plain."""
    return next((c for m, c in MODE_COLOR.items() if m in name), "0.35")


def short(name: str) -> str:
    return name.replace("bid_", "")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


TEXT_W = 418.25 / 72.27          # the thesis's text block, in inches
MIN_PT = 7.5                      # the smallest type on the printed page
PRINT_WIDTH = {"fig_concept_f1": 0.82, "fig_interventions": 0.55}   # \includegraphics width; the rest \linewidth


def readable(fig, name: str):
    """Raise every text below MIN_PT as printed: the figure is scaled to its width in the thesis, so a 7-inch figure
    set at the text width shrinks its type by a sixth."""
    for _ in range(3):   # larger type widens the tight box, which shrinks the print again
        fig.canvas.draw()   # runs the figure's layout engine, if it has one
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        scale = min(1.0, PRINT_WIDTH.get(name, 1.0) * TEXT_W / bb.width)
        floor = MIN_PT / scale
        small = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text() and t.get_fontsize() < floor - 0.05]
        if not small:
            return
        for t in small:
            t.set_fontsize(floor)


def save(fig, out: Path, name: str):
    readable(fig, name)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def cluster_ci(values, clusters, stat, n_boot: int = 1000, seed: int = 0):
    """95% interval of stat(values) over resampled clusters (deals): the decisions of one deal are not independent."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    ids, inv = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        boots.append(stat(values[idx]))
    return float(stat(values)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))
