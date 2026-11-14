"""Section 4.5, how far reinforcement learning moves the convention: each model's bids that break WBridge5's
system, and what partner does when the two one-level minor openings are swapped. The convention cards of this
section are drawn by evaluation/convention_card_figures.py and convention_card_ebu_form.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.report.style import OI, REPO, load, mode_color, plt, save


def fig_strength_vs_drift(ev: Path, out: Path):
    """How far each model of tab:strength has moved from WBridge5's system: its non-pass bids that break the rules, on
    WBridge5's mix of situations, split into large breaks and near misses (convention_drift.near_miss) and sorted by
    the large ones: the share alone ranked a network that opens a little light with one that plays another system."""
    d = load(ev / "drift/drift.json")
    if not d:
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory
    from evaluation.strength_matches import HEATMAP_MODELS
    key, big = "deviation_rate_standardized", "large_deviation_rate_standardized"
    ms = sorted((a for a in HEATMAP_MODELS if a in d), key=lambda a: (d[a][big]["value"], d[a][key]["value"]))
    # with its labels the figure is as wide as the text block, where it is printed, so the type keeps its size
    fig, ax = plt.subplots(figsize=(4.3, 0.22 * len(ms) + 1.1))
    cols = blended_transform_factory(ax.transAxes, ax.transData)   # x in axes units, y in rows
    for i, a in enumerate(ms):
        tot, lg = d[a][key], d[a][big]
        col = OI[5] if a.startswith("brl RL") else OI[8] if a.startswith("brl") else mode_color(a)
        open_bar = "unanchored" in a
        ax.barh(i, 100 * lg["value"], color="white" if open_bar else col, edgecolor=col, lw=1.2, height=0.65,
                hatch="////" if open_bar else None)
        ax.barh(i, 100 * (tot["value"] - lg["value"]), left=100 * lg["value"], color=col, alpha=0.22,
                edgecolor="none", height=0.65)
        ax.errorbar(100 * lg["value"], i, xerr=[[100 * (lg["value"] - lg["ci95"][0])],
                                                [100 * (lg["ci95"][1] - lg["value"])]],
                    fmt="none", ecolor="0.25", capsize=2, lw=0.8)
        # the numbers in two labelled columns right of the plot, not as an unexplained "a (b)" by each bar
        ax.text(1.08, i, f"{100 * lg['value']:.1f}", transform=cols, ha="right", va="center", fontsize=8)
        ax.text(1.20, i, f"{100 * tot['value']:.1f}", transform=cols, ha="right", va="center", fontsize=8)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms, fontsize=8)
    w = d["WBridge5"][key]["value"]
    ax.axvline(100 * w, color="0.5", ls="--", lw=0.8)
    for x, t in ((1.08, "large"), (1.20, "all")):
        ax.text(x, len(ms) - 0.2, t, transform=cols, ha="right", va="bottom", fontsize=8, fontweight="bold")
    ax.legend([Patch(facecolor="0.35"), Patch(facecolor="0.35", alpha=0.22),   # neutral swatches, not a model's colour
               Patch(facecolor="white", edgecolor="0.35", hatch="////"), Line2D([], [], color="0.5", ls="--", lw=0.8)],
              ["large break (95% CI)", "small break (within 2 HCP of the system)",
               "PPO without the anchor", "WBridge5's own bids (false alarms)"],
              fontsize=7.5, loc="lower right", frameon=False)
    ax.set_xlabel("non-pass bids that break WBridge5's system (%),\nweighted to WBridge5's mix of auctions", fontsize=8.5)
    ax.set_ylim(-0.6, len(ms) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig_strength_vs_drift")


# the listener test of the distillation (evaluation.bid_meaning_tests swap)
LISTENER_FILE = REPO / "runs/evaluation/bid_meaning/listener_fsp.json"
LISTENERS = [("brl RL+FSP", "RL+FSP", OI[5]), ("distilled independent", "distilled ind.", OI[6]),
             ("distilled joint", "distilled joint", OI[0]), ("brl SL", "brl SL", OI[7]),
             ("SL independent", "SL ind.", OI[4])]
REPLY = [("pass", "pass", "0.15"), ("bids_diamonds", "bids diamonds", OI[0]), ("bids_clubs", "bids clubs", OI[2]),
         ("other", "another bid", "0.82")]


def fig_message_swap(ev: Path, out: Path):
    """(a) partner's replies that change when 1C and 1D are swapped in the observation; (b) what partner does with a
    weak hand after each."""
    d = load(LISTENER_FILE)
    if not d:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 6.4), gridspec_kw={"height_ratios": (1, 1.5)})   # stacked: (b)'s row labels are long
    ls = [(k, lab, col) for k, lab, col in LISTENERS if k in d["_summary"]]
    x = np.arange(len(ls))
    for j, (op, hatch) in enumerate((("1C", None), ("1D", "///"))):
        v = [100 * d["_summary"][k][op]["changes_when_swapped"] for k, _, _ in ls]
        a1.bar(x + (j - 0.5) * 0.38, v, width=0.38, color=[c for _, _, c in ls], hatch=hatch, edgecolor="white",
               lw=0.4, label=f"real {'1♣' if op == '1C' else '1♦'}, seen as {'1♦' if op == '1C' else '1♣'}")
    a1.set_xticks(x, [lab.replace("distilled ", "distilled\n") for _, lab, _ in ls], fontsize=7.5)
    a1.set_ylabel("partner's replies that change (%)")
    a1.set_xlabel("partner")
    a1.set_ylim(0, 100)
    a1.set_yticks(range(0, 101, 20))
    a1.set_title("(a) Swapping the two openings", fontsize=8.5, loc="left")
    a1.legend(fontsize=6.6, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=1)
    rows = []
    for k, who in (("brl RL+FSP", "RL+FSP's partner"), ("SL independent", "SL independent's partner")):
        for key, heard_real, heard_swap in (("1C (real) vs 1D (swapped in)", "1♣", "1♦"),
                                            ("1D (real) vs 1C (swapped in)", "1♦", "1♣")):
            e = d[k][key]["hcp<=6"]
            for side, heard in (("real", heard_real), ("swapped", heard_swap)):
                v = e[side]
                share = {c: v.get(c, 0.0) for c, _, _ in REPLY[:3]}
                share["other"] = max(0.0, 1 - sum(share.values()))
                tag = f"{heard_real} opened, sees {heard}" + (" (swapped)" if side == "swapped" else "")
                rows.append((who, tag, share, v["n"]))
    y = np.array([len(rows) - 1 - i + (0.9 if i < 4 else 0) for i in range(len(rows))])
    for yi, (_, tag, share, n) in zip(y, rows):
        left = 0
        for c, lab, col in REPLY:
            a2.barh(yi, 100 * share[c], left=left, height=0.72, color=col, edgecolor="white", lw=0.4,
                    label=lab if yi == y[0] else None)
            if share[c] >= 0.12:
                a2.text(left + 50 * share[c], yi, f"{100 * share[c]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if c == "pass" else "black")
            left += 100 * share[c]
    a2.set_yticks(y, [t for _, t, _, _ in rows], fontsize=6.6)
    for who, top in ((rows[0][0], y[0]), (rows[4][0], y[4])):
        a2.text(-1, top + 0.62, who, ha="right", va="center", fontsize=6.8, fontweight="bold")
    a2.set_xlim(0, 100)
    a2.set_xlabel("partner's reply with at most 6 HCP (%)")
    a2.set_title("(b) What partner does with a weak hand", fontsize=8.5, loc="left", pad=22)
    a2.legend(fontsize=6.6, ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False)
    fig.set_layout_engine("tight")
    save(fig, out, "fig_message_swap")


FIGURES = [fig_strength_vs_drift, fig_message_swap]
TABLES = []
