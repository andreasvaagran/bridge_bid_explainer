"""Section 4.3, are the explanations faithful: the bottleneck diagnostics and interventions, completeness, probes
against erasure, and how the one-card counterfactuals line up with the rules."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from evaluation.report.style import MAIN, MODE_COLOR, OI, RANDOM, STATED, load, plt, save


def fig_bottleneck(ev: Path, out: Path):
    """fig_bottleneck_diagnostics (agreement with WBridge5 when the concept path is changed) and fig_interventions
    (concepts replaced by their rule-engine value)."""
    rows = []
    for mode in ("joint", "independent", "sequential"):
        r = load(ev / f"influence/SL_{mode}__test.json")
        if r:
            rows.append((mode, r))
    if not rows:
        return
    keys = [("soft", "as trained"), ("hard", "concepts thresholded"), ("oracle", "rule-engine concepts"),
            ("strain_masked", "suit descriptors zeroed"), ("concepts_zeroed", "all concepts zeroed"),
            ("hidden_zeroed", "hidden path zeroed")]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    w = 0.13
    for j, (k, lab) in enumerate(keys):
        vals = [r["diagnostics"].get(k, {}).get("accuracy_vs_wbridge5", {}).get("value", np.nan) for _, r in rows]
        ax.bar(np.arange(len(rows)) + (j - 2.5) * w, vals, width=w, label=lab, color=OI[j])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([m for m, _ in rows])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("agreement with WBridge5's bid")
    ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "fig_bottleneck_diagnostics")

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for mode, r in rows:
        iv = r["interventions"]
        xs = [int(k) for k in iv["uncertain"]]
        ax.plot(xs, [iv["uncertain"][str(x)] for x in xs], "-o", ms=2.5, color=MODE_COLOR[mode], label=mode)
        ax.plot(xs, [iv["random"][str(x)] for x in xs], "--", color=MODE_COLOR[mode], alpha=0.7)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("concepts replaced by their\nrule-engine value")
    ax.set_ylabel("agreement with WBridge5")
    order = [plt.Line2D([], [], color="0.3", marker="o", ms=2.5, label="most uncertain first"),
             plt.Line2D([], [], color="0.3", ls="--", label="random")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + order, fontsize=7.5, loc="upper left", frameon=False,
              handlelength=1.6, labelspacing=0.25)
    save(fig, out, "fig_interventions")


def fig_completeness(ev: Path, out: Path):
    """How much of each model's bidding a small network recovers from its concept scores (concept_completeness.py)."""
    r = load(ev / "completeness/completeness.json")
    if not r:
        return
    models = [m for m in MAIN + ["SL plain", "brl SL", "brl RL", "brl RL+PFSP", "brl RL+FSP"] if m in r["models"]
              and "sources" in r["models"][m]]
    fig, ax = plt.subplots(figsize=(6.6, 0.36 * len(models) + 1.0))
    y = np.arange(len(models))[::-1]
    w = 0.26
    for j, (v, col, name) in enumerate((("all concepts", OI[4], "all 146 concepts"),
                                         ("no call identity", OI[1], "without the 14 that name the bid"),
                                         ("call identity only", OI[0], "only the 14 that name the bid"))):
        vals = [r["models"][m]["sources"]["model"]["variants"][v]["fidelity"]["g_accuracy"]["value"] for m in models]
        ax.barh(y + (1 - j) * w, vals, height=w, color=col, label=name)
    ctl = [r["models"][m]["sources"].get("random-init control", {}).get("variants", {}).get("all concepts", {})
           .get("fidelity", {}).get("g_accuracy", {}).get("value", np.nan) for m in models]
    ax.scatter(ctl, y, marker="x", color="k", s=18, zorder=3, label="untrained network (control)")
    # each model's own baseline (it differs: the RL networks pass less often than the supervised ones)
    a_r = [r["models"][m]["sources"]["model"]["variants"]["all concepts"]["fidelity"]["baseline_accuracy"] for m in models]
    ax.scatter(a_r, y, marker="|", color="0.35", s=140, lw=1.2, zorder=3, label="most frequent legal bid")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("fidelity")
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    sep = len([m for m in models if m in MAIN]) - 0.5
    ax.axhline(y[0] - sep, color="0.7", lw=0.8)
    save(fig, out, "fig_completeness")


ERASE_SHOWN = ["brl SL", "brl RL", "brl RL+PFSP", "brl RL+FSP", "SL plain", "SL joint"]


def fig_probes_erasure(ev: Path, out: Path):
    """(a) what a linear probe reads from each layer; (b) how many bids change when LEACE erases it."""
    r = load(ev / "probes/probes.json")
    er = load(ev / "erasure/erasure.json")
    if not r or not er:
        return
    extra = load(ev / "probes/probes_plain.json")  # the matched plain network and the joint seed replicates
    if extra:
        r["probes"].update(extra["probes"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": (1.15, 1)})
    styles = {"brl SL": (OI[7], "o"), "brl RL": (OI[6], "s"), "brl RL+PFSP": (OI[5], "D"), "brl RL+FSP": (OI[1], "^"),
              "random init": ("0.6", "x")}
    layers = ["input", "layer 1", "layer 2", "layer 3", "layer 4"]
    for name, (col, mk) in styles.items():
        p = r["probes"].get(name)
        if not p:
            continue
        ls = [k for k in layers if k in p]
        key = "own_call" if "own_call" in p[ls[0]] else "wbridge5_call"
        a1.plot([layers.index(k) for k in ls], [p[k][key]["macro_f1"] for k in ls], marker=mk, ms=4, lw=1.1, color=col,
                label="untrained network" if name == "random init" else name)
    for name, col in (("SL joint (trunk)", MODE_COLOR["joint"]), ("SL plain (trunk)", "0.35")):
        p = r["probes"].get(name)
        if p and "layer 1" in p:
            a1.scatter([4], [p["layer 1"]["own_call"]["macro_f1"]], marker="*", s=70, color=col, zorder=3,
                       label=f"{name.split(' (')[0]}: last hidden layer")
    a1.set_xticks(range(5), layers)
    a1.set_ylim(0, 1)
    a1.set_ylabel("macro-$F_1$ of a linear probe (held out)")
    a1.set_title("(a) Represented: probe $F_1$ by layer", fontsize=8.5)
    a1.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    models = [m for m in ERASE_SHOWN if m in er["models"]]
    y = np.arange(len(models))[::-1]
    erased, rand = [], []
    for m in models:
        L = er["models"][m]["layers"]
        s = L[max(L, key=int)]["sets"]["no call identity"]
        erased.append(100 * s["erased"]["flip"])
        rand.append(100 * s["random"]["flip"])
    a2.barh(y + 0.17, erased, height=0.34, color=STATED, label="the 132 descriptive concepts erased")
    a2.barh(y - 0.17, rand, height=0.34, color=RANDOM, label="a random subspace of the same size")
    for yi, v, w in zip(y, erased, rand):
        a2.text(v + 1, yi + 0.17, f"{v:.0f}", va="center", fontsize=6.6)
        a2.text(w + 1, yi - 0.17, f"{w:.0f}", va="center", fontsize=6.6, color="0.4")
    a2.set_yticks(y, models, fontsize=7)
    a2.set_xlim(0, 70)
    a2.set_xlabel("own bids that change (%)")
    a2.set_title("(b) Used: erasing them", fontsize=8.5)
    a2.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=1, frameon=False)
    fig.set_layout_engine("tight")
    save(fig, out, "fig_probes_erasure")


CF_MODELS = ["SL joint", "SL independent", "SL sequential", "PPO independent", "SL plain", "brl SL", "brl RL",
             "brl RL+PFSP", "brl RL+FSP"]
CF_LINES = {"SL independent": MODE_COLOR["independent"], "SL joint": MODE_COLOR["joint"], "SL plain": "0.35",
            "brl SL": OI[7], "brl RL+FSP": OI[5]}


def _cf_tag(m):
    return m.replace(" ", "_").replace("+", "p")


def fig_counterfactual_alignment(ev: Path, out: Path):
    """(a) decisions with a one-card counterfactual, by confidence; (b) whether random one-card changes that alter
    the rules' description of the bid change the bid more often than those that do not."""
    rs = [(m, load(ev / f"counterfactuals/{_cf_tag(m)}__test.json")) for m in CF_MODELS]
    rs = [(m, r) for m, r in rs if r]
    if not rs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": (1, 1.4)})
    edges = [0.0, 0.8, 0.95, 0.99, 0.999, 1.0001]
    labels = ["<0.8", "0.8–\n0.95", "0.95–\n0.99", "0.99–\n0.999", "≥0.999"]
    for m, col in CF_LINES.items():
        r = dict(rs).get(m)
        p = ev / f"counterfactuals/{_cf_tag(m)}__test_records.pkl"
        if not r or not p.exists():
            continue
        rec = pickle.load(open(p, "rb"))
        conf = np.array([x["conf"] for x in rec])
        has = np.array([x["n_cf"] > 0 for x in rec])
        b = np.digitize(conf, edges) - 1
        cov = [100 * has[b == k].mean() if (b == k).sum() >= 10 else np.nan for k in range(len(labels))]
        a1.plot(range(len(labels)), cov, "-o", ms=3, lw=1.1, color=col,
                label=f"{m} ({100 * r['coverage']['value']:.1f}% overall)")
    a1.set_xticks(range(len(labels)), labels, fontsize=7)
    a1.set_xlabel("the model's probability for its bid")
    a1.set_ylabel("decisions with a one-card counterfactual (%)")
    a1.set_ylim(40, 102)
    a1.set_title("(a) Coverage, by confidence", fontsize=8.5)
    a1.legend(fontsize=6.6, loc="lower left")
    x = np.arange(len(rs))
    ra = [r["rule_alignment"] for _, r in rs]
    a2.bar(x - 0.2, [100 * a["p_flip_given_description_changed"] for a in ra], 0.4, color=OI[4],
           label="the change alters the rules' description")
    a2.bar(x + 0.2, [100 * a["p_flip_given_description_unchanged"] for a in ra], 0.4, color=RANDOM,
           label="it leaves the description as it was")
    abbr = {"independent": "ind.", "sequential": "seq."}
    a2.set_xticks(x, [" ".join(abbr.get(w, w) for w in m.split(" ")).replace("brl RL+", "RL+") for m, _ in rs],
                  fontsize=6.8, rotation=35, ha="right")
    for xi, a in zip(x, ra):
        a2.text(xi, 100 * a["p_flip_given_description_changed"] + 1.5, f"{a['cohen_kappa']:.2f}", ha="center",
                fontsize=6.3, color="0.3")
    a2.set_ylabel("random one-card changes\nthat change the bid (%)")
    a2.set_ylim(0, 100)
    a2.set_title("(b) Rule alignment (Cohen's κ over the bars)", fontsize=8.5)
    a2.legend(fontsize=6.6, loc="upper right", frameon=False)
    fig.set_layout_engine("tight")
    save(fig, out, "fig_counterfactual_alignment")


FIGURES = [fig_bottleneck, fig_completeness, fig_probes_erasure, fig_counterfactual_alignment]
TABLES = []
