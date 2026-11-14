"""Section 4.6, keeping the gain explainable: the KL anchor (tab:ablation, fig_anchor_ablation) and the models
distilled from brl RL+FSP's games (tab:distill_strength, fig_distill_explanations)."""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from evaluation.common import RUNS
from evaluation.report.rq1_cost import heatmap_body, heatmap_cells
from evaluation.report.style import MODE_COLOR, OI, RANDOM, REPO, STATED, cluster_ci, load, pct, plt, save

MODES = ("independent", "joint", "sequential")   # the order of tab:ablation
MODE_HEAD = {"independent": "Independent (true bottleneck)", "joint": "Joint", "sequential": "Sequential (true bottleneck)"}


def ablation_table(ev: Path) -> list:
    """Each PPO run of tab:models at the checkpoint the convergence rule stopped it, below its supervised start
    (evaluation/anchor_ablation.py names them "SL <mode> (start)" and after the run)."""
    ab = load(ev / "ablation/ablation.json")
    if not ab:
        return []
    is_run = lambda name: "(start)" not in name  # noqa: E731
    mode_of = lambda r: r.get("meta", {}).get("mode", "joint")  # noqa: E731
    order = sorted(ab["runs"].items(), key=lambda kv: (MODES.index(mode_of(kv[1])), is_run(kv[0]),
                                                       -(kv[1].get("meta", {}).get("anchor") or 0)))
    main_lr = {mode_of(r): r["meta"]["lr"] for n, r in ab["runs"].items() if is_run(n)}
    rows, last_mode = [], None
    for name, run in order:
        p = run["points"][-1]
        mode, anchor = mode_of(run), run.get("meta", {}).get("anchor")
        if mode != last_mode:
            e = int(math.floor(math.log10(main_lr[mode])))
            m = main_lr[mode] / 10 ** e
            lr_txt = f"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else f"${m:g} \\times 10^{{{e}}}$"
            rows.append(("\\addlinespace\n" if last_mode else "")
                        + f"\\multicolumn{{9}}{{l}}{{\\textit{{{MODE_HEAD[mode]}}}, learning rate {lr_txt}}} \\\\")
            last_mode = mode
        if is_run(name):
            lab = "\\textbf{\\boldmath " + ("no anchor" if not anchor else f"$\\eta = {anchor:g}$") + "}"
        else:
            lab = "supervised start"
        kl = p.get("kl_to_sl")
        kl_txt = "--" if kl is None else (f"{kl:.2f}" if kl >= 0.1 else f"{kl:.4f}")
        rows.append(f"\\hangindent=1em\\hspace*{{1em}}{lab} & {p['update']:,} & {p['vs_own_sl']['imps_per_board']:+.2f} & "
                    f"{p['vs_brl_sl']['imps_per_board']:+.2f} & {kl_txt} & "
                    f"{pct(p['wbridge5_agreement']['value'], 1)} & {pct(p['deviation_rate_non_pass']['value'], 2)} & "
                    f"{pct(p['cards']['wbridge5_card_described_supported'], 1) if p.get('cards') else '--'} & "
                    f"{p['concept_f1_selfplay']:.3f} \\\\")

    def half_width(key):   # the range of the runs' 95% half-widths
        ws = [(r["points"][-1][key]["ci95"][1] - r["points"][-1][key]["ci95"][0]) / 2
              for n, r in ab["runs"].items() if is_run(n)]
        lo, hi = min(ws), max(ws)
        return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}--{hi:.2f}"

    return ["% Table: anchor ablation\n\\begin{table}[htbp]\\centering\\footnotesize\n"
            "\\caption[The anchor ablation runs]{The anchor ablation: every PPO run where the convergence rule "
            "stopped it (Section~\\ref{sec:ppo_convergence}), the six PPO models of Table~\\ref{tab:models}, all "
            "trained with four-seat self-play (\\texttt{evaluation/anchor\\_ablation.py}). $\\eta$: weight of "
            "the KL anchor. \\emph{Strength}: IMPs per board in 10{,}000 duplicate boards with random vulnerability against the run's own "
            "supervised start and against brl's supervised network (95\\% intervals $\\pm$" + half_width("vs_own_sl")
            + " and $\\pm$" + half_width("vs_brl_sl") + "). \\emph{Drift}: KL divergence from the supervised policy on the run's own games. Agree: "
            "the share of the 110{,}497 test decisions where it makes WBridge5's bid. Breaks: the share of its "
            "non-pass bids in 2{,}000 self-play deals whose hand does not meet what the bid requires in WBridge5's "
            "system (Section~\\ref{sec:exp_drift}, WBridge5 itself: 0.43\\%). Card: the share of those bids that "
            "WBridge5's convention card describes (WBridge5 itself: 95.5\\%). $F_1$: concept macro-$F_1$ on the "
            "run's own self-play bids.}\n\\label{tab:ablation}\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.3cm}rrrrrrrr}\\toprule\n"
            " & & \\multicolumn{2}{c}{Strength} & \\multicolumn{4}{c}{Drift from WBridge5} & Concepts \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-9}\n"
            " & & vs own & vs brl & & Agree & Breaks & Card & \\\\\n"
            "Run & Updates & SL & SL & KL & (\\%) & (\\%) & (\\%) & $F_1$ \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"]


def fig_anchor_ablation(ev: Path, out: Path):
    """(a) each run's path, gain over its supervised start against its KL from it; (b) at the stopping checkpoint,
    strength against brl SL and the share of system-breaking bids, beside brl's RL networks. The colour is the
    mode, the fill the anchor."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ab = load(ev / "ablation/ablation.json")
    if not ab:
        return
    d = load(ev / "drift/drift.json") or {}
    st = load(ev / "strength/strength.json") or {}
    runs = {n: r for n, r in ab["runs"].items() if "(start)" not in n}
    # close to the width it is printed at (the text block, 5.8 in), so the small type stays 7 pt or more
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.1, 3.5), gridspec_kw={"width_ratios": (1.1, 1)})
    stops, starts = [], []
    for name, run in runs.items():
        mode, eta = run["meta"]["mode"], run["meta"]["anchor"] or 0.0
        col = MODE_COLOR[mode]
        mfc = "white" if not eta else col
        pts = [p for p in run["points"] if "kl_to_sl" in p]
        kl = [p["kl_to_sl"] for p in pts]
        g = [p["vs_own_sl"]["imps_per_board"] for p in pts]
        a1.plot(kl, g, ls="-", marker="o", ms=3, lw=1.0, color=col, mfc=mfc, alpha=0.9)
        lo, hi = pts[-1]["vs_own_sl"]["ci95"]
        a1.errorbar(kl[-1], g[-1], yerr=[[g[-1] - lo], [hi - g[-1]]], color=col, capsize=2, lw=0.8)
        last = run["points"][-1]
        x, y = 100 * last["deviation_rate_non_pass"]["value"], last["vs_brl_sl"]["imps_per_board"]
        lo, hi = last["vs_brl_sl"]["ci95"]
        a2.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=col, mfc=mfc, ms=4.2, capsize=2, lw=0.8,
                    ls="none")
        stops.append((x, y, col, mfc))
    a1.set_xscale("log")
    a1.axhline(0, color="0.6", lw=0.7)
    a1.set_xlabel("KL from the supervised policy, on its own play")
    a1.set_ylabel("IMPs/board against its supervised start")
    a1.set_title("(a) Along each run", fontsize=9)
    h = [Line2D([], [], marker="o", ls="none", color=MODE_COLOR[m], ms=4, label=m)
         for m in ("joint", "independent", "sequential")]
    h += [Line2D([], [], marker="o", ls="none", color="0.3", ms=4, label=f"$\\eta$ = {RUNS['eta']} (filled)"),
          Line2D([], [], marker="o", ls="none", color="0.3", mfc="white", ms=4, label="no anchor (open)"),
          Line2D([], [], marker="x", ls="none", color="0.3", ms=5, label="supervised start, in (b)"),
          Line2D([], [], marker="*", ls="none", color=OI[7], ms=7, label="brl's RL networks, in (b)")]
    a1.legend(handles=h, loc="upper left", fontsize=7.5, frameon=False, handletextpad=0.3, labelspacing=0.3)
    for mode in ("joint", "independent", "sequential"):
        s0 = ab["runs"].get(f"SL {mode} (start)")
        if s0:
            p = s0["points"][0]
            a2.plot(100 * p["deviation_rate_non_pass"]["value"], p["vs_brl_sl"]["imps_per_board"], marker="x", ms=5,
                    color=MODE_COLOR[mode], ls="none")
            starts.append((100 * p["deviation_rate_non_pass"]["value"], p["vs_brl_sl"]["imps_per_board"],
                           MODE_COLOR[mode]))
    vs = {m["A"]: m for m in st.get("matches", []) if m["B"] == "brl SL" and m["vulnerability"] == "random"}
    for a in ("brl RL", "brl RL+PFSP", "brl RL+FSP"):
        if a in vs and a in d:
            xx, yy = 100 * d[a]["deviation_rate_non_pass"]["value"], vs[a]["imps_per_board"]
            a2.plot(xx, yy, "*", color=OI[7], ms=9)
            a2.annotate(a, (xx, yy), xytext=(-6, 4) if a != "brl RL" else (7, -3), textcoords="offset points",
                        fontsize=7.5, ha="right" if a != "brl RL" else "left")
    if "WBridge5" in d:
        w = 100 * d["WBridge5"]["deviation_rate_non_pass"]["value"]
        a2.axvline(w, color="0.6", ls="--", lw=0.8)
        a2.text(w * 1.06, 1.42, "WBridge5's\nown bids", fontsize=7.5, color="0.4", va="top")
    a2.set_xscale("log")
    a2.set_xlim(0.3, 30)
    a2.set_ylim(-0.05, 1.5)
    a2.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a2.set_xlabel("non-pass bids breaking\nWBridge5's system (%)")
    a2.set_ylabel("IMPs/board against brl SL")
    a2.set_title("(b) Where each run stops", fontsize=9)
    near = [(x, y, c, f) for x, y, c, f in stops if x < 1.5]
    if near:   # the runs that stay at WBridge5's floor, enlarged
        ax = a2.inset_axes([0.47, 0.07, 0.5, 0.5])
        for x, y, c, f in near:
            ax.plot(x, y, "o", color=c, mfc=f, ms=4.2)
        for x, y, c in starts:
            ax.plot(x, y, "x", color=c, ms=5)
        xs = [v[0] for v in near] + [v[0] for v in starts]
        ys = [v[1] for v in near] + [v[1] for v in starts]
        ax.set_xlim(min(xs) - 0.03, max(xs) + 0.03)
        ax.set_ylim(min(ys) - 0.06, max(ys) + 0.06)
        ax.tick_params(labelsize=7, length=2)
        ax.set_title("the runs near 0.5%, enlarged", fontsize=7.5, pad=2)
        a2.indicate_inset_zoom(ax, edgecolor="0.5", lw=0.6)
    fig.set_layout_engine("tight")
    save(fig, out, "fig_anchor_ablation")


def distill_strength_table(ev: Path) -> list:
    """The distilled models' duplicate matches (evaluation.strength_matches --distill): each distilled model (rows)
    against the models the thesis compares it with and against the other distilled models (columns), all with random
    vulnerability. Built like tab:strength; the reference models' matches with each other are in tab:strength only."""
    sd = load(ev / "strength/strength_distill.json")
    if not sd or not sd.get("matches"):
        return []
    from evaluation.strength_matches import DISTILL_REFERENCES, DISTILLED
    cell = heatmap_cells(sd["matches"], lambda a, b: "random")
    rows = [n for n in DISTILLED if any(n == k[0] for k in cell)]
    cols = [n for n in DISTILL_REFERENCES if any(n == k[1] for k in cell)] + rows
    head, body = heatmap_body(cell, rows, cols)
    return ["% Table: head-to-head strength of the distilled models (heatmap)\n\\begin{table}[htbp]\\centering\\scriptsize\n"
            "\\caption[Duplicate matches of the distilled models]{Duplicate matches of the distilled models (rows) "
            "against the models they are compared with and against each other, on the 10{,}000 held-out deals of "
            "Table~\\ref{tab:strength}, all with random vulnerability. Dist.: trained on brl RL+FSP's games. Each cell: IMPs per "
            "board for the row model against the column model, shaded blue where the row model is stronger and red where "
            "it is weaker, and bold where the 95\\% CI excludes zero. Small grey number: share (\\%) of the WBridge5 test "
            "positions where the two make the same bid.}\n\\label{tab:distill_strength}\n"
            "\\setlength{\\tabcolsep}{2pt}\\renewcommand{\\arraystretch}{1.25}\n"
            "\\begin{adjustbox}{max width=\\linewidth}\n"
            f"\\begin{{tabular}}{{l*{{{len(cols)}}}{{>{{\\centering\\arraybackslash}}p{{2.9em}}}}}}\\toprule\n"
            f"Model & {head} \\\\\\midrule\n{body}\n\\bottomrule\\end{{tabular}}\n\\end{{adjustbox}}\\end{{table}}\n"]


DISTILL_DIR = REPO / "runs/distill"   # scripts/distill_rl_fsp.py --out, as pipeline/evaluate.sh runs it


def distill_explanation_sets(ev: Path):
    """(group, name, records of its explained decisions in its own self-play), in the order of the figure."""
    sets = []
    for m in ("joint", "independent", "sequential"):
        p = ev / f"explanations/SL_{m}__selfplay_records.pkl"
        if p.exists():
            sets.append(("trained on WBridge5", f"SL {m}", p))
    for m in ("joint", "independent", "sequential"):
        p = next((DISTILL_DIR / f"explain/faithfulness_{m}").glob("*_records.pkl"), None)
        if p:
            sets.append(("distilled from RL+FSP", f"dist. {m}", p))
    return sets


def fig_distill_explanations(ev: Path, out: Path):
    """The share of the bid's margin the stated concepts carry and how often removing them changes the bid, for the
    SL models and the distilled ones, against as many random true concepts (non-pass bids, own self-play)."""
    sets = distill_explanation_sets(ev)
    if not sets:
        return
    stats = []
    for g, name, p in sets:
        u = [r for r in pickle.load(open(p, "rb"))["uniform"] if r["bid"] != "P" and "stated_share" in r]
        deal = [r["deal"] for r in u]
        row = {"group": g, "name": name}
        for key, f, src in (("share", np.median, "stated_share"), ("rshare", np.nanmedian, "random_stated_share"),
                            ("flip", np.mean, "stated_flip"), ("rflip", np.nanmean, "random_stated_flip")):
            vals = [float(r.get(src, np.nan)) for r in u]
            row[key] = cluster_ci(vals, deal, f)
        stats.append(row)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    x, prev = [], None
    for s in stats:
        x.append((x[-1] + 1 if x else 0) + (0.7 if prev is not None and s["group"] != prev else 0))
        prev = s["group"]
    x = np.array(x)
    for ax, (k, rk, title, ylab) in zip(axes, (("share", "rshare", "(a) Share of the bid's margin they carry",
                                               "median share of the margin (%)"),
                                              ("flip", "rflip", "(b) Removing them changes the bid",
                                               "bids that change (%)"))):
        for off, key, col, lab in ((-0.19, k, STATED, "the concepts the explanation states"),
                                   (0.19, rk, RANDOM, "as many random true concepts")):
            v = np.array([100 * s[key][0] for s in stats])
            lo = np.array([100 * s[key][1] for s in stats])
            hi = np.array([100 * s[key][2] for s in stats])
            ax.bar(x + off, v, width=0.38, color=col, label=lab)
            ax.errorbar(x + off, v, yerr=[v - lo, hi - v], fmt="none", ecolor="0.25", lw=0.7, capsize=1.5)
        ax.set_xticks(x, [s["name"].replace("independent", "ind.").replace("sequential", "seq.") for s in stats],
                      fontsize=6.5, rotation=30, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8.5)
        groups = []
        for s, xi in zip(stats, x):
            if not groups or groups[-1][0] != s["group"]:
                groups.append([s["group"], xi, xi])
            groups[-1][2] = xi
        for g, a, b in groups:
            ax.text((a + b) / 2, 99, g.replace(" from ", " from\n").replace(" on ", " on\n"), ha="center", va="top",
                    fontsize=6.6, color="0.3")
    fig.set_layout_engine("tight", rect=(0, 0.07, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.8, frameon=False)
    save(fig, out, "fig_distill_explanations")


FIGURES = [fig_anchor_ablation, fig_distill_explanations]
TABLES = [ablation_table, distill_strength_table]
