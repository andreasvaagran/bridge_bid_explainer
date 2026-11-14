"""Section 4.2, what a true bottleneck costs: the concept F1 figure, the strength heatmap (tab:strength) and where
the time goes (fig_cost)."""
from __future__ import annotations

from pathlib import Path

from evaluation.common import concept_category
from evaluation.report.style import CAT_COLOR, OI, load, pct, plt, save, short

F1_LABEL = {"bid_is_sacrifice_candidate": "a sacrifice", "bid_is_negative_invite": "a negative invite (2 cases)",
            "bid_is_penalty_double": "a penalty double", "bid_double_with_length_in_their_suit":
            "a double with length\nin their suit", "bid_is_cuebid": "a cuebid",
            "bid_shows_jump_shift_strength": "jump-shift strength"}   # the six weakest, in words


def fig_concept_f1(ev: Path, out: Path):
    r = load(ev / "concepts/concept_quality.json")
    if not r:
        return
    pc = r["SL independent"]["wb5"]["per_concept"]
    fig, ax = plt.subplots(figsize=(5.1, 3.0))   # printed at this width (0.88 of the text block): the type keeps its size
    for n, v in pc.items():
        ax.scatter(v["support"], v["f1"], s=10, color=CAT_COLOR[concept_category(n)], alpha=0.8)
    used = sorted({concept_category(n) for n in pc})
    ax.set_xscale("log")
    # the six worst concepts named, right of their points; a label that would cover one above it moves down, with a
    # leader line (three labels around 100-200 decisions ran into each other)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = []
    dots = {n: ax.transData.transform((max(v["support"], 0.5), v["f1"])) for n, v in pc.items()}   # in pixels
    for n, v in sorted(pc.items(), key=lambda kv: -kv[1]["f1"])[-6:]:
        xy = (v["support"], v["f1"])
        for dy in range(-3, -90, -2):
            t = ax.annotate(F1_LABEL.get(n, short(n).replace("_", " ")), xy, fontsize=7.5, xytext=(4, dy),
                            textcoords="offset points", va="center")
            bb = t.get_window_extent(rend)  # the text alone: a leader line would overlap the label above it
            bb = bb.from_extents(bb.x0 - 8, bb.y0, bb.x1 + 8, bb.y1)   # and a gap beside it: two labels never read as one
            on_dot = any(bb.x0 - 3 <= x <= bb.x1 + 3 and bb.y0 - 3 <= y <= bb.y1 + 3   # a label never covers a point
                         for m, (x, y) in dots.items() if m != n)
            if not on_dot and not any(bb.overlaps(o) for o in placed):
                break
            t.remove()
        placed.append(bb)
        if dy < -3:
            ax.annotate("", xy, xytext=(4, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5", shrinkA=0, shrinkB=2))
    shown = {"strain": "suit descriptor", "length / strain": "length / suit"}  # the thesis says suit, not strain
    ax.legend([plt.Line2D([], [], marker="o", ls="", color=CAT_COLOR[k], ms=4) for k in used], [shown.get(k, k) for k in used],
              fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.13), ncol=3, frameon=False, columnspacing=1.0,
              handletextpad=0.2)   # in the empty lower half of the plot, not beside it
    ax.set_xscale("log")
    ax.set_xlabel("positive test decisions (log)")
    ax.set_ylabel("F1 vs rule engine")
    save(fig, out, "fig_concept_f1")


# ---------------------------------------------------------------- duplicate-match heatmaps (also tab:distill_strength)

def heatmap_cells(matches, vulnerability) -> dict:
    """{(row, column): (IMPs/board for row, CI excludes 0, same-bid share)}; each match fills two cells, the second read
    from B's side. `vulnerability(a, b)` is the protocol's setting for the pair; other settings are left out."""
    cell = {}
    for m in matches:
        if m["vulnerability"] != vulnerability(m["A"], m["B"]):
            continue
        x, lo95, hi95, same = m["imps_per_board"], *m["ci95"], m["call_agreement_on_wbridge5_positions"]
        cell[m["A"], m["B"]] = (x, lo95 > 0 or hi95 < 0, same)
        cell[m["B"], m["A"]] = (-x, lo95 > 0 or hi95 < 0, same)
    return cell


def heatmap_body(cell, rows, cols, row_groups=None) -> tuple:
    """The LaTeX head and body rows of a duplicate-match heatmap: IMPs per board for the row model, shaded blue (row
    stronger) or red (row weaker), bold where the 95% CI excludes zero, with the same-bid share below."""
    abbr = lambda n: (n.replace("independent", "ind.").replace("sequential", "seq.")  # noqa: E731
                      .replace(" unanchored", " unanch.").replace("distilled", "dist."))
    lo, hi = (247, 247, 247), {1: (29, 90, 142), -1: (214, 45, 32)}  # bxc blue / bxred, faded towards white

    def shade(x):
        t = 0.6 * min(abs(x) / 1.5, 1.0)
        c = hi[1 if x >= 0 else -1]
        return ",".join(str(round(l + t * (h - l))) for l, h in zip(lo, c))

    def fmt(a, b):
        if a == b:
            return "\\cellcolor[RGB]{225,225,225}"
        if (a, b) not in cell:
            return "--"
        x, sig, same = cell[a, b]
        v = f"{x:+.2f}".replace("-", "$-$")
        v = f"\\textbf{{{v}}}" if sig else v
        return (f"\\cellcolor[RGB]{{{shade(x)}}}\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{v}\\\\"
                f"[-0.4ex]{{\\tiny\\color{{black!75}}{pct(same)}}}\\end{{tabular}}")

    head = " & ".join("\\rotatebox{70}{" + abbr(c) + "}" for c in cols)
    ends = {sum(row_groups[:k]) for k in range(1, len(row_groups or []))}   # a gap after the last row of each group
    body = "\n".join(abbr(a) + " & " + " & ".join(fmt(a, b) for b in cols) + " \\\\"
                     + ("\\addlinespace[3pt]" if i + 1 in ends else "") for i, a in enumerate(rows))
    return head, body


def model_group(name: str) -> int:
    """The block of tab:strength a model belongs to: brl SL, this thesis's supervised models, PPO with the anchor,
    PPO without it, brl's RL networks. The blocks are set apart by white gaps, so the 16 x 16 cells read as 5 x 5."""
    if name.startswith("brl"):
        return 0 if name == "brl SL" else 4
    return 1 if name.startswith("SL") else 3 if "unanchored" in name else 2


def strength_table(ev: Path) -> list:
    """Every model against every other under the thesis's protocol (evaluation.strength_matches --all_pairs: supervised
    pairs non-vulnerable, the rest random vulnerability)."""
    st = load(ev / "strength/strength.json")
    if not st:
        return []
    from evaluation.strength_matches import HEATMAP_MODELS, protocol_vulnerability
    cell = heatmap_cells(st["matches"], protocol_vulnerability)
    ms = [n for n in HEATMAP_MODELS if any(n in k for k in cell)]
    from itertools import groupby
    groups = [len(list(g)) for _, g in groupby(ms, key=model_group)]
    head, body = heatmap_body(cell, ms, ms, row_groups=groups)
    col = ">{\\centering\\arraybackslash}p{2.9em}"
    spec = "l" + "@{\\hspace{5pt}}".join(f"*{{{n}}}{{{col}}}" for n in groups)
    return ["% Table: head-to-head strength (heatmap, every pair)\n\\begin{table}[htbp]\\centering\\scriptsize\n"
            "\\caption[Duplicate matches between the models]{Duplicate matches between every pair of models on 10{,}000 "
            "held-out deals (each deal played at two tables with the seats swapped, and double-dummy scoring). Matches between "
            "two supervised models (the top-left block: brl SL and the four SL models) are played non-vulnerable, as their training data were, and all "
            "others with random vulnerability. Each cell: IMPs per board for the row model against the column model, shaded blue where the "
            "row model is stronger and red where it is weaker, and bold where the 95\\% CI excludes zero. Small grey number: "
            "share (\\%) of the 110{,}497 WBridge5 test positions where the two make the same bid. Each match fills two "
            "cells, mirrored across the diagonal. The gaps set apart five groups of models: brl SL, the supervised models "
            "of this thesis, PPO with the anchor, PPO without it, and brl's RL networks. Example: in the row SL joint, "
            "the first cell reads $+0.15$ over 93.6, so the supervised joint model beats brl SL by 0.15 IMPs per board, "
            "and the two make the same bid at 93.6\\% of the test positions.}\n\\label{tab:strength}\n"
            "\\setlength{\\tabcolsep}{2pt}\\renewcommand{\\arraystretch}{1.25}\n"
            "\\begin{adjustbox}{max width=\\linewidth}\n"
            f"\\begin{{tabular}}{{{spec}}}\\toprule\n"
            f"Model & {head} \\\\\\midrule\n{body}\n\\bottomrule\\end{{tabular}}\n\\end{{adjustbox}}\\end{{table}}\n"]


SUMMARY_MATCHES = [  # (row model, column model): the matches Section 4.2 discusses, in its order
    ("SL plain", "SL joint"), ("SL independent", "SL joint"), ("SL sequential", "SL joint"),
    ("SL joint", "brl SL"), ("SL independent", "brl SL"), ("SL sequential", "brl SL"),
    ("PPO joint", "SL joint"), ("PPO joint unanchored", "SL joint"),
    ("PPO joint unanchored", "PPO independent unanchored")]


def strength_summary_table(ev: Path) -> list:
    """tab:strength_summary: the few matches of tab:strength the text discusses, with their 95% intervals."""
    st = load(ev / "strength/strength.json")
    if not st:
        return []
    from evaluation.strength_matches import protocol_vulnerability
    found = {}
    for m in st["matches"]:
        if m["vulnerability"] != protocol_vulnerability(m["A"], m["B"]):
            continue
        x, (lo, hi), same = m["imps_per_board"], m["ci95"], m["call_agreement_on_wbridge5_positions"]
        found[m["A"], m["B"]] = (x, lo, hi, same)
        found[m["B"], m["A"]] = (-x, -hi, -lo, same)
    num = lambda v: "0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "$-$")  # noqa: E731
    rows = [f"{a} & {b} & {num(found[a, b][0])} & [{num(found[a, b][1])}, {num(found[a, b][2])}] & "
            f"{pct(found[a, b][3])} \\\\" for a, b in SUMMARY_MATCHES if (a, b) in found]
    if not rows:
        return []
    for k in (6, 3):   # a gap after the supervised comparisons and after those against brl SL
        rows.insert(k, "\\addlinespace[3pt]")
    return ["% Table: the matches of tab:strength the text discusses\n\\begin{table}[htbp]\\centering\\small\n"
            "\\caption[The matches discussed in the text]{The matches of Table~\\ref{tab:strength} that the text "
            "discusses. IMPs per board for the first model against the second, with the 95\\% interval, and the "
            "share of the WBridge5 test positions where the two make the same bid.}\n\\label{tab:strength_summary}\n"
            "\\begin{tabular}{llrcr}\\toprule\n"
            "Model & against & IMPs/board & 95\\% CI & same bid (\\%) \\\\\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\\end{tabular}\n\\end{table}\n"]


# ---------------------------------------------------------------- what the concepts cost (runtime_cost.py's medians)

EXPLAIN_PARTS = [  # (key in explain.json, label, colour); "frame rest" is the frame without the parts listed separately
    ("forward pass", "forward pass", OI[8]),
    ("rule engine (concept truth)", "rule engine", OI[5]),
    ("concept flips", "concept flips", OI[6]),
    ("card counterfactuals", "card counterfactuals", OI[4]),
    ("rule justification", "rule justification", OI[3]),
    ("frame rest", "rest of the frame", OI[2]),
    ("teacher text", "teacher text", OI[0]),
    ("checker", "checker", OI[1]),
    ("integrated gradients", "Integrated Gradients", OI[7]),
    ("rest", "the rest of the record", "0.85"),
]


def explain_means(e: dict) -> dict:
    """Mean ms per call of each disjoint part (the frame's own sub-parts taken out of 'frame')."""
    ms = {k: v["mean"] for k, v in e["ms"].items()}
    inner = ("rule engine (concept truth)", "concept flips", "card counterfactuals", "rule justification",
             "trust flag")
    out = {k: ms.get(k, 0.0) for k in ("forward pass", "teacher text", "checker", "integrated gradients")}
    out.update({k: ms.get(k, 0.0) for k in inner if k != "trust flag"})
    out["frame rest"] = ms.get("frame", 0.0) - sum(ms.get(k, 0.0) for k in inner)
    out["rest"] = ms["total"] - sum(out.values())
    out["total"] = ms["total"]
    return out


def fig_cost(ev: Path, out: Path):
    """(a) Supervised learning: the data prepared once (the deals replayed, the rule engine's labels) next to one
    training run; (b) one PPO update of the joint design and of a true bottleneck, by part, on the GPU measured alone
    (runtime_gpu/; the CPU parts if it is missing); (c) one explained bid in the interface, by part, on the CPU,
    where the interface runs the bidding models."""
    d = ev / "runtime"
    if not (d / "ppo.json").exists() or not (d / "explain.json").exists():
        return
    ppo, ex = load(d / "ppo.json")["results"], load(d / "explain.json")["results"]
    on_gpu = (ev / "runtime_gpu/ppo.json").exists()  # the training ran on the GPU; the CPU parts are the fallback
    if on_gpu:
        ppo = load(ev / "runtime_gpu/ppo.json")["results"]
    lab, logs = load(d / "labels.json"), load(d / "logs.json")
    if lab and logs:
        fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(7.0, 7.0),
                                         gridspec_kw={"height_ratios": (1, 1, 1.25), "hspace": 1.6})
        lab, runs = lab["results"], logs["results"]["sl_gpu_minutes"]
        rules = lab["all_minutes_rule_engine"]
        replay = lab["all_minutes"] - rules
        run = sorted(runs[m]["minutes"] for m in ("joint", "independent", "sequential"))[1]
        a0.barh(0, replay, color="0.75", height=0.62, label="deals replayed (needed without concepts too)")
        a0.barh(0, rules, left=replay, color=OI[5], height=0.62, label="concept labels (rule engine)")
        a0.barh(1, run, color=OI[1], height=0.62, label="training")
        a0.text(replay + rules + 1, 0, f"{replay + rules:.0f} min, of which the labels {rules:.0f}", va="center", fontsize=7)
        a0.text(run + 1, 1, f"{run:.0f} min", va="center", fontsize=7)
        a0.set_yticks([0, 1], ["prepare the data\n(once, CPU)", "one training run\n(GPU, three at once)"], fontsize=7.5)
        a0.invert_yaxis()
        a0.set_xlim(0, run * 1.25)
        a0.set_xlabel(f"minutes ({lab['all_decisions'] / 1e6:.1f} million decisions)", fontsize=7.5)
        a0.set_title("(a) Supervised learning: the data are labelled once, before training", fontsize=8.5, loc="left")
        a0.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2)
        tag = "bc"
    else:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 3.9), gridspec_kw={"height_ratios": (1, 1.25), "hspace": 1.1})
        tag = "ab"
    parts = [("rollout_s", "self-play roll-out", OI[8]), ("gae_s", "GAE", "0.75"),
             ("anchor_reference_logits_s", "anchor reference", OI[3]),
             ("ppo_update_anchored_s", "PPO update (10 epochs)", OI[1]),
             ("label_pool_s", "concept labels (rule engine)", OI[5]),
             ("concept_gradient_step_s", "concept gradient step", OI[0])]
    # The GPU's timeline of one update. The joint design's labels are made by the CPU workers while the GPU runs the
    # PPO update (training/ppo.py submits them before the update and collects them after it), so they are drawn on
    # their own row under that update and add nothing to its length: in the thesis's joint runs the trainer waited a
    # median of 3 ms for them.
    rows = {"joint": 0.0, "independent": 1.45}
    lab_y = 0.62
    for m, y in rows.items():
        left = 0.0
        for key, label, c in parts:
            if key == "label_pool_s" or (m != "joint" and key == "concept_gradient_step_s"):
                continue
            v = ppo[m][key]["median"]
            if m == "joint" and key == "concept_gradient_step_s":
                lv = ppo[m].get("label_fast_pool_cold_s", ppo[m]["label_pool_s"]["median"])
                a1.barh(lab_y, lv, left=update_start, color=OI[5], height=0.3, label="concept labels (rule engine, CPU)")
                left = max(left, update_start + lv)
            if key.startswith("ppo_update"):
                update_start = left
            a1.barh(y, v, left=left, color=c, height=0.62, label=label if m == "joint" else None)
            left += v
        a1.text(left + (0.02 if on_gpu else 0.3), y, f"{left:.2f} s" if on_gpu else f"{left:.1f} s", va="center",
                fontsize=7)
    a1.set_yticks([0.0, lab_y, rows["independent"]], ["joint", "CPU, at the same time", "independent\n(true bottleneck)"],
                  fontsize=7.5)
    a1.tick_params(axis="y", length=0)
    a1.invert_yaxis()
    a1.set_xlabel("seconds per update (GPU, run alone)" if on_gpu
                  else "seconds per update (CPU)", fontsize=7.5)
    a1.set_title(f"({tag[0]}) One PPO update: the bids are labelled at every update", fontsize=8.5, loc="left")
    a1.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=3)
    models = [m for m in ("independent SL", "joint SL", "brl SL") if m in ex]
    for i, m in enumerate(models):
        em = explain_means(ex[m])
        left = 0.0
        for key, label, c in EXPLAIN_PARTS:
            v = max(em.get(key, 0.0), 0.0)
            a2.barh(i, v, left=left, color=c, height=0.62, label=label if i == 0 else None)
            left += v
        a2.text(left + 0.6, i, f"{em['total']:.0f} ms", va="center", fontsize=7)
    a2.set_yticks(range(len(models)), [m if m.startswith("brl") else m.replace(" SL", "") for m in models], fontsize=7.5)
    a2.invert_yaxis()
    a2.set_xlabel("milliseconds per explained bid (CPU)", fontsize=7.5)
    a2.set_title(f"({tag[1]}) One explained bid in the interface", fontsize=8.5, loc="left")
    a2.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4)
    save(fig, out, "fig_cost")


def runtime_table(ev: Path) -> list:
    """tab:runtime (appendix): every measurement of evaluation/runtime_cost.py, one column per network."""
    d = ev / "runtime"
    parts = {k: load(d / f"{k}.json") for k in ("static", "infer", "labels", "sl", "ppo", "explain", "logs")}
    if not all(parts.values()):
        return []
    st, inf, lab, sl, ppo, ex, logs = (parts[k]["results"] for k in ("static", "infer", "labels", "sl", "ppo", "explain", "logs"))
    cols = ["brl SL", "SL plain", "joint SL", "independent SL", "sequential SL"]
    num = lambda x, f="{:,.0f}": f.format(x).replace(",", "{,}")  # noqa: E731
    med = lambda k: sl[k]["median"]  # noqa: E731
    runs, gpu = logs["sl_gpu_minutes"], logs["ppo_gpu"]
    by = {(v["mode"], v["anchor"] > 0, v["concept_coef"] > 0 or v["mode"] != "joint"): v["s_per_update"]["median"]
          for v in gpu.values()}

    def cpu_update(m):
        r = ppo[m]
        t = sum(r[k]["median"] for k in ("rollout_s", "gae_s", "anchor_reference_logits_s", "ppo_update_anchored_s"))
        if m == "joint":
            t += r.get("label_fast_pool_cold_s", r["label_pool_s"]["median"]) + r["concept_gradient_step_s"]["median"]
        return t
    rows = [
        ("Parameters (thousands)", [num(st[c]["total"] / 1e3) for c in cols]),
        ("of which concept layer", ["--"] + [num(st[c]["parameters"]["hard_state_head"] / 1e3) for c in cols[1:]]),
        ("One decision, CPU (ms)", [f"{inf[c]['batch1_ms']['median']:.2f}" for c in cols]),
        ("4{,}096 decisions, CPU (ms)", [f"{inf[c]['batch4096_ms']:.0f}" for c in cols]),
        ("Labelling the supervised data, CPU (min)$^a$", ["--", "--"] + [f"{lab['all_minutes_rule_engine']:.1f}"] * 3),
        ("Supervised step, CPU (ms)", [f"{med('no concept head (brl architecture)'):.0f}", f"{med('plain'):.0f}",
                                       f"{med('joint'):.0f}", f"{med('independent'):.0f}",
                                       f"{med('sequential A'):.0f} / {med('sequential B'):.0f}"]),
        ("Supervised run, GPU (min)", ["--", f"{sum(runs[f'plain seed {s}']['minutes'] for s in (44, 1, 2)) / 3:.0f}"]
         + [f"{runs[m]['minutes']:.0f}" for m in ("joint", "independent", "sequential")]),
        ("PPO update, GPU (s)", ["--", f"{by['joint', False, False]:.2f}$^b$", f"{by['joint', True, True]:.2f}",
                                 f"{by['independent', True, True]:.2f}", f"{by['sequential', True, True]:.2f}"]),
        ("PPO update, CPU (s)", ["--", "--", f"{cpu_update('joint'):.1f}", f"{cpu_update('independent'):.1f}", "--"]),
        ("Explained bid, CPU (ms)", [f"{ex['brl SL']['ms']['total']['mean']:.0f}", "--",
                                     f"{ex['joint SL']['ms']['total']['mean']:.0f}",
                                     f"{ex['independent SL']['ms']['total']['mean']:.0f}", "--"]),
    ]
    cpu = parts["ppo"]["machine"]["cpu"].replace("(R)", "").replace("(TM)", "")
    ref = {m: ppo[m]["anchor_reference_logits_s"]["median"] for m in ("joint", "independent")}
    qwen = logs["qwen_gpu"]["eval_adapter_testsplit"]["s_per_explanation"]
    body = "\n".join(f"{name} & " + " & ".join(v) + " \\\\" for name, v in rows)
    return ["% Table: runtime (evaluation/report/rq1_cost.py)\n\\begin{table}[htbp]\\centering\\small\n"
            "\\caption[What the concepts cost in computation]{What the concepts cost in computation "
            f"(\\texttt{{evaluation/runtime\\_cost.py}}). CPU: {cpu} ({parts['ppo']['machine']['cores']} threads), medians over "
            "repeated runs after compilation; GPU: the wall-clock of the real runs on one 8\\,GB NVIDIA RTX 3060 Ti, from "
            "their logs. A supervised step is one batch of 1{,}024 decisions (forward, backward, AdamW); brl's column is "
            "the same network without the concept layer, trained on the policy loss alone; the sequential mode's two "
            "phases are given separately (concepts only / policy only). A PPO update is 1{,}024 games $\\times$ 32 bids "
            "with the anchor; for the joint model it includes the concept step. An explained bid is one record of the "
            "interface (\\texttt{nlg/decision\\_record.py}) with counterfactuals and Integrated Gradients, mean over the "
            "bids of 100 deals bid by the model at all four seats. The supervised and PPO runs shared the GPU with other "
            "runs, so their rates compare the columns only roughly; the CPU rows are measured in isolation. "
            f"$^a$The rule engine's part of preparing the {num(lab['all_decisions'] / 1e6, '{:.1f}')} million decisions "
            f"({num(lab['all_minutes'], '{:.1f}')} minutes with the replay of the deals, on {lab['pool']['workers']} threads; "
            f"{lab['serial']['rule_engine_ms_per_decision']:.2f}\\,ms per decision on one). One pass serves every concept "
            "model and seed. "
            f"$^b$Without the anchor, whose reference forward pass adds {ref['joint']:.2f}\\,s to a joint update and "
            f"{ref['independent']:.2f}\\,s to an independent one on the CPU (the KL term itself is below the measurement "
            f"noise). The language-model explainer adds {qwen:.1f}\\,s per explanation on the GPU.}}\n"
            "\\label{tab:runtime}\n\\begin{tabular}{lrrrrr}\\toprule\n"
            "Measure & brl SL & Plain & Joint & Independent & Sequential \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n"]


FIGURES = [fig_concept_f1, fig_cost]
TABLES = [strength_table, strength_summary_table, runtime_table]
