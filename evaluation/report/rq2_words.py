"""Section 4.4, explanations in words: the Qwen explainer (tab:qwen), how faithful the stated reasons are, the
warning signal, and the follow-up questions (tab:followup)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from evaluation.report.style import MODE_COLOR, OI, REPO, load, pct, plt, save


def qwen_table(ev: Path) -> list:
    rows = []
    # the deployed explainer and the untrained model, on the same 550 decisions with the teacher's current texts
    for tag, lab in (("eval_adapter_testsplit", "Qwen3-8B + QLoRA, deployed"),
                     ("eval_base_testsplit", "Qwen3-8B, no fine-tuning")):
        r = load(ev / f"qwen/{tag}/eval.json")
        if r:
            # deployed behaviour: retries, then the repair of the two known false clauses (nlg.eval_qwen.repair)
            retried = (pct(r.get("pass_rate_with_retries_and_repair", r["pass_rate_with_retries"])) if r.get("retries")
                       else "--")
            rows.append(f"{lab} & {r['n']} & {pct(r['consistent_rate'])} & {pct(r['style_rate'])} & {pct(r['pass_rate'])} & "
                        f"{retried} & {r['token_f1']:.2f} \\\\")
    if not rows:
        return []
    return ["% Table: Qwen explainer\n\\begin{table}[htbp]\\centering\\small\n"
            "\\caption[The Qwen explainer on held-out positions]{The Qwen explainer on held-out WBridge5 test positions (up to 10 per bidding situation, 548 in all), greedy decoding. "
            "With retries: up to 3 sampled retries when a check fails, then the repair of a known false clause (deployed "
            "behaviour). Both are scored against the teacher's current texts. Token $F_1$ against the teacher's explanation of the same "
            "decision.}\n\\label{tab:qwen}\n"
            "\\begin{tabular}{lrrrrrr}\\toprule\n"
            "Explainer & $n$ & Consistent & Style & Both & With retries & Token $F_1$ \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"]


def fig_explanation_faithfulness(ev: Path, out: Path):
    """The concepts an explanation states against random and top concepts: their share of the bid's margin, the drop
    of p(bid) and how often the bid changes when they are removed (independent SL, non-pass bids)."""
    r = load(ev / "explanations/SL_independent__test.json")
    if not r:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    groups = [("stated", "stated in the explanation"), ("random_stated", "random concepts (same number)"),
              ("top_stated", "top concepts by contribution")]
    for ax, (metric, lab) in zip(axes, (("share", "share of the bid's margin"), ("compr", "drop of p(bid) when removed"),
                                        ("flip", "bid changes when removed"))):
        for j, (g, gl) in enumerate(groups):
            v = r.get(f"{g}_{metric}_nonpass")
            if v:
                ax.bar(j, v["value"], color=OI[[4, 8, 2][j]],
                       yerr=[[v["value"] - v["ci95"][0]], [v["ci95"][1] - v["value"]]], capsize=3, label=gl)
        ax.set_xticks([])
        ax.set_title(lab, fontsize=8.5)
        ax.set_ylim(0, 1)
    fig.set_layout_engine("tight", rect=(0, 0.1, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7, frameon=False)
    save(fig, out, "fig_explanation_faithfulness")


def fig_warning(ev: Path, out: Path):
    """(a) ROC of the concept warning and of the policy's confidence for bids that differ from WBridge5's; (b) how
    often each certainty band differs (nlg/trust_calibration.json)."""
    from evaluation.common import auroc, wilson
    tc = load(REPO / "nlg/trust_calibration.json")
    recs = {}
    for m in ("independent", "joint"):
        p = ev / f"explanations/SL_{m}__test_records.pkl"
        if p.exists():
            recs[m] = pickle.load(open(p, "rb"))["uniform"]
    if not recs or not tc:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": (1.1, 1)})   # (a)'s legend in its empty corner
    signals = (("concept", lambda x: 0.5 - x["min_margin_concept_uncertainty"], "-"),
               ("confidence", lambda x: 1 - x["conf"], "--"))
    for m in ("independent", "joint"):
        if m not in recs:
            continue
        err = np.array([bool(x["error"]) for x in recs[m]])
        for lab, f, ls in signals:
            s = np.array([f(x) for x in recs[m]])
            order = np.argsort(-s, kind="stable")
            tp = np.cumsum(err[order]) / err.sum()
            fp = np.cumsum(~err[order]) / (~err).sum()
            a1.plot(np.r_[0, fp], np.r_[0, tp], ls=ls, color=MODE_COLOR[m], lw=1.2,
                    label=f"{m}: {lab} ({auroc(s, err):.2f})")
    a1.plot([0, 1], [0, 1], color="0.7", lw=0.7, ls=":")
    a1.set_xlabel("agreements flagged")
    a1.set_ylabel("departures flagged")
    a1.set_title("(a) Flagging the departures", fontsize=8.5)
    a1.legend(fontsize=6.6, loc="lower right", title="signal (AUROC)", title_fontsize=6.6, frameon=False,
               handlelength=1.5, borderaxespad=0.2)
    bands = None
    for j, (m, col) in enumerate((("independent", MODE_COLOR["independent"]), ("joint", MODE_COLOR["joint"]))):
        b = tc.get(f"SL {m}")
        if not b:
            continue
        bands = b["bins"]
        x = np.arange(len(bands)) + (j - 0.5) * 0.38
        rate = np.array([r for _, _, r, _ in bands])
        ci = [wilson(int(round(r * n)), n) for _, _, r, n in bands]
        a2.bar(x, 100 * rate, width=0.38, color=col, label=f"{m} (all decisions: {100 * b['overall']:.1f}%)")
        a2.errorbar(x, 100 * rate, yerr=[100 * (rate - np.array([c[1] for c in ci])),
                                         100 * (np.array([c[2] for c in ci]) - rate)],
                    fmt="none", ecolor="0.25", lw=0.7, capsize=1.5)
    labels = [f"<{hi:g}" if lo == 0 else (f"≥\n{lo:g}" if hi >= 1 else f"{lo:g}–\n{hi:g}") for lo, hi, _, _ in bands]
    a2.set_xticks(range(len(bands)), labels, fontsize=7)
    a2.set_xlabel("certainty of the least certain contributing concept")
    a2.set_ylabel("bids that differ from WBridge5's (%)")
    a2.set_ylim(0, 60)
    a2.set_title("(b) How often each band differs", fontsize=8.5)
    a2.legend(fontsize=6.5, loc="upper right", frameon=False)
    fig.set_layout_engine("tight")
    save(fig, out, "fig_warning")


def followup_table(ev: Path) -> list:
    fu = load(ev / "followup/followup_eval.json")
    if not fu:
        return []
    rows = []
    groups = ("train-phrasing", "held-out phrasing", "indirect", "hand-written", "all")
    lab = {"rule": "keyword parser + teacher", "base": "Qwen3-8B, no fine-tuning", "tuned": "Qwen3-8B + QLoRA"}
    for sysname in ("rule", "base", "tuned"):
        r = fu["results"].get(sysname)
        if not r:
            continue
        sm = r["summary"]
        cells = " & ".join(pct(sm[g]["parse_acc"], 1) if g in sm else "--" for g in groups)
        ans = sm["all"].get("answer_pass")
        rows.append(f"{lab[sysname]} & {cells} & {pct(ans, 1) if ans is not None else '--'} & {pct(sm['all']['e2e'], 1)} \\\\")
    n = {g: fu["results"][next(iter(fu["results"]))]["summary"].get(g, {}).get("n", 0) for g in groups}
    halves = [(100 * (r["summary"][g]["parse_ci95"][1] - r["summary"][g]["parse_ci95"][0]) / 2, g)
              for r in fu["results"].values() for g in groups if g in r["summary"] and "parse_ci95" in r["summary"][g]]
    max_half = max((h for h, g in halves if g != "all"), default=0)
    all_half = max((h for h, g in halves if g == "all"), default=0)
    return ["% Table: follow-up questions\n\\begin{table}[htbp]\\centering\\small\n"
            "\\caption[Follow-up questions]{Follow-up questions about held-out decisions of the independent SL model (\\texttt{nlg/followup\\_questions\\_eval.py}). "
            "Query: share of questions turned into the right query, by group: phrasings used in training ("
            f"{n['train-phrasing']}), phrasings held out of training ({n['held-out phrasing']}), questions that name the bid "
            f"or card only through the auction (indirect, {n['indirect']}), and questions written separately from the "
            f"generator ({n['hand-written']}). Answer: share of answers, given the right query, that pass every check. "
            "End to end: the right query and an answer that passes. The teacher's template answers pass by construction, and the "
            "untrained Qwen3-8B was run on every second question (275). 95\\% Wilson intervals: at most $\\pm$"
            + f"{max_half:.0f}" + " points for a group, $\\pm$" + f"{all_half:.0f}" + " for all questions.}\n"
            "\\label{tab:followup}\n\\begin{tabular}{lrrrrrrr}\\toprule\n"
            " & \\multicolumn{5}{c}{Query (\\%)} & & \\\\\n"
            "System & trained & held out & indirect & hand-written & all & Answer (\\%) & End to end (\\%) \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"]


FIGURES = [fig_explanation_faithfulness, fig_warning]
TABLES = [qwen_table, followup_table]
