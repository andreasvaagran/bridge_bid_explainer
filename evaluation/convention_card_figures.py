#!/usr/bin/env python3
"""The cards learned from play, shown: `departures` (the one definition of "differs from WBridge5"),
the interactive page (cards.html) and the thesis's table and figure (--thesis_dir)."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evaluation.convention_cards_from_play import WB5, slug
from evaluation.common import MODELS

HCP_SHIFT = 2          # either end of the 5th-95th percentile HCP range moves this many points
LEN_SHIFT = 2          # either end of the length range held moves this many cards ...
LEN_MEDIAN_SHIFT = 1   # ... or its median this many
SHARE_SHIFT = 0.10     # the share of hands with the entry's feature ("5+ cards") moves this much
FREQ_RATIO = 1.5       # the call is made this many times as often, or 1/FREQ_RATIO as often
AVG_LEN = 3.25         # an average hand's length in a suit
STANDS_OUT = 0.5       # a "key suit" is at least this many cards from AVG_LEN on average
SUITS = "SHDC"
SYM = {"S": "\u2660", "H": "\u2665", "D": "\u2666", "C": "\u2663"}

# the models in the order the page, the table and the printed cards list them
GROUPS: List[Tuple[str, List[str]]] = [
    ("Reference", [WB5]),
    ("Supervised (SL)", ["SL joint", "SL independent", "SL sequential"]),
    ("PPO with KL anchor", ["PPO joint", "PPO independent", "PPO sequential"]),
    ("PPO without anchor", ["PPO joint unanchored", "PPO independent unanchored", "PPO sequential unanchored"]),
    ("Plain network (no concepts)", ["SL plain", "PPO plain no-anchor", "PPO plain unanchored"]),
    ("brl (Kita et al. 2024)", ["brl SL", "brl RL", "brl RL+PFSP", "brl RL+FSP", "brl RL scratch"]),
]


def short(name: str) -> str:
    return (name.replace("independent", "indep.").replace("sequential", "seq.")
            .replace(" unanchored", ", no anchor").replace(" no-anchor", ", no anchor"))


def call_text(c: str) -> str:
    return c[:-1] + SYM[c[-1]] if len(c) == 2 and c[0].isdigit() and c[-1] in SYM else c


def load_cards(cards_dir: Path) -> Dict[str, dict]:
    """{model name: card} for every card in cards_dir, in GROUPS order (unlisted models last, under "Other")."""
    names = {slug(n): n for n in [WB5, *MODELS]}
    found = {}
    for f in sorted(cards_dir.glob("*.json")):
        c = json.loads(f.read_text())
        if "headline" in c:
            found[names.get(f.stem, f.stem)] = c
    order = [n for _, ns in GROUPS for n in ns if n in found]
    return {n: found[n] for n in order + sorted(set(found) - set(order))}


def groups_of(cards: Dict[str, dict]) -> List[Tuple[str, List[str]]]:
    listed = {n for _, ns in GROUPS for n in ns}
    out = [(t, [n for n in ns if n in cards]) for t, ns in GROUPS]
    out.append(("Other", [n for n in cards if n not in listed]))
    return [(t, ns) for t, ns in out if ns]


# ---------------------------------------------------------------- departures from WBridge5
def departures(e: dict, w: dict) -> Dict[str, List[str]]:
    """{kind: [reason, ...]} where entry e departs from WBridge5's entry w; kinds hcp, length, shape, frequency,
    few (too few calls for ranges). Empty: no departure."""
    out: Dict[str, List[str]] = {}
    add = lambda k, s: out.setdefault(k, []).append(s)  # noqa: E731
    if "hcp" not in e:
        if "hcp" in w:
            add("few", f"too few calls ({e['n']}) to give its ranges")
    else:
        h, r = e["hcp"], w.get("hcp")
        if r and (abs(h[0] - r[0]) >= HCP_SHIFT or abs(h[1] - r[1]) >= HCP_SHIFT):
            add("hcp", f"{h[0]}\u2013{h[1]} HCP where WBridge5 shows {r[0]}\u2013{r[1]}")
        a, b = e.get("held", {}).get("range"), w.get("held", {}).get("range")
        if a and b and (abs(a[0] - b[0]) >= LEN_SHIFT or abs(a[2] - b[2]) >= LEN_SHIFT
                        or abs(a[1] - b[1]) >= LEN_MEDIAN_SHIFT):
            add("length", f"{a[0]}\u2013{a[2]} in {e['held']['of']} (median {a[1]}) where WBridge5 holds "
                          f"{b[0]}\u2013{b[2]} (median {b[1]})")
        if e.get("measure") != w.get("measure"):
            add("shape", f"{e.get('measure') or 'no feature'} where WBridge5 shows {w.get('measure') or 'none'}")
        elif e.get("measure") and abs(e["measure_share"] - w["measure_share"]) >= SHARE_SHIFT:
            add("shape", f"{e['measure']} in {e['measure_share']:.0%} of hands, WBridge5 {w['measure_share']:.0%}")
    fe, fw = e.get("frequency") or 0, w.get("frequency") or 0
    if fw and (fe / fw >= FREQ_RATIO or fe / fw <= 1 / FREQ_RATIO):
        add("frequency", f"chosen {fe:.1%} of the time, WBridge5 {fw:.1%}")
    return out


def card_departures(card: dict, ref: dict) -> List[Dict[str, List[str]]]:
    refh = {(h["group"], h["label"]): h for h in ref["headline"]}
    return [departures(h, refh[(h["group"], h["label"])]) if (h["group"], h["label"]) in refh else {}
            for h in card["headline"]]


# ---------------------------------------------------------------- the interactive page
def page_data(cards: Dict[str, dict]) -> dict:
    ref = cards[WB5]
    models = []
    for name, c in cards.items():
        dep = card_departures(c, ref) if name != WB5 else [{} for _ in c["headline"]]
        models.append({
            "key": slug(name), "name": name, "short": short(name), "decisions": c["decisions"],
            "coverage": c["coverage"], "fidelity": c["fidelity"], "wb5_fidelity": c.get("wbridge5_fidelity"),
            "differing": c.get("contexts_differing"), "source": c.get("source", ""),
            "headline": [{"g": h["group"], "l": h["label"], "n": h["n"], "f": h.get("frequency") or 0,
                          "hcp": h.get("hcp"), "m": h.get("measure"), "ms": h.get("measure_share"),
                          "hl": h.get("held"), "hb": h.get("held_by_call"), "sp": h.get("suit_profiles"),
                          "al": d} for h, d in zip(c["headline"], dep)]})
    return {"groups": [{"title": t, "keys": [slug(n) for n in ns]} for t, ns in groups_of(cards)],
            "models": models, "thresholds": {"hcp": HCP_SHIFT, "len": LEN_SHIFT, "median": LEN_MEDIAN_SHIFT,
                                             "share": SHARE_SHIFT, "freq": FREQ_RATIO, "avg": AVG_LEN,
                                             "stands_out": STANDS_OUT}}


def write_page(cards: Dict[str, dict], out: Path) -> None:
    tpl = (Path(__file__).with_name("convention_card_page.html")).read_text()
    data = json.dumps(page_data(cards), separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    out.write_text(tpl.replace("__DATA__", data))


# ---------------------------------------------------------------- the thesis's table and figure
def _tex(s: str) -> str:
    for k, v in {"\u2660": "\\spades{}", "\u2665": "\\hearts{}", "\u2666": "\\diamonds{}", "\u2663": "\\clubs{}",
                 "%": "\\%", "&": "\\&", "_": "\\_"}.items():
        s = s.replace(k, v)
    return s


def _pct(x: Optional[float]) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def thesis_table(cards: Dict[str, dict]) -> str:
    ref = cards[WB5]
    lab = "1\u2666 (1st/2nd seat)"
    rows = []
    for t, ns in groups_of(cards):
        for n in ns:
            c = cards[n]
            red = "--" if n == WB5 else str(sum(bool(d) for d in card_departures(c, ref)))
            e = next((h for h in c["headline"] if h["label"] == lab), {})
            r = e.get("held", {}).get("range")
            d1 = f"{e['hcp'][0]}--{e['hcp'][1]}, {r[0]}--{r[2]}" if "hcp" in e and r else "--"
            # the share of the model's decisions made in auctions it plays differently, not their number
            diff = sum(e["share_of_decisions"] for e in c["contexts"].values() if e.get("differs"))
            rows.append(f"{_tex(n)} & {red} & {_pct(c['fidelity'])} & {_pct(c.get('wbridge5_fidelity'))} & "
                        f"{_pct(diff) if n != WB5 else '--'} & {d1} \\\\")
        rows.append("\\midrule")
    rows.pop()
    n_e = len(ref["headline"])
    return ("% Table: convention cards learned from play (evaluation/convention_card_figures.py)\n"
            "\\begin{table}[htbp]\\centering\\small\n"
            "\\caption[Convention cards learned from play]{Convention cards learned from 50{,}000 deals of each model's "
            "own play (\\texttt{evaluation/convention\\_cards\\_from\\_play.py}). Departures: how many of the "
            f"{n_e} entries of the headline card (the main openings, responses and competitive bids) differ from WBridge5's (an end of the HCP range moves {HCP_SHIFT}+ "
            f"points, an end of the length range moves {LEN_SHIFT}+ cards or its median {LEN_MEDIAN_SHIFT}+ card, a "
            f"feature's share {int(SHARE_SHIFT * 100)}+ points, or the bid is made {FREQ_RATIO:g} times as often or "
            "less than two-thirds as often). Own card and WBridge5's card: share of the held-out decisions that each "
            "card's decision trees predict. Played differently: share of the model's decisions made in auctions where "
            "its own card predicts its bids 10+ points better than WBridge5's card does. 1\\diamonds{} opening: HCP and diamonds held, "
            "5th--95th percentile.}\n"
            "\\label{tab:convention_cards}\n\\begin{adjustbox}{max width=\\linewidth}\n"
            "\\begin{tabular}{lrrrrl}\\toprule\n"
            f"Model & Departures (of {n_e}) & Own card (\\%) & WBridge5's card (\\%) & Played differently (\\%) & "
            "1\\diamonds{} opening \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\n\\end{adjustbox}\\end{table}\n")


FIG_MODELS = [WB5, "SL joint", "PPO joint", "PPO joint unanchored", "PPO plain no-anchor", "brl RL+FSP",
              "brl RL scratch"]
FIG_PANELS = [("1\u2666 (1st/2nd seat)", "", "1D", "1\u2666 opening"),
              ("2\u2665", "1NT", "2H", "2\u2665 over 1NT"),
              ("Takeout double", "1H", "X", "Double of 1\u2665")]


def suit_figure(cards: Dict[str, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from evaluation.report.style import OI, save   # the thesis's rcParams and palette

    models = [m for m in FIG_MODELS if m in cards]
    cmap = LinearSegmentedColormap.from_list("short_long", [OI[5], "#f2f2f2", OI[4]])   # short / average / long
    fig, axes = plt.subplots(1, len(FIG_PANELS), figsize=(6.3, 0.32 * len(models) + 1.0), sharey=True)
    for ax, (label, after, call, title) in zip(axes, FIG_PANELS):
        grid = []
        for m in models:
            e = next((h for h in cards[m]["headline"] if h["label"] == label), {})
            p = next((p for p in e.get("suit_profiles", []) if p["after"] == after and p["call"] == call), None)
            grid.append([p["mean"][s] if p else float("nan") for s in SUITS])
        im = ax.imshow(grid, cmap=cmap, vmin=AVG_LEN - 2.75, vmax=AVG_LEN + 2.75, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                txt = "too few" if v != v else f"{v:.1f}"
                dark = v == v and abs(v - AVG_LEN) > 1.6
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if dark else "black")
        ax.set_xticks(range(4), [SYM[s] for s in SUITS], fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(models)), [m.replace(" no-anchor", " unanchored") for m in models])  # the thesis's names
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("mean length held (cards)")
    cb.ax.axhline(AVG_LEN, color="black", lw=0.6)
    cb.outline.set_visible(False)
    save(fig, out, "fig_card_suits")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards_dir", required=True, help="the --out_dir of evaluation/convention_cards_from_play.py")
    ap.add_argument("--page", default=None, help="the interactive page (default <cards_dir>/cards.html)")
    ap.add_argument("--thesis_dir", default=None, help="writes tables/convention_cards.tex and "
                                                       "figures/fig_card_suits.{pdf,png} here")
    args = ap.parse_args()
    d = Path(args.cards_dir)
    cards = load_cards(d)
    if WB5 not in cards:
        raise SystemExit(f"{d} has no WBridge5 card; run evaluation/convention_cards_from_play.py first")
    page = Path(args.page) if args.page else d / "cards.html"
    write_page(cards, page)
    print(f"[cards] {len(cards)} cards -> {page}")
    if args.thesis_dir:
        t = Path(args.thesis_dir)
        (t / "tables").mkdir(parents=True, exist_ok=True)
        (t / "figures").mkdir(parents=True, exist_ok=True)
        (t / "tables" / "convention_cards.tex").write_text(thesis_table(cards))
        suit_figure(cards, t / "figures")
        print(f"[cards] table and figure -> {t}/tables/convention_cards.tex, {t}/figures/fig_card_suits.pdf")


if __name__ == "__main__":
    main()
