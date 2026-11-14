#!/usr/bin/env python3
"""The real decisions behind the thesis's example figures, each chosen by a fixed rule from the test split
or a stored evaluation output, written as TikZ (bxfigures.sty macros). --only recomputes one example;
--render_only rewrites the TikZ from examples.json."""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The computation needs the project's environment (numpy, JAX, the checkpoints); --render_only, which only
# rewrites the TikZ from a stored examples.json, runs with a bare Python.
try:
    import numpy as np

    from concepts.auction_obs import auction_from_obs  # noqa: E402,F401
    from concepts.vocabulary import gloss_for_concept  # noqa: E402
    from concepts.hand import decode_hand  # noqa: E402
    from evaluation.concept_influence import head_parts  # noqa: E402
    from evaluation.common import BIDS, MODELS, load_test_split, log_softmax_masked  # noqa: E402
    from training.checkpoint import load_checkpoint  # noqa: E402
    _IMPORT_ERROR = None
except ImportError as _e:  # pragma: no cover - only without the project's environment
    _IMPORT_ERROR = _e

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "runs" / "evaluation"   # --eval_dir sets it
RANKS = "AKQJT98765432"
SEATS = "NESW"

# short labels for the concept pills (the glosses are written for sentences)
SHORT = {
    "bid_shows_5_plus_cards": "bids a natural suit of five or more cards",
    "bid_shows_4_plus_cards": "bids a natural suit of four or more cards",
    "bid_shows_6_plus_cards": "bids a suit of 6+ cards",
    "bid_is_weak_two": "is a weak two opening",
    "bid_shows_major_suit": "shows a major suit",
    "bid_shows_quick_tricks_2plus": "has 2+ quick tricks",
    "bid_shows_stopper_in_their_suit": "has a stopper in the opponents' suit",
    "bid_is_1_level_opening": "is a one-level opening",
    "bid_shows_opening_values": "shows opening values",
}


# ----------------------------------------------------------------------------- formatting
def tex_bid(b: str) -> str:
    if b in ("P", "X", "XX"):
        return {"P": r"\bxbid{}{p}", "X": r"\bxbid{}{x}", "XX": r"\bxbid{}{r}"}[b]
    lvl, st = b[0], b[1:]
    return r"\bxbid{%s}{%s}" % (lvl, {"C": "c", "D": "d", "H": "h", "S": "s", "NT": "n"}[st])


def hand_dict(obs) -> dict:
    h = decode_hand(np.asarray(obs))
    return {s: sorted(h.cards.get(s, []), key=RANKS.index) for s in "SHDC"}


def hand_from_str(s: str) -> dict:
    out = {k: [] for k in "SHDC"}
    for part in s.split():
        su, rk = part.split(":")
        out[su] = sorted([c for c in rk if c in RANKS], key=RANKS.index)   # "S:-" is a void
    return out


def hand_text(h: dict) -> str:
    return " ".join(f"{s}:{''.join(h[s]) or '-'}" for s in "SHDC")


def tex_hand(h: dict, at: str, style: dict | None = None) -> str:
    style = style or {}
    cards = []
    for s in "SHDC":
        for r in h[s]:
            cards.append(f"{'10' if r == 'T' else r}/{s.lower()}/{style.get(s + r, 'n')}")
    return r"\bxhand{%s}{%s}" % (at, ", ".join(cards))


def hcp(h: dict) -> int:
    return sum({"A": 4, "K": 3, "Q": 2, "J": 1}.get(r, 0) for s in h for r in h[s])


def shape(h: dict) -> str:
    return "-".join(str(len(h[s])) for s in "SHDC")


def short(name: str) -> str:
    return SHORT.get(name, gloss_for_concept(name))


def tex_escape(s: str) -> str:
    import re
    s = re.sub(r"(?<![\w-])([1-7])(NT|C|D|H|S)(?!\w)", lambda m: "\x00" + m.group(1) + m.group(2) + "\x01", s)
    s = (s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
             .replace("♠", r"\bxS{}").replace("♥", r"\bxH{}").replace("♦", r"\bxD{}").replace("♣", r"\bxC{}"))
    return re.sub("\x00(.*?)\x01", lambda m: tex_bid(m.group(1)), s)


def pct(p: float) -> str:
    return f"{100 * p:.0f}\\%" if p >= 0.005 else r"${<}1$\%"


# ----------------------------------------------------------------------------- models
class Head:
    """A concept checkpoint with its policy layer split into the hidden and concept parts."""

    def __init__(self, name: str):
        import jax
        self.name = name
        self.ck = load_checkpoint(str(MODELS[name]))
        self.Wh, self.Wc, self.b = head_parts(self.ck)
        self.fwd = jax.jit(lambda o: self.ck.net.apply(self.ck.params, o))
        self.names = list(self.ck.hard_concepts)

    def run(self, obs: np.ndarray):
        out = self.fwd(obs.astype(np.float32))
        return (np.asarray(out["policy_logits"], np.float64), np.asarray(out["concept_acts"], np.float64))


def probs(L: np.ndarray, legal: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_masked(L, legal))


def top(L, legal):
    p = probs(L[None], legal[None])[0]
    k = int(p.argmax())
    return BIDS[k], float(p[k]), p


# ----------------------------------------------------------------------------- examples
def ex_bypass(pos, joint: Head, ind: Head, rows: np.ndarray) -> dict:
    """First position (an opening bid of 1S/1H by the first seat to bid) where both models make WBridge5's
    bid, the joint model keeps it with every concept off, and the independent model changes it when one of its
    three largest contributions from an active, rule-confirmed concept (no strain descriptor) is switched off."""
    Lj, Aj = joint.run(pos.obs[rows])
    Li, Ai = ind.run(pos.obs[rows])
    for t, i in enumerate(rows):
        calls, _ = auction_from_obs(pos.obs[i])
        wb5 = BIDS[pos.action[i]]
        if calls or wb5 not in ("1S", "1H"):
            continue
        lg = pos.legal[i]
        bj, pj, _ = top(Lj[t], lg)
        bi, pi, _ = top(Li[t], lg)
        if bj != wb5 or bi != wb5:
            continue
        bj0, pj0, _ = top(Lj[t] - Aj[t] @ joint.Wc, lg)
        if bj0 != wb5:
            continue
        a = Ai[t]
        k_bid = BIDS.index(wb5)
        p_all = probs(Li[t][None], lg[None])[0]
        p_all[k_bid] = 0
        runner = int(p_all.argmax())
        m = a * (ind.Wc[:, k_bid] - ind.Wc[:, runner])
        targets = pos.targets[i]
        cand = [k for k in np.argsort(-m) if not ind.names[k].startswith("bid_is_in_") and targets[k] and a[k] > 0.5]
        for k in cand[:3]:
            bi0, pi0, _ = top(Li[t] - a[k] * ind.Wc[k], lg)
            if bi0 != wb5:
                h = hand_dict(pos.obs[i])
                # the same concept in the joint model, for the caption
                kj = joint.names.index(ind.names[k])
                bjk, pjk, _ = top(Lj[t] - Aj[t][kj] * joint.Wc[kj], lg)
                return {"row": int(i), "hand": hand_text(h), "hcp": hcp(h), "shape": shape(h), "auction": calls,
                        "wbridge5": wb5, "joint": [bj, pj], "joint_all_off": [bj0, pj0],
                        "joint_concept_off": [bjk, pjk], "independent": [bi, pi],
                        "concept": ind.names[k], "gloss": short(ind.names[k]), "activation": float(a[k]),
                        "independent_concept_off": [bi0, pi0]}
    raise RuntimeError("no bypass example found")


def ex_intervention(pos, ind: Head, rows: np.ndarray, wrong_side: bool = True) -> dict:
    """First independent-model error, in an auction of at most two earlier bids, that correcting only the
    least certain concept (to the rule engine's value) turns into WBridge5's bid; preferably one where that
    concept was on the wrong side of 0.5."""
    L, A = ind.run(pos.obs[rows])
    for t, i in enumerate(rows):
        calls, _ = auction_from_obs(pos.obs[i])
        if len(calls) > 2:
            continue
        lg = pos.legal[i]
        b0, p0, _ = top(L[t], lg)
        wb5 = BIDS[pos.action[i]]
        if b0 == wb5:
            continue
        a = A[t]
        k = int(np.argmin(np.abs(a - 0.5)))
        if ind.names[k].startswith("bid_is_in_"):
            continue
        tv = float(pos.targets[i][k])
        if wrong_side and (a[k] > 0.5) == bool(tv):
            continue
        a2 = a.copy(); a2[k] = tv
        b1, p1, _ = top(L[t] - a @ ind.Wc + a2 @ ind.Wc, lg)
        if b1 != wb5:
            continue
        h = hand_dict(pos.obs[i])
        shown = [int(j) for j in np.argsort(-a)[:4] if j != k and not ind.names[j].startswith("bid_is_in_")][:3]
        return {"row": int(i), "hand": hand_text(h), "hcp": hcp(h), "shape": shape(h), "auction": calls,
                "wbridge5": wb5, "before": [b0, p0], "after": [b1, p1],
                "concept": ind.names[k], "gloss": short(ind.names[k]), "activation": float(a[k]), "truth": bool(tv),
                "others": [[ind.names[j], short(ind.names[j]), float(a[j]), bool(pos.targets[i][j])] for j in shown]}
    if wrong_side:
        return ex_intervention(pos, ind, rows, wrong_side=False)
    raise RuntimeError("no intervention example found")


def ex_faithfulness(pos, ind: Head) -> dict:
    recs = pickle.load(open(EVAL / "explanations" / "SL_independent__test_records.pkl", "rb"))["uniform"]
    for r in recs:
        if (r.get("bid") in ("P",) or not r.get("stated_flip") or r.get("random_stated_flip", 1) > 0.25
                or r.get("error") or not r.get("consistent") or r.get("words", 99) > 40 or r.get("n_stated", 0) > 3):
            continue
        i = int(r["row"])
        calls, _ = auction_from_obs(pos.obs[i])
        if len(calls) > 3:
            continue
        L, A = ind.run(pos.obs[i:i + 1])
        lg = pos.legal[i]
        b0, p0, _ = top(L[0], lg)
        stated = [n for n in sorted(set(r["reasons"]) | set(r["meaning"])) if n in ind.names]
        S = [ind.names.index(n) for n in stated]
        a2 = A[0].copy(); a2[S] = 0
        b1, p1, _ = top(L[0] - A[0] @ ind.Wc + a2 @ ind.Wc, lg)
        h = hand_dict(pos.obs[i])
        return {"row": i, "hand": hand_text(h), "hcp": hcp(h), "shape": shape(h), "auction": calls,
                "bid": [b0, p0], "text": r["text"], "stated": [[n, short(n), float(A[0][ind.names.index(n)])] for n in stated],
                "stated_removed": [b1, p1], "random_flip_share": float(r["random_stated_flip"]),
                "stated_share": float(r["stated_share"]), "random_share": float(r["random_stated_share"])}
    raise RuntimeError("no faithfulness example found")


def ex_anchor(pos) -> dict:
    """First explained test position where SL joint and anchored PPO joint make WBridge5's bid and the
    unanchored PPO joint run does not (a non-pass bid by one of the two)."""
    load = lambda f: {r["row"]: r for r in pickle.load(open(EVAL / "explanations" / f, "rb"))["uniform"]}
    sl, an, no = (load("SL_joint__test_records.pkl"), load("PPO_joint__test_records.pkl"),
                  load("PPO_joint_unanchored__test_records.pkl"))
    for row in sorted(set(sl) & set(an) & set(no)):
        a, b, c = sl[row], an[row], no[row]
        if not (a["bid"] == a["wb5_bid"] == b["bid"] and c["bid"] != a["bid"]):
            continue
        calls, _ = auction_from_obs(pos.obs[row])
        if len(calls) > 2 or "P" in (a["bid"], c["bid"]) or c["conf"] < 0.6:
            continue
        h = hand_dict(pos.obs[row])
        return {"row": int(row), "hand": hand_text(h), "hcp": hcp(h), "shape": shape(h), "auction": calls,
                "wbridge5": a["wb5_bid"], "sl_joint": [a["bid"], a["conf"]], "ppo_anchor": [b["bid"], b["conf"]],
                "ppo_no_anchor": [c["bid"], c["conf"]], "text_sl": a.get("text", ""), "text_no_anchor": c.get("text", ""),
                "no_anchor_note": c.get("deviation", False)}
    raise RuntimeError("no anchor example found")


def ex_drift() -> dict:
    d = json.load(open(EVAL / "drift" / "drift.json"))
    e = d["brl RL+FSP"]["examples"]["opening_1_minor"][0]
    cards = json.load(open(EVAL / "drift" / "convention_cards.json"))
    ex = cards.get("examples", {})
    return {"model": "brl RL+FSP", **e, "deviation_rate": {m: d[m]["deviation_rate_non_pass"] for m in
                                                            ("WBridge5", "brl SL", "brl RL", "brl RL+PFSP", "brl RL+FSP")},
            "card_examples": ex, "opening_bands": opening_bands()}


BANDS = ((0, 7), (8, 11), (12, 14), (15, 17), (18, 40))


def opening_bands() -> dict:
    """What each system does first to bid (1st seat, or 2nd after a pass), by the hand's HCP: the most common
    call and its share per band. WBridge5 on its test deals, RL+FSP in 3,000 self-play deals (seed 0, as the
    deviation rate). Shows RL+FSP's strong passes and its 1C opening on weak hands."""
    from evaluation.common import BIDS, selfplay
    out = {}
    for who, pos in (("WBridge5", load_test_split()), ("brl RL+FSP", selfplay(MODELS["brl RL+FSP"], 3000, seed=0))):
        keep = [i for i in range(len(pos)) if pos.step[i] == 0 or (
            pos.step[i] == 1 and pos.deal[i - 1] == pos.deal[i] and BIDS[pos.action[i - 1]] == "P")]
        H = np.array([hcp(hand_dict(pos.obs[i])) for i in keep])
        A = np.array([BIDS[pos.action[i]] for i in keep])
        bands = []
        for lo, hi in BANDS:
            m = (H >= lo) & (H <= hi)
            v, c = np.unique(A[m], return_counts=True)
            k = int(np.argmax(c))
            bands.append({"hcp": [lo, hi], "n": int(m.sum()), "call": str(v[k]), "share": float(c[k] / m.sum())})
        P = H[A == "P"]
        out[who] = {"bands": bands, "passes_12plus": float((P >= 12).mean()), "n": len(keep)}
    return out


def ex_distill() -> dict:
    """A distilled model's bid in RL+FSP's system, with its explanation: the first 1D opening, in 3,000
    self-play deals of the distilled independent model (scripts/distill_rl_fsp.py; seed 0, random vulnerability,
    as its explanations were measured), on at most two diamonds and four or more clubs, whose teacher explanation
    passes the checks and carries a deviation note (explanation_faithfulness.analyse_decision)."""
    import random
    import jax
    from evaluation.explanation_faithfulness import analyse_decision
    from evaluation.common import selfplay
    from nlg.explanation_frames import Enricher
    from evaluation.common import DISTILLED
    path = DISTILLED["distilled independent"]
    ck = load_checkpoint(str(path))
    en = Enricher(ck)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    pos = selfplay(path, 3000, 0, True)
    for i in range(len(pos)):
        if BIDS[pos.action[i]] != "1D":
            continue
        before, seat, dl = pos.auction(i)
        if any(c["bid"] != "P" for c in before):
            continue
        h = hand_dict(pos.obs[i])
        if len(h["D"]) > 2 or len(h["C"]) < 4:
            continue
        r = analyse_decision(ck, en, fwd, pos, i, random.Random(0))
        if r["bid"] != "1D" or not (r.get("consistent") and r.get("style_ok") and r.get("deviation")):
            continue
        return {"model": "distilled independent", "path": str(path), "row": int(i), "hand": hand_text(h),
                "hcp": hcp(h), "shape": shape(h), "seat": seat, "auction": before, "vul": pos.vul[i], "bid": r["bid"],
                "text": r["text"], "conf": r.get("conf"), "stated_share": r.get("stated_share"),
                "stated_flip": r.get("stated_flip"), "random_stated_flip": r.get("random_stated_flip"),
                "reasons": r.get("reasons"), "deals": 3000, "seed": 0}
    raise RuntimeError("no 1D opening on two diamonds or fewer and four clubs or more")


def ex_swap() -> dict:
    """The message swap (evaluation/bid_meaning_tests.py swap) on one decision: the first decision, in split
    order of RL+FSP's held-out games (runs/distill/prepared/test), where partner replies directly to RL+FSP's 1C
    opening with at most 6 HCP, replies 1D, and passes when 1D is swapped in for 1C in its observation, while a
    partner trained on WBridge5 (SL independent) passes both, as it does most weak hands after either opening (the
    typical case for it: 65-69% of its replies stay the same under the swap)."""
    from concepts.auction_obs import BID_NAMES, auction_from_obs
    from evaluation.bid_meaning_tests import _first_call
    from evaluation.common import policy_fn
    pos = load_test_split(REPO / "runs/distill/prepared/test")
    nets = {"rl": policy_fn(MODELS["brl RL+FSP"])[0], "sl": policy_fn(MODELS["SL independent"])[0]}

    def legal_after(op):   # as bid_meaning_tests.swap: pass, or a bid above the opening
        legal = np.zeros(38, bool)
        legal[0] = True
        legal[3 + BID_NAMES.index(op) + 1:] = True
        return legal

    def swapped(o, frm, to):
        o = o.copy()
        o[:, 8 + BID_NAMES.index(frm) * 12 + 2] = 0
        o[:, 8 + BID_NAMES.index(to) * 12 + 2] = 1
        return o

    def reply(net, o, op):
        lg = legal_after(op)
        lp = log_softmax_masked(np.asarray(nets[net](o.astype(np.float32)))[0], lg)
        a = int(np.argmax(np.where(lg, lp, -np.inf)))
        return BIDS[a], float(np.exp(lp[a]))

    for i in range(len(pos)):
        calls, _ = auction_from_obs(pos.obs[i])
        k = _first_call(calls)
        if not (k < 3 and calls[k:] == ["1C", "P"]):
            continue
        h = hand_dict(pos.obs[i])
        if hcp(h) > 6:
            continue
        o = pos.obs[i:i + 1]
        o_sw = swapped(o, "1C", "1D")
        real, sw = reply("rl", o, "1C"), reply("rl", o_sw, "1D")
        if real[0] != "1D" or sw[0] != "P":
            continue
        sl_real, sl_sw = reply("sl", o, "1C"), reply("sl", o_sw, "1D")
        if sl_real[0] != "P" or sl_sw[0] != "P":
            continue
        return {"row": int(i), "split": "runs/distill/prepared/test", "auction": calls, "hand": hand_text(h),
                "hcp": hcp(h), "shape": shape(h), "rl_real": real, "rl_swapped": sw,
                "sl_real": sl_real, "sl_swapped": sl_sw}
    raise RuntimeError("no weak reply to RL+FSP's 1C that the swap turns into a pass")


def ex_followup() -> dict:
    """A question about a change to the hand that the fine-tuned explainer parses right and the keyword
    parser reads as a bid ("one heart" -> 1H), answered and checked; else any such miss."""
    tuned = [json.loads(l) for l in open(EVAL / "followup" / "records_tuned.jsonl")]
    rule = {r["i"]: r for r in map(json.loads, open(EVAL / "followup" / "records_rule.jsonl"))}
    pick = lambda r, rr: {"i": r["i"], "group": r["group"], "question": r["question"], "gold": r["gold"],
                          "parsed": r["parsed"], "rule_parsed": rr["parsed"], "answer": r["answer"],
                          "hand": r.get("hand"), "bid": r.get("bid"), "auction": r.get("auction")}
    for strict in (True, False):
        for r in tuned:
            rr = rule.get(r["i"])
            if rr is None or not (r["parse_ok"] and r["answer_ok"]) or rr["parse_ok"] or r["group"] == "train-phrasing":
                continue
            if strict and not (r["gold"]["type"] == "change" and rr["parsed"].get("type") == "why_not"
                               and 150 < len(r["answer"]) < 300):
                continue
            return pick(r, rr)
    raise RuntimeError("no follow-up example found")


def _misread(e: dict) -> bool:
    """A flagged claim read out of its context: negated ("lack of a stopper in hearts") or a number that is not
    about the hand ("12 spade support points", a partnership's "maximum of 23 HCP")."""
    t, c = e["text"].lower(), e["claim"].strip().lower()
    i = t.find(c)
    if i < 0:
        return False
    before, after = t[max(0, i - 15):i], t[i + len(c):i + len(c) + 16]
    return any(w in before for w in ("lack of", "no ", "without", "maximum", "at most")) or "support point" in after


def ex_checker() -> dict:
    c = json.load(open(EVAL / "checker" / "checker_coverage.json"))
    fe = c["Qwen3-8B, no fine-tuning (first attempt)"]["false_examples"]
    for kind in ("shape", "hcp", "length", "holding", "stopper"):
        for e in fe.get(kind, []):
            if len(e["text"]) < 420 and not _misread(e):
                return {"kind": kind, **e}
    k = next(iter(fe))
    return {"kind": k, **fe[k][0]}


# ----------------------------------------------------------------------------- TikZ
HEAD = "%% generated by code/evaluation/thesis_example_figures.py -- do not edit; regenerate instead\n"


def tikz_bypass(e: dict) -> str:
    """Two rows, one per model: the concepts switched off (grey) and the bid that is left. The joint model's
    side road around its concepts is drawn as the red arrow of Figure fig:modes."""
    h = hand_from_str(e["hand"])
    j0, i0 = e["joint_all_off"], e["independent_concept_off"]
    gl = tex_escape(e["gloss"])
    return HEAD + rf"""\begin{{tikzpicture}}[bx, tag/.style={{font=\sffamily\scriptsize, text=black!75, align=center, inner sep=1pt}},
  off/.style={{fill=black!12, rounded corners=1mm, inner sep=1mm, font=\sffamily\bfseries\scriptsize, text=black!45, anchor=north west}}]
  {tex_hand(h, '(0.3,0)')}
  % ---- joint: every concept switched off, the bid stays
  \node[font=\sffamily\bfseries\footnotesize, anchor=west] at (0,-1.5) {{Joint}};
  \pic at (2.15,-1.5) {{bxnet}};
  \node[bx arrow] at (4.45,-1.5) {{}};
  \node[fill=black!20, rounded corners=2mm, minimum width=36mm, minimum height=11mm, font=\sffamily\bfseries\scriptsize, text=black!55, align=center] at (7.0,-1.5) {{all 146 concepts\\ switched off}};
  \node[bx arrow] at (9.6,-1.5) {{}};
  \node[bx predcard, minimum width=12mm, minimum height=11mm] (bj) at (10.75,-1.5) {{}};
  \node[bx bidcard, minimum width=9mm, minimum height=6.5mm] at (bj) {{{tex_bid(j0[0])}}};
  \node[tag, anchor=west] at (11.5,-1.5) {{{pct(j0[1])}}};
  \draw[-{{Latex[length=2.5mm]}}, line width=1.6pt, bxred!80, rounded corners=8pt] (3.0,-0.95) -- (3.0,-0.62) -- (10.75,-0.62) -- (10.75,-0.9);
  % ---- independent: one concept switched off, the bid changes
  \node[font=\sffamily\bfseries\footnotesize, anchor=west] at (0,-3.6) {{Independent}};
  \pic at (2.15,-3.6) {{bxnet}};
  \node[bx arrow] at (4.45,-3.6) {{}};
  \node[off, text width=34mm] (o1) at (5.2,-3.0) {{{gl}\hfill 0}};
  \node[draw=bxred, line width=1.2pt, rounded corners=1mm, inner sep=0.5mm, fit=(o1)] {{}};
  \node[bx pill, text width=34mm, anchor=north west] (o2) at ($(o1.south west)+(0,-0.11)$) {{\strut 145 concepts as predicted}};
  \begin{{scope}}[on background layer]
    \node[bx panel, fit=(o1) (o2)] {{}};
  \end{{scope}}
  \node[bx arrow] at (9.6,-3.6) {{}};
  \node[bx predcard, minimum width=12mm, minimum height=11mm, fill=bxpurple!70] (bi) at (10.75,-3.6) {{}};
  \node[bx bidcard, minimum width=9mm, minimum height=6.5mm] at (bi) {{{tex_bid(i0[0])}}};
  \node[tag, anchor=west] at (11.5,-3.6) {{{pct(i0[1])}}};
\end{{tikzpicture}}
"""


def tikz_intervention(e: dict) -> str:
    """The test-time intervention on one decision, drawn as Figure fig:cbm_training: the least certain concept
    (framed red) is set to the rule engine's value (framed green), and the bid is recomputed."""
    h = hand_from_str(e["hand"])
    rest = [rf"{tex_escape(g)} \hfill {a:.2f}" for _, g, a, _ in e["others"]]
    before = ", ".join(f"plain/{{{t}}}" for t in [rf"{tex_escape(e['gloss'])} \hfill {e['activation']:.2f}"] + rest)
    after = ", ".join(f"plain/{{{t}}}" for t in [rf"{tex_escape(e['gloss'])} \hfill {int(e['truth'])}"] + rest)
    return HEAD + rf"""\begin{{tikzpicture}}[bx]
  {tex_hand(h, '(0.3,0)')}
  \node[bx arrow] at (5.95,0) {{}};
  \bxpanel{{(6.75,0.95)}}{{50mm}}{{{before}}}
  \node[draw=bxred, line width=1.2pt, rounded corners=1mm, inner sep=0.5mm, fit=(bxpill0)] {{}};
  \coordinate (bxtop) at (bxpanelbox.south);
  \node[bx arrow] at (12.35,0) {{}};
  \node[bx predcard, minimum height=12mm] (p) at (13.55,0) {{}}; \node[bx bidcard] at (p) {{{tex_bid(e['before'][0])}}};
  \node[bx label, below=0.5mm of p] {{{pct(e['before'][1])}}};
  \bxpanel{{(6.75,-2.45)}}{{50mm}}{{{after}}}
  \node[draw=bxgreen, line width=1.2pt, rounded corners=1mm, inner sep=0.5mm, fit=(bxpill0)] {{}};
  \coordinate (bxbot) at (bxpanelbox.north);
  \path (bxtop) -- (bxbot) node[midway, single arrow, fill=bxred, draw=none, rotate=-90, minimum width=6.5mm,
    minimum height=5mm, single arrow head extend=1.2mm, inner sep=0pt] (bxfix) {{}};
  \node[anchor=west, font=\sffamily\scriptsize\bfseries, text=bxred] at ([xshift=2.5mm]bxfix.center) {{set to the rule engine's value}};
  \node[bx arrow] at (12.35,-3.4) {{}};
  \node[bx predcard, minimum height=12mm] (q) at (13.55,-3.4) {{}}; \node[bx bidcard] at (q) {{{tex_bid(e['after'][0])}}};
  \node[bx label, below=0.5mm of q] {{{pct(e['after'][1])}}};
\end{{tikzpicture}}
"""


def tikz_faithfulness(e: dict) -> str:
    """The removal test on one decision: the explanation (a speech bubble), its stated concepts switched off
    (the bid changes), and as many random true concepts switched off (the share of draws that change it)."""
    h = hand_from_str(e["hand"])
    stated = ", ".join(rf"off/{{{tex_escape(g)}\hfill 0}}" for _, g, _ in e["stated"])
    share = e["random_flip_share"]
    n = len(e["stated"])
    return HEAD + rf"""\begin{{tikzpicture}}[bx, tag/.style={{font=\sffamily\scriptsize, text=black!75, align=center, inner sep=1pt}}]
  \providecommand{{\bxpilloff}}[1]{{\strut\textcolor{{black!50}}{{#1}}}}
  {tex_hand(h, '(0.3,0)')}
  \node[bx arrow] at (5.95,0) {{}};
  \node[bx predcard, minimum width=12mm, minimum height=11mm] (p) at (7.1,0) {{}}; \node[bx bidcard, minimum width=9mm, minimum height=6.5mm] at (p) {{{tex_bid(e['bid'][0])}}};
  \node[bx label, below=0.5mm of p] {{{pct(e['bid'][1])}}};
  \node[bx textbox, text width=56mm, fill=white, anchor=west] (t) at (8.4,0) {{\emph{{``{tex_escape(e['text'])}''}}}};
  \draw[black!20, fill=white] ($(t.west)+(0.02,0.2)$) -- ($(t.west)+(-0.35,0)$) -- ($(t.west)+(0.02,-0.2)$);
  % ---- switch off the concepts the explanation states
  \node[bx head, minimum width=60mm, anchor=north west] at (0,-1.55) {{Switch off the {n} concept{'s' if n != 1 else ''} it states}};
  \bxpanel{{(0.15,-2.6)}}{{46mm}}{{{stated}}}
  \node[bx arrow] at (5.45,-3.35) {{}};
  \node[bx predcard, minimum width=12mm, minimum height=11mm, fill=bxpurple!70] (q) at (6.55,-3.35) {{}}; \node[bx bidcard, minimum width=9mm, minimum height=6.5mm] at (q) {{{tex_bid(e['stated_removed'][0])}}};
  \node[tag, below=0.5mm of q] {{{pct(e['stated_removed'][1])}}};
  % ---- switch off as many random true concepts
  \node[bx head, minimum width=58mm, anchor=north west] at (8.3,-1.55) {{Switch off {n} random true concept{'s' if n != 1 else ''}}};
  \foreach \k in {{0,...,19}} {{
    \pgfmathsetmacro{{\bxx}}{{8.45+mod(\k,10)*0.55}}
    \pgfmathsetmacro{{\bxy}}{{-2.9-floor(\k/10)*0.75}}
    \pgfmathparse{{\k<{round(20 * share)} ? 1 : 0}}
    \ifnum\pgfmathresult=1
      \node[bx bidcard, draw=bxpurple, line width=0.8pt, minimum width=4.5mm, minimum height=5.5mm, font=\sffamily\bfseries\tiny] at (\bxx,\bxy) {{\textcolor{{bxpurple}}{{$\neq$}}}};
    \else
      \node[bx bidcard, draw=bxteal, line width=0.8pt, minimum width=4.5mm, minimum height=5.5mm, font=\sffamily\bfseries\tiny] at (\bxx,\bxy) {{\textcolor{{bxteal}}{{=}}}};
    \fi
  }}
\end{{tikzpicture}}
"""


def tikz_anchor(e: dict) -> str:
    h = hand_from_str(e["hand"])
    a, b, c = e["sl_joint"], e["ppo_anchor"], e["ppo_no_anchor"]
    card = lambda x, y, name, bid, p, col: rf"""
  \node[bx predcard, minimum height=12mm, fill={col}] (c{int(x * 10)}) at ({x},{y}) {{}}; \node[bx bidcard] at (c{int(x * 10)}) {{{tex_bid(bid)}}};
  \node[bx label, below=1mm of c{int(x * 10)}, align=center] {{{name}\\ {pct(p)}}};"""
    return HEAD + rf"""\begin{{tikzpicture}}[bx]
  {tex_hand(h, '(0.3,0)')}
  \node[bx arrow] at (5.95,0) {{}};{card(7.5, 0, 'SL joint', a[0], a[1], 'bxteal')}{card(9.9, 0, 'PPO, with the anchor', b[0], b[1], 'bxteal')}{card(12.3, 0, 'PPO, no anchor', c[0], c[1], 'bxpurple!70')}
\end{{tikzpicture}}
"""


def tikz_drift(e: dict) -> str:
    """A bid outside WBridge5's system: the hand, the bid and what the system promises, and below it what each
    system does first to bid, by HCP (RL+FSP's strong passes and its 1C opening on weak hands)."""
    h = hand_from_str(e["hand"])
    bid = e["bid"]
    rows = []
    ob = e.get("opening_bands")
    if ob:
        xs = [3.6 + 2.1 * k for k in range(len(BANDS))]
        for x, (lo, hi) in zip(xs, BANDS):
            lab = f"{lo}+" if hi >= 40 else f"{lo}--{hi}"
            rows.append(rf"  \node[tag] at ({x},-1.45) {{{lab} HCP}};")
        for y, who in ((-2.25, "WBridge5"), (-3.35, "brl RL+FSP")):
            rows.append(rf"  \node[font=\sffamily\bfseries\scriptsize, anchor=west] at (0.3,{y}) {{{who}}};")
            for x, bd in zip(xs, ob[who]["bands"]):
                mark = who == "brl RL+FSP" and ((bd["call"] == "P" and bd["hcp"][0] >= 12) or (bd["call"] == "1C" and bd["hcp"][1] <= 11))
                fill = "bxpurple!70" if mark else "black!25"
                rows.append(rf"  \node[bx predcard, minimum width=10mm, minimum height=8mm, fill={fill}] (c) at ({x - 0.35},{y}) {{}}; "
                            rf"\node[bx bidcard, minimum width=7.5mm, minimum height=5.5mm] at (c) {{{tex_bid(bd['call'])}}};")
                rows.append(rf"  \node[tag, anchor=west] at ({x + 0.25},{y}) {{{pct(bd['share'])}}};")
    conv = "\n".join(rows)
    return HEAD + rf"""\begin{{tikzpicture}}[bx, tag/.style={{font=\sffamily\scriptsize, text=black!75, align=center, inner sep=1pt}}]
  {tex_hand(h, '(0.3,0)')}
  \node[bx arrow] at (5.95,0) {{}};
  \node[bx predcard, minimum width=12mm, minimum height=11mm, fill=bxpurple!70] (p) at (7.1,0) {{}}; \node[bx bidcard, minimum width=9mm, minimum height=6.5mm] at (p) {{{tex_bid(bid)}}};
  \node[bx textbox, text width=48mm, anchor=west, fill=white, draw=bxred!60] at (8.0,0) {{{tex_escape(e['note'])}}};
{conv}
\end{{tikzpicture}}
"""


def tikz_distill(e: dict) -> str:
    """A distilled model's 1D opening that shows clubs: the hand, the bid and the checked explanation, whose note
    says the bid departs from WBridge5's system, but no concept can say what it shows instead."""
    h = hand_from_str(e["hand"])
    return HEAD + rf"""\begin{{tikzpicture}}[bx]
  {tex_hand(h, '(0.3,0)')}
  \node[bx arrow] at (5.95,0) {{}};
  \node[bx predcard, minimum width=12mm, minimum height=11mm, fill=bxpurple!70] (p) at (7.1,0) {{}}; \node[bx bidcard, minimum width=9mm, minimum height=6.5mm] at (p) {{{tex_bid(e['bid'])}}};
  \node[font=\sffamily\scriptsize, text=black!75] at (7.1,-0.85) {{{pct(e['conf'])}}};
  \node[bx textbox, text width=55mm, anchor=west, fill=white, draw=bxred!60, font=\sffamily\scriptsize] at (8.0,0) {{{tex_escape(e['text'])}}};
\end{{tikzpicture}}
"""


def tikz_swap(e: dict) -> str:
    """The message swap on one hand: partner's hand, and partner's reply when it hears RL+FSP's real 1C and when 1D is
    swapped in, for RL+FSP's partner and for a partner trained on WBridge5."""
    h = hand_from_str(e["hand"])
    card = lambda x, y, bid, p, col: rf"""
  \node[bx predcard, minimum width=11mm, minimum height=9mm, fill={col}] (c{int(x * 10)}{int(abs(y) * 10)}) at ({x},{y}) {{}}; \node[bx bidcard, minimum width=8.5mm, minimum height=6mm] at (c{int(x * 10)}{int(abs(y) * 10)}) {{{tex_bid(bid)}}};
  \node[bx label, anchor=west] at ({x + 0.65},{y}) {{{pct(p)}}};"""
    rows = []
    for y, heard, rl, sl in ((0.55, r"partner hears \bxbid{1}{c}", e["rl_real"], e["sl_real"]),
                             (-0.75, r"\bxbid{1}{d} swapped in", e["rl_swapped"], e["sl_swapped"])):
        rows.append(rf"""
  \node[bx label, anchor=east, align=right] at (8.0,{y}) {{{heard}}};""" + card(9.0, y, rl[0], rl[1], "bxpurple!70")
                    + card(11.6, y, sl[0], sl[1], "bxteal"))
    return HEAD + rf"""\begin{{tikzpicture}}[bx]
  {tex_hand(h, '(0.3,-0.1)')}
  \node[bx label, align=center] at (9.25,1.45) {{RL+FSP's partner}};
  \node[bx label, align=center] at (11.85,1.45) {{trained on WBridge5}};{''.join(rows)}
\end{{tikzpicture}}
"""


def _first_change(answer: str):
    """The first card change of a change answer: (card in, card out, new bid, old bid), or None."""
    import re
    m = re.search(r"With the ([♠♥♦♣])(10|[AKQJT2-9]) instead of the ([♠♥♦♣])(10|[AKQJT2-9]).*?would bid (\S+) instead of (\S+?)[.,]", answer)
    if not m:
        return None
    su = {"♠": "S", "♥": "H", "♦": "D", "♣": "C"}
    return (su[m.group(1)], m.group(2).replace("10", "T"), su[m.group(3)], m.group(4).replace("10", "T"),
            m.group(5), m.group(6))


def tikz_followup(e: dict) -> str:
    """A follow-up question as a chat: the player's question, the explainer's parse as three tags (the keyword
    parser's wrong parse crossed out), and the answer drawn as the hand with one card swapped and the new bid."""
    q = tex_escape(e["question"])
    parsed = " ".join(tex_escape({"C": "♣", "D": "♦", "H": "♥", "S": "♠"}.get(str(v), str(v)).replace("_", " "))
                      for v in e["parsed"].values())
    rule = (" ".join(tex_escape(str(v).replace("_", " ")) for v in e["rule_parsed"].values())
            if e["rule_parsed"] else "no query")
    ch = _first_change(e["answer"])
    if ch and e.get("hand"):
        s_in, r_in, s_out, r_out, new_bid, old_bid = ch
        h = hand_from_str(e["hand"])
        h2 = {k: list(v) for k, v in h.items()}
        h2[s_out] = [r for r in h2[s_out] if r != r_out]
        h2[s_in] = sorted(h2[s_in] + [r_in], key=RANKS.index)
        names = {"C": "club", "D": "diamond", "H": "heart", "S": "spade"}
        kind, suit = e["parsed"].get("kind"), e["parsed"].get("suit")
        changed = (f"one {names[suit]} fewer" if kind == "shorter" and suit in names else
                   f"one {names[suit]} more" if kind == "longer" and suit in names else "after the change")
        answer = rf"""  {tex_hand(h, '(0.3,-1.9)', {s_out + r_out: 'o'})}
  \node[bx label, anchor=south west] at (0,-1.35) {{the hand: {tex_bid(old_bid)}}};
  \node[bx arrow] at (5.95,-1.9) {{}};
  {tex_hand(h2, '(6.9,-1.9)', {s_in + r_in: 'i'})}
  \node[bx label, anchor=south west] at (6.6,-1.35) {{{changed}: {tex_bid(new_bid)}}};
  \node[bx predcard, minimum width=12mm, minimum height=11mm] (a) at (12.9,-1.9) {{}}; \node[bx bidcard, minimum width=9mm, minimum height=6.5mm] at (a) {{{tex_bid(new_bid)}}};
  \node[font=\sffamily\scriptsize, anchor=west] at (13.5,-1.9) {{\bxverified}};"""
    else:
        answer = rf"""  \node[bx textbox, text width=143mm, anchor=north west, fill=bxgreen!8] at (0,-1.35) {{\textbf{{Verified answer}}: {tex_escape(e['answer'])}}};"""
    return HEAD + rf"""\begin{{tikzpicture}}[bx, tag/.style={{font=\sffamily\scriptsize, text=black!75, align=center, inner sep=1pt}}]
  \node[bx textbox, fill=white, draw=bxpurple, line width=0.8pt, anchor=west] (q) at (0,0) {{``{q}''}};
  \node[bx arrow] at ($(q.east)+(0.55,0)$) {{}};
  \node[draw=bxpurple, fill=bxpurple!8, rounded corners=1mm, inner sep=1.2mm, font=\sffamily\scriptsize, anchor=west] (pq) at ($(q.east)+(1.15,0)$) {{{parsed}}};
  \node[tag, anchor=south west] at (pq.north west) {{explainer's query: \textcolor{{bxgreen}}{{\textbf{{right}}}}}};
  \node[draw=black!30, fill=black!5, rounded corners=1mm, inner sep=1.2mm, font=\sffamily\scriptsize, text=black!45, anchor=west] (pr) at ($(pq.east)+(1.9,0)$) {{{rule}}};
  \node[tag, anchor=south west, text=black!50] at (pr.north west) {{keyword parser: \textcolor{{bxred}}{{\textbf{{wrong}}}}}};
  \draw[bxred, line width=0.9pt] (pr.south west) -- (pr.north east);
{answer}
\end{{tikzpicture}}
"""


def tikz_checker(e: dict) -> str:
    h = hand_from_str(e["hand"])
    text = tex_escape(e["text"])
    claim = tex_escape(e["claim"])
    text = text.replace(claim, r"\bxbad{" + claim + "}", 1) if claim in text else text
    why = re.sub(r'\bT\b', '10', e['why'])  # the thesis writes the ten as 10
    return HEAD + rf"""\begin{{tikzpicture}}[bx]
  {tex_hand(h, '(0.3,0)')}
  \node[bx textbox, text width=143mm, anchor=north west] at (0,-0.9) {{\emph{{{text}}}}};
  \node[bx textbox, text width=143mm, anchor=north west, fill=bxred!7] at (0,-3.4) {{\textbf{{Checker:}} the text claims ``{claim}''. The hand holds {tex_escape(why)}. \bxrejected}};
\end{{tikzpicture}}
"""


def write_tex(ex: dict, td: Path):
    td.mkdir(parents=True, exist_ok=True)
    for name, fn in (("bypass", tikz_bypass), ("intervention", tikz_intervention), ("faithfulness", tikz_faithfulness),
                     ("anchor", tikz_anchor), ("drift", tikz_drift), ("followup", tikz_followup), ("checker", tikz_checker),
                     ("distill", tikz_distill), ("swap", tikz_swap)):
        if name in ex:
            (td / f"ex_{name}.tex").write_text(fn(ex[name]))


def main():
    global EVAL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval_dir", default=str(EVAL), help="the evaluation results the examples are taken from")
    ap.add_argument("--out", default=None, help="default: <eval_dir>/examples/examples.json")
    ap.add_argument("--tex_dir", default=str(EVAL / "examples" / "tex"))
    ap.add_argument("--rows", type=int, default=20000, help="test positions searched (split order)")
    ap.add_argument("--only", choices=["distill", "swap"], default=None,
                    help="compute only this example and merge it into the stored --out examples.json")
    ap.add_argument("--render_only", action="store_true",
                    help="only rewrite the TikZ files from the stored --out examples.json (no models, no numpy)")
    args = ap.parse_args()
    EVAL = Path(args.eval_dir).resolve()
    args.out = args.out or str(EVAL / "examples" / "examples.json")
    if args.render_only:
        ex = json.loads(Path(args.out).read_text())
        write_tex(ex, Path(args.tex_dir))
        return
    if _IMPORT_ERROR is not None:
        raise _IMPORT_ERROR
    if args.only:   # one example alone, merged into the stored examples.json (distill: self-play, needs the DDS table)
        ex = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        ex[args.only] = {"distill": ex_distill, "swap": ex_swap}[args.only]()
        Path(args.out).write_text(json.dumps(ex, indent=1, ensure_ascii=False, default=float))
        write_tex(ex, Path(args.tex_dir))
        print(json.dumps(ex[args.only], indent=1, ensure_ascii=False, default=float))
        return
    pos = load_test_split()
    rows = np.arange(min(args.rows, len(pos)))
    joint, ind = Head("SL joint"), Head("SL independent")
    ex = {
        "bypass": ex_bypass(pos, joint, ind, rows),
        "intervention": ex_intervention(pos, ind, rows),
        "faithfulness": ex_faithfulness(pos, ind),
        "anchor": ex_anchor(pos),
        "drift": ex_drift(),
        "followup": ex_followup(),
        "checker": ex_checker(),
        "distill": ex_distill(),
        "swap": ex_swap(),
    }
    ex["meta"] = {"models": {m: str(MODELS[m]) for m in ("SL joint", "SL independent")}, "split": pos.source}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ex, indent=1, ensure_ascii=False, default=float))
    write_tex(ex, Path(args.tex_dir))
    print(json.dumps({k: {kk: v for kk, v in e.items() if kk in ("row", "hand", "wbridge5", "bid", "concept", "question")}
                      for k, e in ex.items() if k != "meta"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
