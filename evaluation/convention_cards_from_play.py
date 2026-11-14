#!/usr/bin/env python3
"""A convention card per model, learned from its own play with decision trees: one CART tree over hand
features per auction reached often enough, what each call shows, a headline card, and WBridge5's tree applied to
the agent's calls. Writes <agent>.json, <agent>.html and index.html."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import html
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from concepts.hand import RANKS, SUIT_ORDER, card_bit_index
from evaluation.common import BIDS, MODELS, TEST_SPLIT, checkpoint_stamp, load_test_split, same_checkpoint, selfplay

VERSION = 5                 # bump when the fitting or the headline changes, so cached agents are recomputed
WB5 = "WBridge5"
NUM_CALLS = len(BIDS)
MAX_DEPTH = 6
MIN_CONTEXT = 150           # decisions in the fitting half before an auction gets a tree
MIN_LEAF_SHARE = 0.01       # a rule must cover >= 1% of its auction's decisions (and >= MIN_LEAF)
MIN_LEAF = 10
DEPTH_TOLERANCE = 0.01      # the smallest depth within one point of the best validation accuracy
DIFFERS_GAP = 0.10          # own card better than WBridge5's by >= 10 points here: a different convention
BLACKWOOD = "BLACKWOOD"     # pooled context: partner's 4NT after a suit of ours, RHO passed

# ---------------------------------------------------------------- hand features
# Derived from concepts.hand.card_bit_index (the repo's one card layout), never a second formula.
_IDX = np.array([[card_bit_index(s + r) for r in RANKS] for s in SUIT_ORDER])   # (4 suits, 13 ranks)
SUITS = list(SUIT_ORDER)    # S H D C
FEATURES: List[Tuple[str, int, int]] = (                                         # (name, min, max)
    [("hcp", 0, 37), ("points", 0, 37 + 9)]
    + [(f"len_{s}", 0, 13) for s in SUITS]
    + [("longest", 4, 13), ("shortest", 0, 3), ("balanced", 0, 1), ("aces", 0, 4), ("kings", 0, 4),
       ("controls", 0, 12)]
    + [(f"top3_{s}", 0, 3) for s in SUITS]
    + [(f"stop_{s}", 0, 1) for s in SUITS]
    + [("we_vul", 0, 1), ("they_vul", 0, 1), ("opening_passes", 0, 3), ("passed_hand", 0, 1)])
F = {name: i for i, (name, _, _) in enumerate(FEATURES)}


def hand_features(obs: np.ndarray) -> np.ndarray:
    """(n, len(FEATURES)) int16 from (n, 480) observations; vulnerability columns left 0."""
    bits = np.asarray(obs)[:, -52:].astype(np.int16)
    H = bits[:, _IDX]                                           # (n, 4, 13), rank order 2..A
    lens = H.sum(2)
    A, K, Q, J = H[:, :, 12], H[:, :, 11], H[:, :, 10], H[:, :, 9]
    hcp = (4 * A + 3 * K + 2 * Q + J).sum(1)
    srt = np.sort(lens, 1)
    balanced = (srt[:, 0] >= 2) & ((lens == 2).sum(1) <= 1)
    stop = (A > 0) | ((K > 0) & (lens >= 2)) | ((Q > 0) & (lens >= 3)) | ((J > 0) & (lens >= 4))
    out = np.zeros((len(bits), len(FEATURES)), np.int16)
    out[:, F["hcp"]] = hcp
    out[:, F["points"]] = hcp + np.maximum(lens - 4, 0).sum(1)
    for k, s in enumerate(SUITS):
        out[:, F[f"len_{s}"]] = lens[:, k]
        out[:, F[f"top3_{s}"]] = H[:, k, 10:].sum(1)
        out[:, F[f"stop_{s}"]] = stop[:, k]
    out[:, F["longest"]] = srt[:, 3]
    out[:, F["shortest"]] = np.minimum(srt[:, 0], 3)
    out[:, F["balanced"]] = balanced
    out[:, F["aces"]] = A.sum(1)
    out[:, F["kings"]] = K.sum(1)
    out[:, F["controls"]] = 2 * A.sum(1) + K.sum(1)
    return out


# ---------------------------------------------------------------- auctions as the bidder sees them
def role(prefix: Sequence[str], j: int) -> str:
    """Who made prefix[j], seen by the player now on turn."""
    return ("me", "RHO", "partner", "LHO")[(len(prefix) - j) % 4]


def strain(call: str) -> Optional[str]:
    return None if call in ("P", "X", "XX") else call[1:]


def section(prefix: Sequence[str]) -> str:
    if tuple(prefix) == (BLACKWOOD,) or blackwood_ask(prefix):
        return "Slam bidding"
    opener = next((j for j, c in enumerate(prefix) if c != "P"), None)
    if opener is None:
        return "Openings"
    ours = lambda r: [c for j, c in enumerate(prefix) if role(prefix, j) == r and c != "P"]  # noqa: E731
    who = role(prefix, opener)
    if who == "partner":
        return "Responding to partner's opening" if not ours("me") else "Responder's later calls"
    if who == "me":
        return "Opener's rebids and later calls"
    if not ours("me") and not ours("partner"):
        return "Over the opponents' opening"
    if ours("partner") and not ours("me"):
        return "Advancing partner's overcall or double"
    return "Later competitive calls"


SECTIONS = ["Openings", "Responding to partner's opening", "Opener's rebids and later calls",
            "Responder's later calls", "Over the opponents' opening", "Advancing partner's overcall or double",
            "Later competitive calls", "Slam bidding", "Rarer auctions, by their last round",
            "Rarer auctions, by the last two calls"]


def blackwood_ask(prefix: Sequence[str]) -> bool:
    """Partner's 4NT just before RHO's pass, after a suit bid by our side (nlg.explanation_inputs.bid_category's test)."""
    if len(prefix) < 2 or prefix[-2] != "4NT" or prefix[-1] != "P":
        return False
    return any(role(prefix, j) in ("me", "partner") and strain(c) in tuple("CDHS")
               for j, c in enumerate(prefix[:-2]))


SUIT_SYM = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


def call_text(call: str) -> str:
    return {"P": "Pass", "X": "Dbl", "XX": "Rdbl"}.get(call, call[0] + SUIT_SYM.get(call[1:], call[1:]))


def auction_text(prefix: Sequence[str], tier: str = "auction") -> str:
    """Our side's calls plain, the opponents' in parentheses, '?' for the bidder; opening
    passes are not shown (the trees read their number as a feature)."""
    if tuple(prefix) == (BLACKWOOD,):
        return "… 4NT (Pass) ?  (any auction where partner's 4NT follows a suit of ours)"
    toks = [call_text(c) if role(prefix, j) in ("me", "partner") else f"({call_text(c)})" for j, c in enumerate(prefix)]
    if not prefix:
        return "? (opening, any seat)"
    return ("… " if tier != "auction" else "") + " ".join(toks + ["?"])


# ---------------------------------------------------------------- auction keys (most specific first)
TIERS = ("auction", "last round", "last two calls")
TIER_MARK = {"last round": "…4 ", "last two calls": "…2 "}


def keys(prefix: Sequence[str]) -> List[str]:
    """The auction without its opening passes (their number is a feature), then only its last
    round (my previous call and the three since), then only partner's and RHO's last calls.
    Roles are counted from the end, so cutting the front never changes who made a call."""
    first = next((j for j, c in enumerate(prefix) if c != "P"), len(prefix))
    core = list(prefix[first:])
    out = [" ".join(core)]
    for tier, k in (("last round", 4), ("last two calls", 2)):
        out.append(TIER_MARK[tier] + " ".join(core[-k:]) if len(core) > k else out[-1])
    return out


def key_prefix(key: str) -> Tuple[List[str], str]:
    for tier, mark in TIER_MARK.items():
        if key.startswith(mark):
            return key[len(mark):].split(), tier
    return (key.split() if key else []), "auction"


# ---------------------------------------------------------------- collecting decisions
def collect(pos) -> dict:
    """Per decision: auction prefix, features, call, deal."""
    from concepts.auction_obs import auction_from_obs
    X = hand_features(pos.obs)
    ctx = []
    for i in range(len(pos)):
        calls, _ = auction_from_obs(pos.obs[i])
        ctx.append(" ".join(calls))
        seat = (int(pos.dealer[i]) + len(calls)) % 4
        v = pos.vul[i]
        ns, ew = v in ("NS", "Both"), v in ("EW", "Both")
        X[i, F["we_vul"]], X[i, F["they_vul"]] = (ns, ew) if seat % 2 == 0 else (ew, ns)
        X[i, F["opening_passes"]] = min(next((j for j, c in enumerate(calls) if c != "P"), len(calls)), 3)
        mine = [c for j, c in enumerate(calls) if role(calls, j) == "me"]
        X[i, F["passed_hand"]] = bool(mine) and all(c == "P" for c in mine)
    return {"ctx": np.array(ctx, dtype=object), "X": X, "y": pos.action.astype(np.int64),
            "deal": pos.deal.astype(np.int64)}


# ---------------------------------------------------------------- CART (gini, integer thresholds)
def _best_split(X: np.ndarray, y: np.ndarray, min_leaf: int):
    n = len(y)
    best = None
    for f, (_, lo, hi) in enumerate(FEATURES):
        x = X[:, f].astype(np.int64)
        V = int(x.max()) + 1
        if V <= 1 or x.min() == x.max():
            continue
        hist = np.bincount(x * NUM_CALLS + y, minlength=V * NUM_CALLS).reshape(V, NUM_CALLS)
        left = np.cumsum(hist, 0)[:-1].astype(np.float64)          # x <= t for t = 0..V-2
        nl = left.sum(1)
        nr = n - nl
        ok = (nl >= min_leaf) & (nr >= min_leaf)
        if not ok.any():
            continue
        right = hist.sum(0)[None, :] - left
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = (nl - (left ** 2).sum(1) / np.where(nl > 0, nl, 1)) + (nr - (right ** 2).sum(1) / np.where(nr > 0, nr, 1))
        imp = np.where(ok, imp, np.inf)
        t = int(imp.argmin())
        if best is None or imp[t] < best[2]:
            best = (f, t, float(imp[t]))
    return best


def grow(X, y, max_depth: int, min_leaf: int, depth: int = 0) -> dict:
    counts = np.bincount(y, minlength=NUM_CALLS)
    node = {"counts": counts, "pred": int(counts.argmax()), "n": len(y)}
    if depth == max_depth or len(y) < 2 * min_leaf or counts.max() == len(y):
        return node
    sp = _best_split(X, y, min_leaf)
    parent = len(y) - (counts.astype(np.float64) ** 2).sum() / len(y)
    if sp is None or sp[2] >= parent - 1e-9:
        return node
    f, t, _ = sp
    m = X[:, f] <= t
    node.update(f=f, t=t, left=grow(X[m], y[m], max_depth, min_leaf, depth + 1),
                right=grow(X[~m], y[~m], max_depth, min_leaf, depth + 1))
    return node


def predict(node: dict, X: np.ndarray, depth: int = 99) -> np.ndarray:
    out = np.empty(len(X), np.int64)

    def go(nd, rows, d):
        if "f" not in nd or d == 0:
            out[rows] = nd["pred"]
            return
        m = X[rows, nd["f"]] <= nd["t"]
        go(nd["left"], rows[m], d - 1)
        go(nd["right"], rows[~m], d - 1)
    go(node, np.arange(len(X)), depth)
    return out


def prune(node: dict, depth: int) -> dict:
    """Cut at `depth`, then merge sibling leaves that predict the same call."""
    if "f" not in node or depth == 0:
        return {"counts": node["counts"], "pred": node["pred"], "n": node["n"]}
    l, r = prune(node["left"], depth - 1), prune(node["right"], depth - 1)
    if "f" not in l and "f" not in r and l["pred"] == r["pred"]:
        return {"counts": node["counts"], "pred": node["pred"], "n": node["n"]}
    return {**{k: node[k] for k in ("counts", "pred", "n", "f", "t")}, "left": l, "right": r}


def fit_tree(X, y, deal) -> Optional[dict]:
    """Fit on even deals, choose the depth on deal % 4 == 1, refit on both, score on deal % 4 == 3."""
    tr, va, te = deal % 2 == 0, deal % 4 == 1, deal % 4 == 3
    if tr.sum() < MIN_CONTEXT:
        return None
    min_leaf = max(MIN_LEAF, int(np.ceil(MIN_LEAF_SHARE * tr.sum())))
    full = grow(X[tr], y[tr], MAX_DEPTH, min_leaf)
    accs = [float(np.mean(predict(full, X[va], d) == y[va])) if va.any() else 0.0 for d in range(MAX_DEPTH + 1)]
    depth = next(d for d in range(MAX_DEPTH + 1) if accs[d] >= max(accs) - DEPTH_TOLERANCE)
    fit = tr | va
    min_leaf = max(MIN_LEAF, int(np.ceil(MIN_LEAF_SHARE * fit.sum())))
    tree = prune(grow(X[fit], y[fit], depth, min_leaf), depth)
    p = predict(tree, X[te])
    base = np.bincount(y[fit], minlength=NUM_CALLS).argmax()
    return {"tree": tree, "depth": depth, "test_n": int(te.sum()), "test_acc": float(np.mean(p == y[te])) if te.any() else None,
            "baseline": float(np.mean(y[te] == base)) if te.any() else None, "test_rows": te}


# ---------------------------------------------------------------- rules from a tree
def _cond_text(name: str, lo: int, hi: int, dmin: int, dmax: int) -> str:
    rng = lambda: (f"{lo}" if lo == hi else f"{lo}+" if hi >= dmax else f"≤{hi}" if lo <= dmin  # noqa: E731
                   else f"{lo}–{hi}")
    if name == "hcp":
        return f"{rng()} HCP"
    if name == "points":
        return f"{rng()} total points"
    if name.startswith("len_"):
        return f"{SUIT_SYM[name[-1]]}{rng()}"
    if name == "longest":
        return f"longest suit {rng()}"
    if name == "shortest":
        return {(0, 0): "a void", (0, 1): "a singleton or void", (1, 1): "a singleton"}.get(
            (lo, hi), "no singleton or void" if lo >= 2 else f"shortest suit {rng()}")
    if name == "balanced":
        return "balanced" if lo == 1 else "unbalanced"
    if name in ("aces", "kings"):
        return f"{rng()} {name}"
    if name == "controls":
        return f"{rng()} controls"
    if name.startswith("top3_"):
        return f"{rng()} of {SUIT_SYM[name[-1]]}AKQ"
    if name.startswith("stop_"):
        return f"{'a' if lo == 1 else 'no'} {SUIT_SYM[name[-1]]} stopper"
    if name == "we_vul":
        return "vulnerable" if lo == 1 else "not vulnerable"
    if name == "they_vul":
        return "opponents vulnerable" if lo == 1 else "opponents not vulnerable"
    if name == "opening_passes":
        return f"after {rng()} opening passes".replace("after 0 opening passes", "with no opening pass")
    if name == "passed_hand":
        return "a passed hand" if lo == 1 else "not a passed hand"
    return f"{name} {rng()}"


def rules(tree: dict, X_te: np.ndarray, y_te: np.ndarray) -> List[dict]:
    """One rule per leaf: the call, its conditions, its share of the auction, how often it holds out of sample."""
    out = []
    total = tree["n"]

    def go(nd, bounds, rows):
        if "f" not in nd:
            conds = [_cond_text(FEATURES[f][0], lo, hi, FEATURES[f][1], FEATURES[f][2])
                     for f, (lo, hi) in sorted(bounds.items())
                     if lo > FEATURES[f][1] or hi < FEATURES[f][2]]
            hit = y_te[rows] == nd["pred"]
            out.append({"call": BIDS[nd["pred"]], "when": conds, "share": nd["n"] / total,
                        "purity": float(nd["counts"][nd["pred"]] / nd["n"]),
                        "holds": float(hit.mean()) if len(rows) else None, "test_n": int(len(rows))})
            return
        f, t = nd["f"], nd["t"]
        lo, hi = bounds.get(f, (FEATURES[f][1], FEATURES[f][2]))
        m = X_te[rows, f] <= t
        go(nd["left"], {**bounds, f: (lo, min(hi, t))}, rows[m])
        go(nd["right"], {**bounds, f: (max(lo, t + 1), hi)}, rows[~m])
    go(tree, {}, np.arange(len(X_te)))
    freq = collections.Counter()
    for r in out:
        freq[r["call"]] += r["share"]
    return sorted(out, key=lambda r: (-freq[r["call"]], r["call"], -r["share"]))


def tree_json(nd: dict) -> dict:
    if "f" not in nd:
        return {"call": BIDS[nd["pred"]], "n": int(nd["n"])}
    return {"feature": FEATURES[nd["f"]][0], "le": int(nd["t"]), "n": int(nd["n"]),
            "left": tree_json(nd["left"]), "right": tree_json(nd["right"])}


def tree_from_json(d: dict) -> dict:
    """The fitted tree back into predict()'s form (for applying another agent's card)."""
    if "call" in d:
        return {"pred": BIDS.index(d["call"]), "n": d["n"]}
    return {"f": F[d["feature"]], "t": d["le"], "n": d["n"], "left": tree_from_json(d["left"]),
            "right": tree_from_json(d["right"])}


# ---------------------------------------------------------------- what each call shows
def bid_suit(call: str) -> Optional[str]:
    s = strain(call)
    return s if s in SUITS else None


def profile(X: np.ndarray, y: np.ndarray) -> Dict[str, dict]:
    out = {}
    n = len(y)
    for c in np.unique(y):
        rows = X[y == c]
        if len(rows) < 5:
            continue
        pc = lambda col: [int(v) for v in np.percentile(rows[:, F[col]], [5, 50, 95], method="nearest")]  # noqa: E731
        L = rows[:, [F[f"len_{s}"] for s in SUITS]]
        shapes = collections.Counter("".join(map(str, sorted(r, reverse=True))) for r in L.tolist())
        out[BIDS[c]] = {"n": int(len(rows)), "share": len(rows) / n, "hcp": pc("hcp"),
                        "len": {s: pc(f"len_{s}") for s in SUITS}, "balanced": float(rows[:, F["balanced"]].mean()),
                        "aces": [int(v) for v in np.bincount(rows[:, F["aces"]], minlength=5)],
                        # full distributions, so a printed card can leave out what is rare (convention_card_ebu_form.py)
                        "hcp_dist": [int(v) for v in np.bincount(rows[:, F["hcp"]], minlength=38)],
                        "len_dist": {s: [int(v) for v in np.bincount(L[:, j], minlength=14)] for j, s in enumerate(SUITS)},
                        "shapes": dict(shapes.most_common(12))}
    return out


# ---------------------------------------------------------------- the headline card
def _passes_then(*tails):
    def m(c):
        return any(len(c) >= len(t) and list(c[len(c) - len(t):]) == list(t) and all(x == "P" for x in c[:len(c) - len(t)])
                   for t in tails)
    return m


def _seats(*k):
    return lambda c: len(c) in k and all(x == "P" for x in c)


def _len_bid(k):
    return lambda X, calls, ctx: np.array([X[i, F[f"len_{bid_suit(c)}"]] >= k if bid_suit(c) else False
                                           for i, c in enumerate(calls)])


def _len(s, k):
    return lambda X, calls, ctx: X[:, F[f"len_{s}"]] >= k


def _short_theirs(X, calls, ctx):
    out = []
    for i, c in enumerate(ctx):
        s = next((strain(x) for x in reversed(c) if strain(x) in SUITS), None)
        out.append(s is not None and X[i, F[f"len_{s}"]] <= 2)
    return np.array(out)


def _unbid_major4(X, calls, ctx):
    out = []
    for i, c in enumerate(ctx):
        bid = {strain(x) for x in c}
        majors = [s for s in "HS" if s not in bid]
        out.append(bool(majors) and all(X[i, F[f"len_{s}"]] >= 4 for s in majors))
    return np.array(out)


OVER1 = _passes_then(("1C",), ("1D",), ("1H",), ("1S",))
HEADLINE = [  # (group, label, auction matcher, calls, extra measure text, measure)
    ("Openings", "1NT (1st/2nd seat)", _seats(0, 1), ["1NT"], "balanced", lambda X, c, x: X[:, F["balanced"]] == 1),
    ("Openings", "2NT (1st/2nd seat)", _seats(0, 1), ["2NT"], "balanced", lambda X, c, x: X[:, F["balanced"]] == 1),
    ("Openings", "1♥ / 1♠ (1st/2nd seat)", _seats(0, 1), ["1H", "1S"], "5+ cards", _len_bid(5)),
    ("Openings", "1♦ (1st/2nd seat)", _seats(0, 1), ["1D"], "4+ cards", _len_bid(4)),
    ("Openings", "1♣ (1st/2nd seat)", _seats(0, 1), ["1C"], "3+ cards", _len_bid(3)),
    ("Openings", "1-level suit opening, 3rd seat", _seats(2), ["1C", "1D", "1H", "1S"], "5+ cards", _len_bid(5)),
    ("Openings", "Pass (1st/2nd seat)", _seats(0, 1), ["P"], "", None),
    ("Openings", "2♣", _seats(0, 1, 2, 3), ["2C"], "balanced", lambda X, c, x: X[:, F["balanced"]] == 1),
    ("Openings", "2♦ / 2♥ / 2♠", _seats(0, 1, 2, 3), ["2D", "2H", "2S"], "6+ cards", _len_bid(6)),
    ("Openings", "3-level preempt", _seats(0, 1, 2), ["3C", "3D", "3H", "3S"], "7+ cards", _len_bid(7)),
    ("Over partner's 1NT", "2♣", _passes_then(("1NT", "P")), ["2C"], "a 4-card major",
     lambda X, c, x: np.maximum(X[:, F["len_H"]], X[:, F["len_S"]]) >= 4),
    ("Over partner's 1NT", "2♦", _passes_then(("1NT", "P")), ["2D"], "5+ hearts", _len("H", 5)),
    ("Over partner's 1NT", "2♥", _passes_then(("1NT", "P")), ["2H"], "5+ spades", _len("S", 5)),
    ("Over partner's 1NT", "3NT", _passes_then(("1NT", "P")), ["3NT"], "", None),
    ("Over partner's 1♥/1♠", "1NT", _passes_then(("1H", "P"), ("1S", "P")), ["1NT"], "", None),
    ("Over partner's 1♥/1♠", "Single raise", _passes_then(("1H", "P"), ("1S", "P")), ["2H", "2S"], "3+ support", _len_bid(3)),
    ("Over partner's 1♥/1♠", "Jump raise to 3", _passes_then(("1H", "P"), ("1S", "P")), ["3H", "3S"], "4+ support", _len_bid(4)),
    ("Over partner's 1♥/1♠", "Raise to game", _passes_then(("1H", "P"), ("1S", "P")), ["4H", "4S"], "4+ support", _len_bid(4)),
    ("Over partner's 1♥/1♠", "2NT", _passes_then(("1H", "P"), ("1S", "P")), ["2NT"], "4+ in partner's major",
     lambda X, c, x: np.array([X[i, F[f"len_{strain(cx[-2])}"]] >= 4 for i, cx in enumerate(x)])),
    ("Over the opponents' 1-level opening", "1-level suit overcall", OVER1, ["1D", "1H", "1S"], "5+ cards", _len_bid(5)),
    ("Over the opponents' 1-level opening", "2-level suit overcall", OVER1, ["2C", "2D", "2H", "2S"], "5+ cards", _len_bid(5)),
    ("Over the opponents' 1-level opening", "1NT overcall", OVER1, ["1NT"], "stopper in their suit",
     lambda X, c, x: np.array([X[i, F[f"stop_{strain(cx[-1])}"]] == 1 if strain(cx[-1]) in SUITS else False
                               for i, cx in enumerate(x)])),
    ("Over the opponents' 1-level opening", "Takeout double", OVER1, ["X"], "≤2 in their suit", _short_theirs),
    ("After partner's 1-level opening and an overcall", "Negative double",
     _passes_then(("1C", "1D"), ("1C", "1H"), ("1C", "1S"), ("1D", "1H"), ("1D", "1S"), ("1H", "1S")), ["X"],
     "4+ in the unbid major(s)", _unbid_major4),
]


# The suit whose length each headline entry is about, measured rather than tested against a threshold: the
# threshold measure above says "5+ cards 86%", this says which lengths the hands actually held ("4-6, median 5").
def _held_bid(X, calls, ctx):
    return np.array([X[i, F[f"len_{bid_suit(c)}"]] if bid_suit(c) else -1 for i, c in enumerate(calls)])


def _held_suit(s):
    return lambda X, calls, ctx: X[:, F[f"len_{s}"]]


def _held_partners(X, calls, ctx):
    return np.array([X[i, F[f"len_{strain(cx[-2])}"]] for i, cx in enumerate(ctx)])


def _held_theirs(X, calls, ctx):
    out = []
    for i, c in enumerate(ctx):
        s = next((strain(x) for x in reversed(c) if strain(x) in SUITS), None)
        out.append(X[i, F[f"len_{s}"]] if s else -1)
    return np.array(out)


def _held_unbid_major(X, calls, ctx):
    out = []
    for i, c in enumerate(ctx):
        bid = {strain(x) for x in c}
        majors = [s for s in "HS" if s not in bid]
        out.append(min(X[i, F[f"len_{s}"]] for s in majors) if majors else -1)
    return np.array(out)


_LONGEST = ("longest suit", lambda X, c, x: X[:, F["longest"]])
_BID = ("the bid suit", _held_bid)
HELD = {  # (group, label) -> (which suit, lengths per decision; -1 where it has none)
    ("Openings", "1NT (1st/2nd seat)"): _LONGEST,
    ("Openings", "2NT (1st/2nd seat)"): _LONGEST,
    ("Openings", "1♥ / 1♠ (1st/2nd seat)"): _BID,
    ("Openings", "1♦ (1st/2nd seat)"): _BID,
    ("Openings", "1♣ (1st/2nd seat)"): _BID,
    ("Openings", "1-level suit opening, 3rd seat"): _BID,
    ("Openings", "Pass (1st/2nd seat)"): _LONGEST,
    ("Openings", "2♣"): _LONGEST,
    ("Openings", "2♦ / 2♥ / 2♠"): _BID,
    ("Openings", "3-level preempt"): _BID,
    ("Over partner's 1NT", "2♣"): ("longer major", lambda X, c, x: np.maximum(X[:, F["len_H"]], X[:, F["len_S"]])),
    ("Over partner's 1NT", "2♦"): ("hearts", _held_suit("H")),
    ("Over partner's 1NT", "2♥"): ("spades", _held_suit("S")),
    ("Over partner's 1NT", "3NT"): ("longer major", lambda X, c, x: np.maximum(X[:, F["len_H"]], X[:, F["len_S"]])),
    ("Over partner's 1♥/1♠", "1NT"): ("partner's major", _held_partners),
    ("Over partner's 1♥/1♠", "Single raise"): ("partner's major", _held_bid),
    ("Over partner's 1♥/1♠", "Jump raise to 3"): ("partner's major", _held_bid),
    ("Over partner's 1♥/1♠", "Raise to game"): ("partner's major", _held_bid),
    ("Over partner's 1♥/1♠", "2NT"): ("partner's major", _held_partners),
    ("Over the opponents' 1-level opening", "1-level suit overcall"): _BID,
    ("Over the opponents' 1-level opening", "2-level suit overcall"): _BID,
    ("Over the opponents' 1-level opening", "1NT overcall"): ("their suit", _held_theirs),
    ("Over the opponents' 1-level opening", "Takeout double"): ("their suit", _held_theirs),
    ("After partner's 1-level opening and an overcall", "Negative double"): ("shorter unbid major", _held_unbid_major),
}
assert set(HELD) == {(g, l) for g, l, *_ in HEADLINE}, "every headline entry names the suit whose length it reports"

# Entries whose calls mean something else in some of their auctions: 4H over 1S is a splinter and 2H over 1S a new
# suit, not raises (counted as raises, WBridge5's "raise to game" holds 0 trumps at its 5th percentile), and
# 2C over 1C is a cue-bid, not an overcall. (call, auction) -> whether the decision belongs to the entry.
ONLY = {
    ("Over partner's 1♥/1♠", "Single raise"): lambda c, cx: bid_suit(c) == strain(cx[-2]),
    ("Over partner's 1♥/1♠", "Jump raise to 3"): lambda c, cx: bid_suit(c) == strain(cx[-2]),
    ("Over partner's 1♥/1♠", "Raise to game"): lambda c, cx: bid_suit(c) == strain(cx[-2]),
    ("Over the opponents' 1-level opening", "2-level suit overcall"): lambda c, cx: bid_suit(c) != strain(cx[-1]),
}


def _suit_profiles(X: np.ndarray, calls: List[str], ctx: List[List[str]]) -> List[dict]:
    """All four suits' lengths per (auction, call), most frequent first: pooled over auctions, a takeout double's
    short suit (their suit) averages away, so each auction -- the calls before it, passes left out -- stays apart."""
    by = collections.defaultdict(list)
    for i, (c, cx) in enumerate(zip(calls, ctx)):
        by[(" ".join(x for x in cx if x != "P"), c)].append(i)
    out = []
    for (after, c), idx in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(idx) < 20:
            continue
        L = X[idx][:, [F[f"len_{s}"] for s in SUITS]]
        out.append({"after": after, "call": c, "n": len(idx),
                    "mean": {s: round(float(L[:, j].mean()), 2) for j, s in enumerate(SUITS)},
                    "range": {s: [int(v) for v in np.percentile(L[:, j], [5, 50, 95], method="nearest")]
                              for j, s in enumerate(SUITS)}})
    return out


def _length_stats(v: np.ndarray) -> dict:
    v = v[v >= 0]
    return {"range": [int(x) for x in np.percentile(v, [5, 50, 95], method="nearest")],
            "dist": [int(x) for x in np.bincount(v.astype(int), minlength=14)]} if len(v) else {}


def headline(data: dict, groups: Dict[str, np.ndarray]) -> List[dict]:
    out = []
    for group, label, match, calls, mtext, measure in HEADLINE:
        rows = np.concatenate([idx for ctx, idx in groups.items() if ctx != BLACKWOOD and match(ctx.split() if ctx else [])]
                              or [np.zeros(0, np.int64)])
        chosen = rows[np.isin(data["y"][rows], [BIDS.index(c) for c in calls])]
        if (group, label) in ONLY:
            keep = ONLY[(group, label)]
            chosen = chosen[[keep(BIDS[data["y"][i]], data["ctx"][i].split()) for i in chosen]] if len(chosen) else chosen
        e = {"group": group, "label": label, "calls": calls, "decisions": int(len(rows)), "n": int(len(chosen)),
             "frequency": len(chosen) / len(rows) if len(rows) else None}
        if len(chosen) >= 20:
            X = data["X"][chosen]
            e["hcp"] = [int(v) for v in np.percentile(X[:, F["hcp"]], [5, 95], method="nearest")]
            if label.startswith("Pass"):
                e["hcp"] = [0, int(np.percentile(X[:, F["hcp"]], 95, method="nearest"))]
            cs = [BIDS[a] for a in data["y"][chosen]]
            cx = [data["ctx"][i].split() for i in chosen]
            if measure is not None:
                e["measure"], e["measure_share"] = mtext, float(np.mean(measure(X, cs, cx)))
            of, held = HELD[(group, label)]
            v = held(X, cs, cx)
            e["held"] = {"of": of, **_length_stats(v)}
            e["suit_lengths"] = {s: [int(x) for x in np.percentile(X[:, F[f"len_{s}"]], [5, 50, 95], method="nearest")]
                                 for s in SUITS}
            e["suit_profiles"] = _suit_profiles(X, cs, cx)
            if len(calls) > 1:   # a pooled entry (1H / 1S, weak twos...): the length held per call
                e["held_by_call"] = {c: {"n": int(sum(x == c for x in cs)), **_length_stats(v[[x == c for x in cs]])}
                                     for c in calls if sum(x == c for x in cs) >= 20}
        out.append(e)
    bw = groups.get(BLACKWOOD)
    if bw is not None and len(bw):
        for c in ["5C", "5D", "5H", "5S"]:
            rows = bw[data["y"][bw] == BIDS.index(c)]
            dist = np.bincount(data["X"][rows, F["aces"]], minlength=5)
            out.append({"group": "Blackwood (partner's 4NT after a suit of ours)", "label": f"{call_text(c)} reply",
                        "calls": [c], "decisions": int(len(bw)), "n": int(len(rows)), "aces": [int(v) for v in dist]})
    return out


# ---------------------------------------------------------------- one agent's card
def full_groups(data: dict) -> Dict[str, np.ndarray]:
    g = collections.defaultdict(list)
    for i, c in enumerate(data["ctx"]):
        g[c].append(i)
    return {k: np.array(v) for k, v in g.items()}


def assign(data: dict) -> Dict[str, np.ndarray]:
    """key -> rows: each decision goes to its most specific key with enough decisions to fit on
    (deals in the fitting half), the Blackwood pool additionally; the rest to no key."""
    ks = [keys(c.split() if c else []) for c in data["ctx"]]
    fit_half = data["deal"] % 2 == 0
    out, todo = {}, np.arange(len(ks))
    for t in range(len(TIERS)):
        cnt = collections.Counter(ks[i][t] for i in todo if fit_half[i])
        g = collections.defaultdict(list)
        rest = []
        for i in todo:
            (g[ks[i][t]] if cnt[ks[i][t]] >= MIN_CONTEXT else rest).append(i)
        out.update({k: np.array(v) for k, v in g.items()})
        todo = rest
    bw = [i for i, c in enumerate(data["ctx"]) if blackwood_ask(c.split() if c else [])]
    if sum(fit_half[i] for i in bw) >= MIN_CONTEXT:
        out[BLACKWOOD] = np.array(bw)
    return out


def build_card(data: dict) -> dict:
    total = len(data["y"])
    contexts, covered = {}, collections.Counter()
    for ctx, idx in sorted(assign(data).items(), key=lambda kv: -len(kv[1])):
        X, y, deal = data["X"][idx], data["y"][idx], data["deal"][idx]
        fit = fit_tree(X, y, deal)
        if fit is None:
            continue
        prefix, tier = ([BLACKWOOD], "auction") if ctx == BLACKWOOD else key_prefix(ctx)
        if ctx != BLACKWOOD:
            covered[tier] += len(idx)
        te = fit.pop("test_rows")
        sec = section(prefix) if tier == "auction" else f"Rarer auctions, by {'their last round' if tier == 'last round' else 'the last two calls'}"
        contexts[ctx] = {"auction": auction_text(prefix, tier), "section": sec, "tier": tier, "n": int(len(idx)),
                         "share_of_decisions": len(idx) / total, "depth": fit["depth"], "test_n": fit["test_n"],
                         "fidelity": fit["test_acc"], "majority_baseline": fit["baseline"],
                         "rules": rules(fit["tree"], X[te], y[te]), "tree": tree_json(fit["tree"]),
                         "shows": profile(X, y)}
    exact = [c for k, c in contexts.items() if k != BLACKWOOD]
    fid_n = sum(c["test_n"] for c in exact)
    return {"decisions": total, "contexts": contexts, "coverage": sum(covered.values()) / total if total else 0.0,
            "coverage_by_tier": {t: covered[t] / total for t in TIERS},
            "fidelity": sum(c["fidelity"] * c["test_n"] for c in exact if c["fidelity"] is not None) / max(fid_n, 1),
            "rules": sum(len(c["rules"]) for c in contexts.values()), "headline": headline(data, full_groups(data))}


def compare_to(card: dict, data: dict, ref: dict) -> None:
    """Apply the reference (WBridge5) card's tree to this agent's held-out decisions, per auction."""
    groups = assign(data)
    num = den = 0
    for ctx, e in card["contexts"].items():
        r = ref["contexts"].get(ctx)
        if r is None:
            e["wbridge5_fidelity"] = None
            continue
        idx = groups.get(ctx, np.zeros(0, np.int64))
        te = idx[data["deal"][idx] % 4 == 3]
        if not len(te):
            e["wbridge5_fidelity"] = None
            continue
        acc = float(np.mean(predict(tree_from_json(r["tree"]), data["X"][te]) == data["y"][te]))
        e["wbridge5_fidelity"] = acc
        e["differs"] = bool(e["fidelity"] is not None and e["fidelity"] - acc >= DIFFERS_GAP and len(te) >= 50)
        if ctx != BLACKWOOD:
            num += acc * len(te)
            den += len(te)
    card["wbridge5_fidelity"] = num / den if den else None
    card["contexts_differing"] = sum(bool(e.get("differs")) for e in card["contexts"].values())


# ---------------------------------------------------------------- HTML
CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e1db;--card:#fff;--acc:#2f5d8a;--red:#b3261e;--warn:#fff3d6;--warnline:#e2b649}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#17181a;--fg:#ececec;--mut:#9a9aa0;--line:#33353a;--card:#1f2124;--acc:#8fb7e0;--red:#ff8a80;--warn:#3a3220;--warnline:#8a6d22}}
:root[data-theme=dark]{--bg:#17181a;--fg:#ececec;--mut:#9a9aa0;--line:#33353a;--card:#1f2124;--acc:#8fb7e0;--red:#ff8a80;--warn:#3a3220;--warnline:#8a6d22}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.6rem;margin:.2em 0}h2{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.2em}
.mut{color:var(--mut)}.r{color:var(--red)}a{color:var(--acc)}
table{border-collapse:collapse;width:100%;margin:.5em 0}td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;font-size:.85rem;color:var(--mut)}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:6px 12px}
details.diff{border-color:var(--warnline);background:var(--warn)}summary{cursor:pointer;font-weight:600}summary .mut{font-weight:400}
.call{font-weight:600;white-space:nowrap}.stats{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0}.stats div{min-width:140px}.stats b{display:block;font-size:1.4rem}
code{font-size:.9em}
"""


def _suits_html(s: str) -> str:
    s = html.escape(s)
    return re.sub("([♥♦])", r'<span class="r">\1</span>', s)


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _shows(c: str, p: dict) -> str:
    bits = [f"{p['hcp'][0]}–{p['hcp'][2]} HCP"]
    s = bid_suit(c)
    if s:
        lo, _, hi = p["len"][s]
        bits.append(f"{SUIT_SYM[s]}{lo}–{hi}")
    if strain(c) == "NT" or c == "P":
        bits.append(f"balanced {_pct(p['balanced'])}")
    return ", ".join(bits)


def _headline_val(e: dict) -> str:
    if "aces" in e:
        tot = sum(e["aces"]) or 1
        parts = [f"{k} {'ace' if k == 1 else 'aces'} {100 * v / tot:.0f}%" for k, v in enumerate(e["aces"]) if v / tot >= 0.05]
        return f"{', '.join(parts)} <span class=mut>(n {e['n']})</span>" if e["n"] >= 10 else f"<span class=mut>too few ({e['n']})</span>"
    if "hcp" not in e:
        return "<span class=mut>not played</span>" if e["n"] == 0 else f"<span class=mut>too few ({e['n']})</span>"
    lab = "up to " + str(e["hcp"][1]) if e["label"].startswith("Pass") else f"{e['hcp'][0]}–{e['hcp'][1]}"
    m = f" · {e['measure']} {_pct(e['measure_share'])}" if e.get("measure") else ""
    r = e.get("held", {}).get("range")
    held = f" · {e['held']['of']} {r[0]}–{r[2]} (median {r[1]})" if r else ""
    return f"{lab} HCP{m}{held} <span class=mut>(n {e['n']}, {_pct(e['frequency'])} of these auctions)</span>"


def agent_html(name: str, card: dict, ref: Optional[dict]) -> str:
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>{html.escape(name)} convention card</title><style>{CSS}</style></head><body><main>",
         "<p><a href='index.html'>← all models</a></p>", f"<h1>{_suits_html(name)} — convention card from play</h1>",
         f"<p class=mut>{html.escape(card['source'])}. Rules are decision trees fitted on half the deals; "
         "“holds” is the share of the other deals' decisions where the rule's call was the model's call. "
         "Ranges in “shows” are the 5th–95th percentile of the hands that made the call.</p>",
         "<div class=stats>",
         f"<div><b>{card['decisions']:,}</b>decisions</div>",
         f"<div><b>{_pct(card['coverage'])}</b>in auctions with a tree "
         f"<span class=mut>({_pct(card['coverage_by_tier']['auction'])} by the whole auction)</span></div>",
         f"<div><b>{_pct(card['fidelity'])}</b>predicted by this card</div>"]
    if card.get("wbridge5_fidelity") is not None and name != WB5:
        h.append(f"<div><b>{_pct(card['wbridge5_fidelity'])}</b>predicted by WBridge5's card</div>"
                 f"<div><b>{card['contexts_differing']}</b>auctions played differently</div>")
    h.append(f"<div><b>{card['rules']:,}</b>rules in {len(card['contexts'])} auctions</div></div>")
    # headline
    h.append("<h2>At a glance</h2><div class=scroll><table><tr><th></th><th>Call</th><th>This model</th>"
             + ("<th>WBridge5</th>" if ref and name != WB5 else "") + "</tr>")
    refh = {(e["group"], e["label"]): e for e in (ref or {}).get("headline", [])}
    group = None
    for e in card["headline"]:
        g = html.escape(e["group"]) if e["group"] != group else ""
        group = e["group"]
        h.append(f"<tr><td class=mut>{g}</td><td class=call>{_suits_html(e['label'])}</td><td>{_headline_val(e)}</td>"
                 + (f"<td>{_headline_val(refh[(e['group'], e['label'])])}</td>" if ref and name != WB5 and (e['group'], e['label']) in refh else "")
                 + "</tr>")
    h.append("</table></div>")
    # sections
    by = collections.defaultdict(list)
    for ctx, e in card["contexts"].items():
        by[e["section"]].append((ctx, e))
    for sec in SECTIONS:
        items = sorted(by.get(sec, []), key=lambda kv: -kv[1]["n"])
        if not items:
            continue
        nd = sum(bool(e.get("differs")) for _, e in items)
        h.append(f"<h2>{html.escape(sec)} <span class=mut>({len(items)} auctions"
                 + (f", {nd} played differently from WBridge5" if nd and name != WB5 else "") + ")</span></h2>")
        for ctx, e in items:
            diff = " class=diff" if e.get("differs") else ""
            wb = (f" · WBridge5's card {_pct(e['wbridge5_fidelity'])}" if e.get("wbridge5_fidelity") is not None and name != WB5 else "")
            h.append(f"<details{diff}><summary><code>{_suits_html(e['auction'])}</code> <span class=mut>"
                     f"{e['n']:,} decisions · this card predicts {_pct(e['fidelity'])}{wb}"
                     f" (always the commonest call: {_pct(e['majority_baseline'])})</span></summary>")
            h.append("<div class=scroll><table><tr><th>Call</th><th>When</th><th class=num>share</th><th class=num>holds</th></tr>")
            for r in e["rules"]:
                when = ", ".join(r["when"]) or "otherwise / always"
                h.append(f"<tr><td class=call>{_suits_html(call_text(r['call']))}</td><td>{_suits_html(when)}</td>"
                         f"<td class=num>{_pct(r['share'])}</td><td class=num>{_pct(r['holds'])}"
                         f"<span class=mut> ({r['test_n']})</span></td></tr>")
            h.append("</table></div><p class=mut style='margin:.4em 0 .2em'>What each call shows here:</p>"
                     "<div class=scroll><table><tr><th>Call</th><th class=num>share</th><th>shows</th></tr>")
            for c, p in sorted(e["shows"].items(), key=lambda kv: -kv[1]["n"]):
                if p["share"] < 0.01:
                    continue
                h.append(f"<tr><td class=call>{_suits_html(call_text(c))}</td><td class=num>{_pct(p['share'])}</td>"
                         f"<td>{_suits_html(_shows(c, p))}</td></tr>")
            h.append("</table></div>")
            if ref and name != WB5 and ctx in ref["contexts"]:
                rr = ref["contexts"][ctx]["rules"]
                h.append("<details><summary class=mut>WBridge5's rules in this auction</summary><table>")
                for r in rr:
                    h.append(f"<tr><td class=call>{_suits_html(call_text(r['call']))}</td>"
                             f"<td>{_suits_html(', '.join(r['when']) or 'otherwise / always')}</td>"
                             f"<td class=num>{_pct(r['share'])}</td></tr>")
                h.append("</table></details>")
            h.append("</details>")
    h.append("</main></body></html>")
    return "\n".join(h)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def index_html(cards: Dict[str, dict]) -> str:
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Convention cards from play</title><style>{CSS}</style></head><body><main>",
         "<h1>Convention cards from play</h1>",
         "<p class=mut>Each model's convention, learned from its own self-play with one decision tree per auction "
         "(hand features → call). <b>Predicted by own card</b>: held-out decisions where the card's call is the model's call, "
         "in the auctions frequent enough to have a tree (<b>coverage</b>). <b>Predicted by WBridge5's card</b>: the same, "
         "using the card learned from WBridge5's games. <b>Played differently</b>: auctions where the model's own card "
         f"beats WBridge5's by {100 * DIFFERS_GAP:.0f}+ points.</p>",
         "<div class=scroll><table><tr><th>Model</th><th class=num>decisions</th><th class=num>coverage</th>"
         "<th class=num>predicted by own card</th><th class=num>by WBridge5's card</th><th class=num>auctions</th>"
         "<th class=num>played differently</th><th class=num>rules</th></tr>"]
    for name, c in cards.items():
        h.append(f"<tr><td><a href='{slug(name)}.html'>{html.escape(name)}</a></td><td class=num>{c['decisions']:,}</td>"
                 f"<td class=num>{_pct(c['coverage'])}</td><td class=num>{_pct(c['fidelity'])}</td>"
                 f"<td class=num>{_pct(c.get('wbridge5_fidelity')) if name != WB5 else '–'}</td>"
                 f"<td class=num>{len(c['contexts'])}</td><td class=num>{c.get('contexts_differing', 0) if name != WB5 else '–'}</td>"
                 f"<td class=num>{c['rules']:,}</td></tr>")
    h.append("</table></div>")
    h.append("<h2>Headline agreements</h2><div class=scroll><table><tr><th>Call</th>"
             + "".join(f"<th>{html.escape(n)}</th>" for n in cards) + "</tr>")
    keys = [(e["group"], e["label"]) for e in next(iter(cards.values()))["headline"]]
    for k in keys:
        row = [f"<td class=call>{_suits_html(k[1])}<div class=mut style='font-weight:400'>{html.escape(k[0])}</div></td>"]
        for c in cards.values():
            e = next((x for x in c["headline"] if (x["group"], x["label"]) == k), None)
            row.append(f"<td>{_headline_val(e) if e else '–'}</td>")
        h.append("<tr>" + "".join(row) + "</tr>")
    h.append("</table></div></main></body></html>")
    return "\n".join(h)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", default=list(MODELS), help="names from evaluation.common.MODELS")
    ap.add_argument("--checkpoint", nargs="*", default=[], metavar="NAME=PATH", help="further checkpoints to describe")
    ap.add_argument("--runs", nargs="*", default=[], metavar="DIR",
                    help="PPO run directories (e.g. runs/ppo/*): each run's final checkpoint, named after the run")
    ap.add_argument("--deals", type=int, default=30000, help="self-play deals per model")
    ap.add_argument("--wb5_deals", type=int, default=30000, help="WBridge5 deals from the prepared train split")
    ap.add_argument("--random_vulnerability", action="store_true", help="self-play with pgx's vulnerability (default: nobody)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    settings = {"version": VERSION, "deals": args.deals, "seed": args.seed, "random_vulnerability": args.random_vulnerability}

    def cached(name, path):
        f = out / f"{slug(name)}.json"
        if not f.exists():
            return None
        c = json.loads(f.read_text())
        if c.get("settings") == settings and (path is None or same_checkpoint(c, path)):
            return c
        return None

    cards = {}
    wb5_settings = {"version": VERSION, "deals": args.wb5_deals}
    f = out / f"{slug(WB5)}.json"
    wb5 = json.loads(f.read_text()) if f.exists() else None
    if not wb5 or wb5.get("settings") != wb5_settings:
        pos = load_test_split(TEST_SPLIT.parent / "train", max_deals=args.wb5_deals)
        wb5 = build_card(collect(pos))
        wb5.update(settings=wb5_settings, source=f"WBridge5's own calls on {args.wb5_deals:,} deals of the prepared train split "
                   "(non-vulnerable, dealer North)")
        f.write_text(json.dumps(wb5))
        print(f"[trees] {WB5}: {wb5['decisions']:,} decisions, coverage {wb5['coverage']:.1%}, "
              f"fidelity {wb5['fidelity']:.1%}, {wb5['rules']} rules ({time.time() - t0:.0f}s)", flush=True)
    cards[WB5] = wb5
    from evaluation.common import converged_checkpoint, last_checkpoint
    agents = {a: MODELS[a] for a in args.agents}
    agents.update(dict(kv.split("=", 1) for kv in args.checkpoint))
    for r in map(Path, args.runs):
        if (r / "ppo").is_dir():
            agents[f"PPO run {r.name}"] = converged_checkpoint(r) or last_checkpoint(r / "ppo")
    for name, path in agents.items():
        if not Path(path).exists():
            print(f"[trees] {name}: no checkpoint at {path}, skipped", flush=True)
            continue
        c = cached(name, path)
        if c is None:
            data = collect(selfplay(path, args.deals, seed=args.seed, random_vulnerability=args.random_vulnerability))
            c = build_card(data)
            compare_to(c, data, wb5)
            c.update(settings=settings, **checkpoint_stamp(path),
                     source=f"self-play of {Path(path).name} at all four seats, {args.deals:,} held-out deals, greedy calls, "
                     f"{'random' if args.random_vulnerability else 'no'} vulnerability")
            (out / f"{slug(name)}.json").write_text(json.dumps(c))
        cards[name] = c
        print(f"[trees] {name}: {c['decisions']:,} decisions, coverage {c['coverage']:.1%}, own card {c['fidelity']:.1%}, "
              f"WBridge5's {c['wbridge5_fidelity']:.1%}, {c['contexts_differing']} auctions differ ({time.time() - t0:.0f}s)",
              flush=True)
    for name, c in cards.items():
        (out / f"{slug(name)}.html").write_text(agent_html(name, c, None if name == WB5 else wb5))
    (out / "index.html").write_text(index_html(cards))
    print(f"[done] {len(cards)} cards -> {out}/index.html ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
