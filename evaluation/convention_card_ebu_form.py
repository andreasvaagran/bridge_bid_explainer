#!/usr/bin/env python3
"""A model's convention card on the EBU 20B form, filled in from its own play (convention_cards_from_play.py's
profiles). Rare tails are left off, the way a player leaves exceptions off a card. One A4 page per model, as pdf
and png."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from evaluation.convention_card_figures import load_cards  # noqa: E402
from evaluation.convention_cards_from_play import BLACKWOOD, slug  # noqa: E402

EXCEPTION = 0.05    # a tail holding at most this share of the hands is left off the card (the 5th-95th
                    # percentile of the other cards and the thesis: a range holds for 90% of the hands)
LISTED = 0.03       # a call is listed when made in at least this share of the auction's decisions
SHAPE_SHARE = 0.02  # a 1NT shape is listed when this share of the openings has it
NAMED = 0.75        # a convention is named when this share of the hands fits its definition
MIN_N = 20          # fewer calls than this: too few to say anything
SUITS = "SHDC"
SYM = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
STRAIN_ORDER = "CDHSN"


# ---------------------------------------------------------------- statistics of one call in one auction
def call_text(c: str) -> str:
    if c == "X":
        return "Dbl"
    if c == "XX":
        return "Rdbl"
    return c[:-1] + SYM[c[-1]] if len(c) == 2 and c[-1] in SYM else c


def strain(c: str) -> Optional[str]:
    return None if c in ("P", "X", "XX") else c[1:].replace("NT", "N")


class Profile:
    """The hands with which a model made one call in one auction."""

    def __init__(self, p: dict):
        self.p, self.n = p, p["n"]
        self.share = p.get("share", 0.0)

    def _trim(self, dist: List[int]) -> Tuple[int, int]:
        tot = sum(dist)
        lo, acc = 0, 0
        while lo < len(dist) - 1 and (acc + dist[lo]) / tot <= EXCEPTION:
            acc += dist[lo]
            lo += 1
        hi, acc = len(dist) - 1, 0
        while hi > lo and (acc + dist[hi]) / tot <= EXCEPTION:
            acc += dist[hi]
            hi -= 1
        return lo, hi

    def hcp(self) -> Tuple[int, int]:
        if "hcp_dist" in self.p:
            return self._trim(self.p["hcp_dist"])
        return self.p["hcp"][0], self.p["hcp"][2]

    def length(self, s: str) -> Tuple[int, int]:
        if "len_dist" in self.p:
            return self._trim(self.p["len_dist"][s])
        r = self.p["len"][s]
        return r[0], r[2]

    def at_least(self, s: str, k: int) -> float:
        d = self.p.get("len_dist", {}).get(s)
        if not d:
            return 1.0 if self.p["len"][s][0] >= k else 0.0
        return sum(d[k:]) / sum(d)

    def at_most(self, s: str, k: int) -> float:
        return 1.0 - self.at_least(s, k + 1)

    def balanced(self) -> float:
        return self.p.get("balanced", 0.0)

    def shapes(self) -> Dict[str, float]:
        sh = self.p.get("shapes", {})
        return {k: v / self.n for k, v in sh.items()}


def hcp_text(lo: int, hi: int) -> str:
    return f"{lo}+" if hi >= 30 else f"{lo}–{hi}"


def describe(pr: Profile, their: Optional[str] = None, skip: str = "") -> str:
    """What the hands hold, the way a card says it: "8-15, 5+H, short C"."""
    lo, hi = pr.hcp()
    parts = [hcp_text(lo, hi)]
    long_ = [s for s in SUITS if s not in skip and pr.length(s)[0] >= 4]
    parts += [f"{pr.length(s)[0]}+{SYM[s]}" for s in long_]
    if their and pr.at_most(their, 2) >= NAMED and their not in long_:
        parts.append(f"≤2{SYM[their]}")
    short = [s for s in SUITS if s != their and s not in long_ and pr.length(s)[1] <= 1]
    parts += [f"short {SYM[s]}" for s in short]
    if not long_ and pr.balanced() >= NAMED:
        parts.append("bal")
    return ", ".join(parts)


# ---------------------------------------------------------------- reading a learned card
class Learned:
    def __init__(self, name: str, card: dict):
        self.name, self.card = name, card
        self.ctx = {k: c for k, c in card.get("contexts", {}).items() if c.get("tier") == "auction" or k == BLACKWOOD}

    def prof(self, key: str, call: str) -> Optional[Profile]:
        p = self.ctx.get(key, {}).get("shows", {}).get(call)
        return Profile(p) if p and p["n"] >= MIN_N else None

    def calls(self, key: str, min_share: float = LISTED) -> List[Tuple[str, Profile]]:
        shows = self.ctx.get(key, {}).get("shows", {})
        out = [(c, Profile(p)) for c, p in shows.items() if p["n"] >= MIN_N and p["share"] >= min_share]
        order = {c: i for i, c in enumerate(["P", "X", "XX"])}
        return sorted(out, key=lambda cp: (order.get(cp[0], 3), int(cp[0][0]) if cp[0][0].isdigit() else 0,
                                           STRAIN_ORDER.find(strain(cp[0]) or "C")))

    def headline(self, label: str) -> dict:
        return next((h for h in self.card.get("headline", []) if h["label"] == label), {})


# ---------------------------------------------------------------- naming conventions from the hands
def name_response_to_1nt(call: str, pr: Profile, major4: Optional[float] = None) -> str:
    """major4: share of the 2C hands with a four-card major (either one; the headline card measures it)."""
    if call == "2C" and (major4 if major4 is not None else max(pr.at_least("H", 4), pr.at_least("S", 4))) >= NAMED:
        return f"Stayman ({describe(pr, skip='SHDC')})"
    nxt = {"2D": "H", "2H": "S", "2S": "C", "2NT": "D", "4D": "H", "4H": "S"}
    if call in nxt and pr.at_least(nxt[call], 5) >= NAMED:
        kind = "Texas" if call[0] == "4" else "transfer"
        return f"{kind} to {SYM[nxt[call]]} ({hcp_text(*pr.hcp())})"
    if call == "3NT":
        return f"to play, {hcp_text(*pr.hcp())}"
    return describe(pr)


def name_overcall(their: str, call: str, pr: Profile) -> str:
    s = strain(call)
    their = their if their in SUITS else None
    if call == "X":
        if their and pr.at_most(their, 2) >= 0.6:
            return f"take-out, {hcp_text(*pr.hcp())}"
        return describe(pr, their)
    if their and s == their:   # cue-bid
        two = [x for x in SUITS if x != their and pr.at_least(x, 5) >= NAMED]
        if len(two) >= 2:
            name = "Michaels" if (their in "CD" and set(two) >= {"H", "S"}) or (their in "HS" and len(two) >= 2) else "two-suited"
            return f"{name}: {'+'.join(SYM[x] for x in two)}, {hcp_text(*pr.hcp())}"
        return describe(pr, their)
    if call == "2NT" and their:
        low = [x for x in "CDH" if x != their][:2]
        if all(pr.at_least(x, 5) >= NAMED for x in low):
            return f"Unusual: {'+'.join(SYM[x] for x in low)}, {hcp_text(*pr.hcp())}"
    return describe(pr, their)


def blackwood_replies(lc: Learned) -> Tuple[str, str]:
    shows = lc.ctx.get(BLACKWOOD, {}).get("shows", {})
    reps = {}
    for c in ("5C", "5D", "5H", "5S"):
        a = shows.get(c, {}).get("aces")
        if a and sum(a) >= MIN_N:
            reps[c] = "/".join(str(k) for k, v in enumerate(a) if v / sum(a) >= 0.1)
    if not reps:
        return "—", "—"
    text = ", ".join(f"{call_text(c)} = {v}" for c, v in reps.items())
    name = ("Blackwood (0/3, 1/4)" if reps.get("5C") == "0/3" else
            "Blackwood (0/4, 1, 2, 3)" if reps.get("5C") in ("0/4", "0") else "4NT ace ask")
    return name, text + " aces"


# ---------------------------------------------------------------- the card's contents
def is_jump(prev: str, call: str) -> bool:
    """Whether call skips a level over prev (both suit bids)."""
    lv = lambda c: (int(c[0]), STRAIN_ORDER.find(strain(c)))  # noqa: E731
    lp, sp = lv(prev)
    lc, sc = lv(call)
    cheapest = lp if sc > sp else lp + 1
    return lc > cheapest


def fill(lc: Learned) -> dict:
    """Every field of the card, as text."""
    out: dict = {"notes": []}
    op = lambda c: lc.prof("", c)  # noqa: E731
    # -------- 1NT
    nt = op("1NT")
    if nt:
        lo, hi = nt.hcp()
        out["nt_strength"] = hcp_text(lo, hi)
        odd = [k for k, v in sorted(nt.shapes().items(), key=lambda kv: -kv[1])
               if k not in ("4333", "4432", "5332") and v >= SHAPE_SHARE]
        out["nt_shapes"] = ("balanced" + (", also " + " ".join(odd) if odd else "")) if nt.shapes() else "—"
        out["nt_singleton"] = any(k[-1] in "01" and v >= SHAPE_SHARE for k, v in nt.shapes().items())
    else:
        out["nt_strength"], out["nt_shapes"], out["nt_singleton"] = "—", "—", False
    resp = dict(lc.calls("1NT P"))
    major4 = next((h.get("measure_share") for h in lc.card.get("headline", [])
                   if h["group"] == "Over partner's 1NT" and h["label"] == "2\u2663"), None)
    out["nt_resp"] = {c: name_response_to_1nt(c, resp[c], major4) if c in resp else "—"
                      for c in ("2C", "2D", "2H", "2S", "2NT")}
    out["nt_3l"] = "; ".join(f"{call_text(c)} {name_response_to_1nt(c, p)}" for c, p in resp.items() if c[0] == "3") or "—"
    out["nt_4l"] = "; ".join(f"{call_text(c)} {name_response_to_1nt(c, p)}" for c, p in resp.items() if c[0] == "4") or "—"
    after_x = [(c, p) for c, p in lc.calls("1NT X") if c != "P"]
    out["nt_after_x"] = "; ".join(f"{call_text(c)} = {describe(p)}" for c, p in after_x[:3]) or "—"
    # -------- general description
    maj = [p for p in (op("1H"), op("1S")) if p]
    five = bool(maj) and all(p.length(s)[0] >= 5 for p, s in zip(maj, "HS"))
    words = []
    if nt:
        words.append(("STRONG" if nt.hcp()[0] >= 15 else "WEAK" if nt.hcp()[1] <= 14 else "") + f" 1NT ({out['nt_strength']})")
    words.append("5-CARD MAJORS" if five else "4-CARD MAJORS")
    two_c = op("2C")
    if two_c and two_c.hcp()[0] >= 15:
        words.append("STRONG 2♣")
    weak = [p for p in (op("2D"), op("2H"), op("2S")) if p]
    if weak and all(p.hcp()[1] <= 11 for p in weak):
        words.append("WEAK TWOS")
    out["general"] = "   ".join(w.strip() for w in words)
    # -------- two-level openings
    two = {}
    for c in ("2C", "2D", "2H", "2S", "2NT"):
        p = op(c)
        if not p:
            two[c] = ("—", "—")
            continue
        s = strain(c)
        meaning = describe(p)
        if c == "2C" and p.hcp()[0] >= 15:
            meaning = f"strong, {hcp_text(*p.hcp())}"
        elif s in "DHS" and p.hcp()[1] <= 11 and p.length(s)[0] >= 5:
            meaning = f"weak, {hcp_text(*p.hcp())}, {p.length(s)[0]}+{SYM[s]}"
        rs = [(r, q) for r, q in lc.calls(f"{c} P") if r != "P"][:3]
        two[c] = (meaning, "; ".join(f"{call_text(r)} = {describe(q)}" for r, q in rs) or "—")
    out["two"] = two
    # -------- other opening bids
    rows = []
    third = lc.headline("1-level suit opening, 3rd seat").get("hcp")
    first = lc.headline("1♥ / 1♠ (1st/2nd seat)").get("hcp")
    light3 = bool(third and first and third[0] <= first[0] - 2)
    for c in ("1C", "1D", "1H", "1S"):
        p = op(c)
        if not p:
            rows.append([call_text(c), "—", "", "—", "—", "—"])
            continue
        s = strain(c)
        mn = p.length(s)[0]
        meaning = "natural" if mn >= 3 else f"may be {mn} (bal)" if mn == 2 and p.balanced() >= 0.4 else f"may be {mn}"
        rs = [(r, q) for r, q in lc.calls(f"{c} P") if r not in ("P",)]
        named = []
        for r, q in rs:
            if r == "2NT" and s in "HS" and q.at_least(s, 4) >= NAMED and q.hcp()[0] >= 11:
                named.append(f"2NT Jacoby ({hcp_text(*q.hcp())})")
            elif r == "1NT":
                named.append(f"1NT {hcp_text(*q.hcp())}")
            elif strain(r) == s and r[0] in "234":
                named.append(f"{call_text(r)} {hcp_text(*q.hcp())}")
        rows.append([call_text(c), hcp_text(*p.hcp()), "✓" if light3 else "", str(mn), meaning,
                     "; ".join(named[:4]) or "natural"])
    for label, calls in (("3 bids", ("3C", "3D", "3H", "3S")), ("4 bids", ("4C", "4D", "4H", "4S"))):
        ps = [(c, op(c)) for c in calls if op(c)]
        if ps:
            lo = min(p.hcp()[0] for _, p in ps)
            hi = max(p.hcp()[1] for _, p in ps)
            mn = min(p.length(strain(c))[0] for c, p in ps)
            rows.append([label, hcp_text(lo, hi), "", str(mn), "pre-emptive" if hi <= 12 else "natural", ""])
        else:
            rows.append([label, "—", "", "—", "—", ""])
    out["other_open"] = rows
    if light3:
        out["notes"].append(f"Third seat: one-level openings from {third[0]} HCP (1♥/1♠ in first and second seat from {first[0]}).")
    # -------- defence after a natural one of a suit
    simple, jump, cue, nts = [], [], [], {}
    for o in ("1C", "1D", "1H", "1S"):
        their = o[1]
        for c, p in lc.calls(o, min_share=0.0):
            s = strain(c)
            if c in ("P", "X") or s is None:
                continue
            if s == their:
                cue.append((o, c, p))
            elif s == "N":
                nts.setdefault(c, []).append(p)
            elif is_jump(o, c):
                jump.append((o, c, p))
            else:
                simple.append((o, c, p))

    def pooled(items):
        items = [i for i in items if i[-1].n >= MIN_N]
        if not items:
            return None
        lo = sorted(p.hcp()[0] for *_, p in items)[len(items) // 2]
        hi = sorted(p.hcp()[1] for *_, p in items)[len(items) // 2]
        mn = min(p.length(strain(c))[0] for _, c, p in items)
        return lo, hi, mn

    sp, jp = pooled(simple), pooled(jump)
    out["def_simple"] = f"{hcp_text(sp[0], sp[1])}, {sp[2]}+ cards" if sp else "—"
    out["def_jump"] = (("weak" if jp[1] <= 11 else "intermediate") + f", {hcp_text(jp[0], jp[1])}, {jp[2]}+ cards") if jp else "—"
    cues = [name_overcall(o[1], c, p) for o, c, p in cue if p.n >= MIN_N]
    out["def_cue"] = cues[0] if cues else "—"
    def median_range(ps):
        ps = [p for p in ps if p and p.n >= MIN_N]
        if not ps:
            return "—"
        return hcp_text(sorted(p.hcp()[0] for p in ps)[len(ps) // 2], sorted(p.hcp()[1] for p in ps)[len(ps) // 2])

    out["def_1nt"] = median_range(nts.get("1NT", []))
    out["def_1nt_prot"] = median_range([lc.prof(f"{o} P P", "1NT") for o in ("1C", "1D", "1H", "1S")])
    two_nt = [(o, lc.prof(o, "2NT")) for o in ("1C", "1D", "1H", "1S")]
    two_nt = [(o, p) for o, p in two_nt if p]
    out["def_2nt"] = name_overcall(two_nt[0][0][1], "2NT", two_nt[0][1]) if two_nt else "—"
    tx = [(o, lc.prof(o, "X")) for o in ("1C", "1D", "1H", "1S")]
    tx = [(o, p) for o, p in tx if p]
    out["def_x"] = name_overcall(tx[0][0][1], "X", tx[0][1]) if tx else "—"
    # -------- opponents open with something else
    vs = {}
    for label, keys in (("Strong 1NT", ["1NT"]), ("Weak 2", ["2D", "2H", "2S"]), ("Weak 3", ["3C", "3D", "3H", "3S"]),
                        ("4 bids", ["4H", "4S"])):
        parts = []
        for k in keys:
            x = lc.prof(k, "X")
            if x:
                parts.append(f"X over {call_text(k)}: {name_overcall(strain(k) if strain(k) != 'N' else '', 'X', x)}")
                break
        if label == "Strong 1NT":
            for c, p in lc.calls("1NT"):
                if c not in ("P", "X"):
                    parts.append(f"{call_text(c)} = {describe(p)}")
        vs[label] = "; ".join(parts[:3]) or "—"
    out["def_vs"] = vs
    # -------- slam
    out["blackwood"] = blackwood_replies(lc)
    # -------- competitive auctions
    neg_levels = []
    for o in ("1C", "1D", "1H", "1S"):
        for k, c in lc.ctx.items():
            parts = k.split()
            if len(parts) == 2 and parts[0] == o and parts[1] not in ("P", "X", "XX"):
                x = lc.prof(k, "X")
                if not x:
                    continue
                bid = {strain(o), strain(parts[1])}
                majors = [s for s in "HS" if s not in bid]
                if majors and min(x.at_least(s, 4) for s in majors) >= 0.6:
                    neg_levels.append(parts[1])
    if neg_levels:
        top = max(neg_levels, key=lambda c: (int(c[0]), STRAIN_ORDER.find(strain(c))))
        out["neg_level"] = call_text(top)
    else:
        out["neg_level"] = "—"
    after_tx = {}
    for o in ("1H", "1S", "1C", "1D"):
        for c, p in lc.calls(f"{o} X", min_share=0.02):
            s = strain(c)
            key = ("Redouble" if c == "XX" else "2NT" if c == "2NT" else
                   "Jump raise" if s == o[1] and c[0] == "3" else
                   "Jump in new suit" if s and s not in (o[1], "N") and is_jump(o, c) and c[0] >= "2" else
                   "New suit" if s and s not in (o[1], "N") else None)
            if key and key not in after_tx:
                txt = describe(p, skip="")
                if key == "2NT" and o[1] in "HS" and p.at_least(o[1], 3) >= NAMED:
                    txt = f"raise, {hcp_text(*p.hcp())}, {p.length(o[1])[0]}+{SYM[o[1]]}"
                elif key == "Jump raise":
                    txt = ("weak" if p.hcp()[1] <= 9 else "limit") + f", {hcp_text(*p.hcp())}"
                elif key == "Redouble":
                    txt = f"{hcp_text(*p.hcp())}"
                after_tx[key] = txt
    out["after_tx"] = after_tx
    return out


# ---------------------------------------------------------------- drawing the EBU 20B
W, H = 11.69, 8.27     # A4 landscape
M = 0.3
FS = 7.0
LINE = 0.135
INK, SHADE, RULE = "#111111", "#d9d9d9", "#555555"


class Panel:
    def __init__(self, ax, x0: float, x1: float, top: float):
        self.ax, self.x0, self.x1, self.y = ax, x0, x1, top

    def _wrap(self, text: str, width_in: float, fs: float = FS) -> List[str]:
        chars = max(4, int(width_in / (fs / 72 * 0.56)))
        out = []
        for para in str(text).split("\n"):
            out += textwrap.wrap(para, chars) or [""]
        return out

    def title(self, text: str, size: float = 8.4):
        h = 0.2
        self.ax.add_patch(Rectangle((self.x0, self.y - h), self.x1 - self.x0, h, fc=INK, lw=0))
        self.ax.text(self.x0 + 0.06, self.y - h / 2, text.upper(), color="white", fontsize=size, fontweight="bold",
                     va="center")
        self.y -= h

    def table(self, headers: Optional[List[str]], rows: List[List[str]], widths: List[float],
              bold_first: bool = True, shade: bool = False, head_fs: float = 6.2):
        total = self.x1 - self.x0
        ws = [w / sum(widths) * total for w in widths]
        all_rows = ([headers] if headers else []) + rows
        for ri, row in enumerate(all_rows):
            head = headers is not None and ri == 0
            fs = head_fs if head else FS
            cells = [self._wrap(v, w - 0.08, fs) for v, w in zip(row, ws)]
            h = max(len(c) for c in cells) * LINE + 0.07
            x = self.x0
            for ci, (c, w) in enumerate(zip(cells, ws)):
                fc = SHADE if (head or shade) else "white"
                self.ax.add_patch(Rectangle((x, self.y - h), w, h, fc=fc, ec=RULE, lw=0.5))
                for li, line in enumerate(c):
                    self.ax.text(x + 0.04, self.y - 0.05 - (li + 0.5) * LINE, line, fontsize=fs, va="center",
                                 color=INK, fontweight="bold" if (head or (bold_first and ci == 0)) else "normal")
                x += w
            self.y -= h

    def kv(self, rows: List[Tuple[str, str]], label_w: float = 0.32):
        self.table(None, [[k, v] for k, v in rows], [label_w, 1 - label_w])

    def text(self, lines: List[str], fs: float = FS, h_min: float = 0.0):
        wrapped = [l for t in lines for l in self._wrap(t, self.x1 - self.x0 - 0.1, fs)]
        h = max(h_min, len(wrapped) * LINE + 0.08)
        self.ax.add_patch(Rectangle((self.x0, self.y - h), self.x1 - self.x0, h, fc="white", ec=RULE, lw=0.5))
        for i, l in enumerate(wrapped):
            self.ax.text(self.x0 + 0.05, self.y - 0.06 - (i + 0.5) * LINE, l, fontsize=fs, va="center", color=INK)
        self.y -= h

    def gap(self, h: float = 0.08):
        self.y -= h


def _title(ax, name: str) -> float:
    """The model's name across the top, as a partnership writes its names on the card."""
    y = H - M
    ax.text(M, y - 0.22, name, fontsize=17, fontweight="bold", color=INK, va="center")
    ax.text(W - M, y - 0.22, "CONVENTION CARD", fontsize=9, fontweight="bold", color=INK, va="center", ha="right")
    ax.plot([M, W - M], [y - 0.45, y - 0.45], lw=1.4, color=INK)
    return y - 0.58


def draw_card(name: str, card: dict, out: Path) -> None:
    """One A4 landscape page: the model's name, then its agreements (no play, no comparison with another system)."""
    f = fill(Learned(name, card))
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, color="white", lw=0))
    top = _title(ax, name)
    left = Panel(ax, M, W / 2 - 0.12, top)
    right = Panel(ax, W / 2 + 0.12, W - M, top)

    p = left
    p.title("General description of bidding methods")
    p.text([f["general"]], fs=8.2)
    p.gap()
    p.title("1NT openings and responses")
    p.kv([("Strength", f["nt_strength"]), ("Shape constraints", f["nt_shapes"]),
          ("May have singleton", "\u2713 yes" if f["nt_singleton"] else "no")], 0.3)
    p.table(["Responses", ""], [[call_text(c), v] for c, v in f["nt_resp"].items()]
            + [["3 level", f["nt_3l"]], ["4 level", f["nt_4l"]],
               ["After their double", f["nt_after_x"]]], [0.75, 3.0])
    p.gap()
    p.title("Two-level openings and responses")
    p.table(["", "Meaning", "Responses"], [[call_text(c), m, r] for c, (m, r) in f["two"].items()],
            [0.45, 1.4, 2.6])
    p.gap()
    p.title("Slam conventions")
    p.table(["Name", "Meaning of responses"], [[f["blackwood"][0], f["blackwood"][1]]], [1.4, 2.6])
    if f["notes"]:
        p.gap()
        p.title("Supplementary details")
        p.text(f["notes"])

    q = right
    q.title("Other opening bids")
    q.table(["", "HCP", "see note", "Min length", "Meaning", "Special responses"],
            f["other_open"], [0.5, 0.6, 0.45, 0.5, 1.0, 2.4])
    q.gap()
    q.title("Defensive methods after opponents open")
    q.table(["Opponents open one of a suit", ""],
            [["Simple overcall", f["def_simple"]], ["Jump overcall", f["def_jump"]], ["Cue bid", f["def_cue"]],
             ["Double", f["def_x"]], ["1NT direct / protective", f"{f['def_1nt']}  /  {f['def_1nt_prot']}"],
             ["2NT", f["def_2nt"]]], [1.3, 2.6])
    q.table(["Opponents open with", ""], [[k, v] for k, v in f["def_vs"].items()], [1.3, 2.6])
    q.gap()
    q.title("Competitive auctions")
    q.kv([("Negative doubles up to", f["neg_level"])], 0.34)
    q.table(["After their take-out double", ""],
            [[k, f["after_tx"].get(k, "\u2014")] for k in ("Redouble", "New suit", "Jump in new suit", "Jump raise", "2NT")],
            [1.3, 2.6])
    lo = min(left.y, right.y)
    if lo < M - 0.05:
        raise ValueError(f"{name}: the card runs off the page ({lo:.2f} in)")
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards_dir", required=True, help="the --out_dir of evaluation/convention_cards_from_play.py")
    ap.add_argument("--models", nargs="*", default=None, help="model names (default: every card in cards_dir)")
    ap.add_argument("--out_dir", default=None, help="default <cards_dir>/ebu")
    ap.add_argument("--prefix", default="", help="file name prefix, e.g. fig_card_ebu_")
    args = ap.parse_args()
    cards = load_cards(Path(args.cards_dir))
    out = Path(args.out_dir) if args.out_dir else Path(args.cards_dir) / "ebu"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models or list(cards):
        if name not in cards:
            raise SystemExit(f"no card for {name!r} in {args.cards_dir}")
        draw_card(name, cards[name], out / f"{args.prefix}{slug(name)}")
        print(f"[ebu] {name} -> {out / (args.prefix + slug(name))}.pdf")


if __name__ == "__main__":
    main()
