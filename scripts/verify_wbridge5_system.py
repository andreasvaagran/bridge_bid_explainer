#!/usr/bin/env python3
"""Re-measures the facts about WBridge5's system that the rules were calibrated to, on the whole training
split (notes/wbridge5_system.md)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

SPLIT = REPO / "data" / "wbridge5" / "prepared" / "train"
BIDS = ["P", "X", "XX"] + [f"{lvl}{s}" for lvl in range(1, 8) for s in ("C", "D", "H", "S", "NT")]
RANK = {"C": 0, "D": 1, "H": 2, "S": 3}


def _real(c: str) -> bool:
    return c not in ("P", "X", "XX")


def _suit(c: str):
    return c[-1] if _real(c) and c[-1] in RANK else None


def _level(c: str) -> int:
    return int(c[0]) if _real(c) else 0


def _deal_records(calls, hands, out):
    """All measurements of one deal. calls: WBridge5's calls in order; hands: seat -> HandFeatures."""
    from concepts.predicates_hand import losing_trick_count, rule_of_20_satisfied, wb5_points
    from concepts.rules import _opener_reopening_double
    from concepts.types import AuctionContext

    n = len(calls)
    o = next((i for i, c in enumerate(calls) if _real(c)), None)
    seat = lambda i: i % 4  # noqa: E731  (dealer = North = 0)
    aces = lambda h: sum("A" in h.cards[s] for s in "SHDC")  # noqa: E731
    # opening strength (concepts.rules._wb5_can_open: 12+ HCP, 11 with 13+ points, or the Rule of 20),
    # first and second seat: one-level suit openings, and the passes made instead of opening
    for k in range(min(2, n)):
        if k > 0 and _real(calls[k - 1]):
            break
        hk = hands[seat(k)]
        can = hk.hcp >= 12 or (hk.hcp >= 11 and wb5_points(hk) >= 13) or rule_of_20_satisfied(hk)
        if calls[k] in ("1C", "1D", "1H", "1S"):
            out["open_strength"].append(can)
        elif calls[k] == "P":
            out["pass_strength"].append(can)
        if _real(calls[k]):
            break
    if o is not None:
        oc, h = calls[o], hands[seat(o)]
        x = _suit(oc)
        if oc in ("2D", "2H", "2S"):
            L = h.lengths[x]
            other = {"H": "S", "S": "H"}.get(x)
            out["weak_two"].append((o + 1, x, L, h.hcp, wb5_points(h),
                                    h.lengths[other] if other else -1))
        if oc in ("3C", "3D", "3H", "3S"):
            out["preempt3"].append(h.lengths[x])
        if oc == "2C":
            out["two_clubs"].append((h.hcp, wb5_points(h), losing_trick_count(h)))
        if oc == "1NT":
            out["one_nt"].append(h.hcp)
        if oc == "2NT":
            out["two_nt"].append(h.hcp)
        if oc in ("1C", "1D"):
            out["one_minor"].append((oc, h.lengths["C"], h.lengths["D"], h.lengths["H"], h.lengths["S"]))
        if (h.lengths["S"], h.lengths["H"], h.lengths["D"], h.lengths["C"]) == (4, 4, 3, 2):
            out["shape4432"].append(oc)
        if oc == "4NT":
            out["four_nt_opening"].append((h.lengths["C"], h.lengths["D"]))
        # direct actions over a one-level suit opening, by the next player (LHO of the opener)
        if _level(oc) == 1 and x and o + 1 < n:
            nxt, hl = calls[o + 1], hands[seat(o + 1)]
            if nxt == "2" + x:  # Michaels: a direct cuebid
                out["michaels"].append(sum(hl.lengths[s] >= 5 for s in "SHDC") >= 2)
            if nxt == "2NT":    # unusual 2NT: the two lowest unbid suits
                low = sorted((s for s in "CDHS" if s != x), key=RANK.get)[:2]
                out["unusual_2nt"].append(all(hl.lengths[s] >= 5 for s in low))
            if nxt == "X":      # takeout double of a one-level opening
                out["takeout_double"].append((hl.hcp, hl.lengths[x]))
            # negative double: partner opened 1x, RHO overcalled a suit, responder doubles
            if o + 2 < n and _suit(nxt) and _suit(nxt) != x and calls[o + 2] == "X":
                out["negative_double"].append(hands[seat(o + 2)].hcp)
            # 2-over-1: RHO passed, responder bids a lower-ranking new suit at the two level
            if o + 2 < n and nxt == "P":
                r = calls[o + 2]
                if _level(r) == 2 and _suit(r) and _suit(r) != x and RANK[_suit(r)] < RANK[x]:
                    out["two_over_one"].append(hands[seat(o + 2)].hcp)
        # Gerber: responder's 4C directly over partner's 1NT / 2NT opening, RHO passing
        if oc in ("1NT", "2NT") and o + 2 < n and calls[o + 1] == "P":
            out["nt_responses"] += 1
            if calls[o + 2] == "4C":
                out["gerber"] += 1
        # 1NT (P) 2S (P): the relay
        if oc == "1NT" and o + 4 < n and calls[o + 1:o + 4] == ["P", "2S", "P"]:
            hr = hands[seat(o + 2)]
            out["nt_2s"].append((calls[o + 4], max(hr.lengths["C"], hr.lengths["D"]), hr.lengths["S"]))
        # 1m (P) 1M (P) 1NT (P) 2m': natural or new minor forcing?
        if (oc in ("1C", "1D") and o + 6 < n and calls[o + 1] == "P" and calls[o + 2] in ("1H", "1S")
                and calls[o + 3] == "P" and calls[o + 4] == "1NT" and calls[o + 5] == "P"
                and calls[o + 6] in ("2C", "2D") and calls[o + 6][-1] != oc[-1]):
            hr = hands[seat(o + 2)]
            m2, maj = calls[o + 6][-1], calls[o + 2][-1]
            passed = o + 8 < n and calls[o + 7] == "P" and calls[o + 8] == "P"
            out["nmf"].append((hr.lengths[m2], hr.lengths[maj], passed))
        # double of a 1NT overcall of partner's one-level suit opening: does partner leave it in?
        if (_level(oc) == 1 and x and o + 4 < n and calls[o + 1] == "1NT" and calls[o + 2] == "X"
                and calls[o + 3] == "P"):
            out["dbl_1nt_overcall"].append(calls[o + 4] == "P")
        # direct double of a 1NT opening
        if oc == "1NT" and o + 3 < n and calls[o + 1] == "X" and calls[o + 2] == "P":
            out["dbl_1nt_opening"].append(calls[o + 3] == "P")
    for k in range(n):
        c = calls[k]
        me, h = seat(k), hands[seat(k)]
        # Blackwood replies: partner's 4NT, RHO passed, our side had bid a suit before the 4NT, no opponent bid
        if (k >= 2 and calls[k - 2] == "4NT" and calls[k - 1] == "P" and c in ("5C", "5D", "5H", "5S")
                and any(_suit(calls[i]) for i in range(k - 2) if seat(i) in (me, (me + 2) % 4))
                and not any(_real(calls[i]) for i in range(k) if seat(i) in ((me + 1) % 4, (me + 3) % 4))):
            out["blackwood"].append((c, aces(h)))
        if c == "X":
            seq = [(i, calls[i]) for i in range(k)]
            while seq and seq[0][1] == "P":
                seq.pop(0)
            # support double (nlg.explanation_inputs.support_double_suit): 1m (P or X) 1r/2r (RHO's suit z) X by opener
            if len(seq) == 4:
                (im, m_), (il, lho), (ir, rr), (iz, zz) = seq
                m, r, z = _suit(m_), _suit(rr), _suit(zz)
                if (seat(im) == me and m_[0] == "1" and m and lho in ("P", "X") and seat(ir) == (me + 2) % 4
                        and rr[0] in "12" and r and r != m and z and z not in (m, r)):
                    top = 2 if lho == "X" else 3
                    if _level(zz) <= top:
                        out["support_double"].append((lho, _level(zz), h.lengths[r]))
            # opener's reopening double (concepts.rules._opener_reopening_double)
            if _opener_reopening_double(AuctionContext(bids=calls[:k], bid="X", dealer=0)):
                last = max(i for i in range(k) if _real(calls[i]))
                z = _suit(calls[last])
                taken_out = None
                if k + 2 < n and calls[k + 1] == "P":
                    taken_out = calls[k + 2] != "P"
                out["reopening_double"].append((_level(calls[last]), h.lengths[z], taken_out))


KEYS = ["weak_two", "preempt3", "two_clubs", "one_nt", "two_nt", "one_minor", "shape4432", "four_nt_opening",
        "michaels", "unusual_2nt", "takeout_double", "negative_double", "two_over_one", "nt_2s", "nmf",
        "dbl_1nt_overcall", "dbl_1nt_opening", "blackwood", "support_double", "reopening_double",
        "open_strength", "pass_strength"]


def _work(span):
    from concepts.hand import decode_hand
    lo, hi = span
    offsets = np.load(SPLIT / "deal_offsets.npy", mmap_mode="r")
    obs_bits = np.load(SPLIT / "obs_bits.npy", mmap_mode="r")
    action = np.load(SPLIT / "action.npy", mmap_mode="r")
    out = {k: [] for k in KEYS}
    out["gerber"] = out["nt_responses"] = 0
    for d in range(lo, hi):
        a, b = int(offsets[d]), int(offsets[d + 1])
        calls = [BIDS[int(x)] for x in action[a:b]]
        hands = {}
        for k in range(min(4, b - a)):  # every seat calls at least once; its first decision holds its hand
            obs = np.unpackbits(obs_bits[a + k])[:480]
            hands[k] = decode_hand(obs)
        if len(hands) < 4:
            continue
        _deal_records(calls, hands, out)
    return out


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def share(xs):
    xs = list(xs)
    return (sum(1 for x in xs if x) / len(xs) if xs else float("nan")), len(xs)


def summarise(R) -> dict:
    S = {}
    w2 = R["weak_two"]
    third = [r for r in w2 if r[0] == 3]
    S["weak_two_third_seat_5_cards"] = share(r[2] == 5 for r in third)
    S["weak_two_seat12_5_cards"] = share(r[2] == 5 for r in w2 if r[0] in (1, 2))
    S["weak_two_major_5_cards"] = share(r[2] == 5 for r in w2 if r[1] in "HS")
    S["weak_two_6plus_or_5"] = share(r[2] >= 5 for r in w2)
    S["weak_two_points_ge_7"] = share(r[4] >= 7 for r in w2)
    S["weak_two_hcp_ge_12"] = share(r[3] >= 12 for r in w2)
    S["weak_two_hcp_5_10"] = share(5 <= r[3] <= 10 for r in w2)
    # a six-card suit and 5-10 HCP: a common summary of SAYC, NOT the ACBL booklet (kept for comparison)
    S["weak_two_textbook_6_cards_and_5_10"] = share(r[2] == 6 and 5 <= r[3] <= 10 for r in w2)
    # the ACBL booklet (SP-3, January 2006): "a six-card suit of reasonable quality and 5-11 HCP.
    # On rare occasions it may be a very good five-card suit."
    S["weak_two_hcp_5_11"] = share(5 <= r[3] <= 11 for r in w2)
    S["weak_two_booklet_6_cards_and_5_11"] = share(r[2] == 6 and 5 <= r[3] <= 11 for r in w2)
    S["weak_two_seat12_major_4plus_other_major"] = share(r[5] >= 4 for r in w2 if r[0] in (1, 2) and r[1] in "HS")
    S["preempt3_6plus"] = share(L >= 6 for L in R["preempt3"])
    S["preempt3_7plus"] = share(L >= 7 for L in R["preempt3"])
    c2 = R["two_clubs"]
    S["two_clubs_hcp_ge_22"] = share(r[0] >= 22 for r in c2)
    S["two_clubs_points21_or_ltc4"] = share(r[1] >= 21 or r[2] <= 4 for r in c2)
    S["two_clubs_rule"] = share(r[0] >= 22 or r[1] >= 21 or r[2] <= 4 for r in c2)
    S["one_level_opening_seat12_meets_open_rule"] = share(R["open_strength"])
    S["opening_pass_seat12_meets_open_rule"] = share(R["pass_strength"])
    S["one_nt_15_17"] = share(15 <= x <= 17 for x in R["one_nt"])
    S["two_nt_20_21"] = share(20 <= x <= 21 for x in R["two_nt"])

    def longer_minor_rule(oc, c, d):
        if oc == "1C":
            return c > d or (c == d == 3)
        return d > c or (c == d and c >= 4)
    om = R["one_minor"]
    S["one_minor_longer_minor_rule"] = share(longer_minor_rule(r[0], r[1], r[2]) for r in om)
    S["one_club_three_cards"] = share(r[1] == 3 for r in om if r[0] == "1C")
    sh = [c for c in R["shape4432"] if c in ("1C", "1D")]
    S["shape_4432_minor_openings_1D"] = share(c == "1D" for c in sh)
    S["shape_4432_openings"] = dict(Counter(R["shape4432"]).most_common(6))
    S["four_nt_opening_5_5_minors"] = share(c >= 5 and d >= 5 for c, d in R["four_nt_opening"])
    S["four_nt_opening_6_6_minors"] = share(c >= 6 and d >= 6 for c, d in R["four_nt_opening"])
    S["michaels_two_5card_suits"] = share(R["michaels"])
    S["unusual_2nt_5_5_lowest_unbid"] = share(R["unusual_2nt"])
    td = R["takeout_double"]
    S["takeout_double_hcp_p5_p50_p95"] = [pct([r[0] for r in td], q) for q in (5, 50, 95)] + [len(td)]
    S["takeout_double_len_doubled_suit_median"] = pct([r[1] for r in td], 50)
    nd = R["negative_double"]
    S["negative_double_hcp_p5_p50_p95"] = [pct(nd, q) for q in (5, 50, 95)] + [len(nd)]
    t1 = R["two_over_one"]
    S["two_over_one_hcp_p5_p50_p95_min_max"] = [pct(t1, q) for q in (5, 50, 95)] + [min(t1), max(t1), len(t1)]
    ns = R["nt_2s"]
    S["nt_2s_opener_3c"] = share(r[0] == "3C" for r in ns)
    S["nt_2s_responder_6plus_minor"] = share(r[1] >= 6 for r in ns)
    S["nt_2s_responder_0_3_spades"] = share(r[2] <= 3 for r in ns)
    nm = R["nmf"]
    S["nmf_4plus_in_bid_minor"] = share(r[0] >= 4 for r in nm)
    S["nmf_5plus_in_major"] = share(r[1] >= 5 for r in nm)
    S["nmf_opener_passed"] = share(r[2] for r in nm)
    S["double_1nt_overcall_left_in"] = share(R["dbl_1nt_overcall"])
    S["double_1nt_opening_left_in"] = share(R["dbl_1nt_opening"])
    S["gerber_4c_over_nt_openings"] = [R["gerber"], R["nt_responses"]]
    bw = defaultdict(Counter)
    for c, a in R["blackwood"]:
        bw[c][a] += 1
    S["blackwood_replies_aces"] = {c: dict(sorted(bw[c].items())) for c in ("5C", "5D", "5H", "5S")}
    sd = defaultdict(list)
    for lho, lvl, L in R["support_double"]:
        sd[f"after {'LHO pass' if lho == 'P' else 'LHO double'}, RHO at level {lvl}"].append(L == 3)
    S["support_double_exactly_3"] = {k: share(v) for k, v in sorted(sd.items())}
    rd = R["reopening_double"]
    S["reopening_double_0_2_in_their_suit_levels_1_2"] = share(r[1] <= 2 for r in rd if r[0] <= 2)
    S["reopening_double_taken_out_level1"] = share(r[2] for r in rd if r[0] == 1 and r[2] is not None)
    S["reopening_double_taken_out_level2"] = share(r[2] for r in rd if r[0] == 2 and r[2] is not None)
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_deals", type=int, default=None, help="default: the whole training split")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    offsets = np.load(SPLIT / "deal_offsets.npy", mmap_mode="r")
    n_deals = len(offsets) - 1 if args.max_deals is None else min(args.max_deals, len(offsets) - 1)
    step = 20_000
    spans = [(i, min(i + step, n_deals)) for i in range(0, n_deals, step)]
    R = {k: [] for k in KEYS}
    R["gerber"] = R["nt_responses"] = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for part in pool.imap_unordered(_work, spans):
            for k in KEYS:
                R[k] += part[k]
            R["gerber"] += part["gerber"]
            R["nt_responses"] += part["nt_responses"]
    S = summarise(R)
    S["_deals"] = n_deals
    S["_decisions"] = int(offsets[n_deals])
    S["_split"] = str(SPLIT.relative_to(REPO))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(S, indent=1))
    for k, v in S.items():
        print(f"{k:50s} {v}")


if __name__ == "__main__":
    main()
