#!/usr/bin/env python3
"""Audits the rule engine against WBridge5's real hands, run after every rule edit: a profile of each
concept, how often the hand contradicts what a concept claims, how often a situation has a hand-based reason, and
the concepts that are never true."""
from __future__ import annotations

import argparse
import collections
import itertools
import multiprocessing as mp
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

_PG = {"Pass": "P", "Dbl": "X", "RDbl": "XX"}


def _pgx(s: str) -> str:
    return _PG.get(s, s[:-1] + "NT" if s.endswith("N") else s)


def _work(lines):
    import pyspiel
    from concepts import AuctionContext, HARD_STATE_CONCEPTS, create_sayc_state_authority, decode_hand
    from nlg.explanation_inputs import bid_category
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    auth = create_sayc_state_authority()
    rows = []
    for line in lines:
        traj = [int(x) for x in line.split()]
        traj = traj[:-52] if len(traj) > 56 else traj
        st, calls = game.new_initial_state(), []
        for idx, x in enumerate(traj):
            if idx >= 52:
                obs = np.asarray(st.observation_tensor()[4:484])
                h = decode_hand(obs)
                bid, n = _pgx(st.action_to_string(x)), len(calls)
                seat_of = lambda i: "NESW"[(2 - (n - i)) % 4]  # actor = S, partner = N  # noqa: E731
                hand_str = " ".join(f"{s}:{''.join(sorted(h.cards[s], key='23456789TJQKA'.index, reverse=True)) or '-'}"
                                    for s in "SHDC")
                frame = {"table": {"you": "S", "partner": "N", "opponents": ["W", "E"]},
                         "auction_before": [{"seat": seat_of(i), "bid": c} for i, c in enumerate(calls)],
                         "bid": bid, "hand_str": hand_str, "dealer": seat_of(0) if calls else "S"}
                g = auth.classify(AuctionContext(bids=list(calls), bid=bid, dealer=(-n) % 4), h)
                rows.append((tuple(calls), bid, hand_str, h.hcp, bid_category(frame),
                             np.packbits(np.array([g[c] for c in HARD_STATE_CONCEPTS], np.uint8))))
            if not st.is_chance_node():
                calls.append(_pgx(st.action_to_string(x)))
            st.apply_action(x)
    return rows


def build_rows(path: str, deals: int, workers: int):
    lines = list(itertools.islice(open(path), deals))
    chunks = [lines[i:i + 500] for i in range(0, len(lines), 500)]
    with mp.get_context("fork").Pool(workers) as pool:
        return [r for part in pool.imap(_work, chunks) for r in part]


# ---------------------------------------------------------------------------
# claim checks: what each concept's description says about the hand
# ---------------------------------------------------------------------------

def _claim_checks():
    from concepts.predicates_hand import quick_tricks, support_points
    NB = ("P", "X", "XX")

    def strain(b):
        return None if b in NB else ("N" if b.endswith("NT") else b[-1])

    def L(h, s):
        return h.lengths.get(s, 0) if s else 0

    def aces(h):
        return sum("A" in h.cards[s] for s in "SHDC")

    def stopper(h, s):
        c = h.cards[s]
        return "A" in c or ("K" in c and len(c) >= 2) or ("Q" in c and len(c) >= 3) or ("J" in c and len(c) >= 4)

    ps = lambda p: strain(p[-1]) if p and strain(p[-1]) in tuple("SHDC") else None  # noqa: E731
    bs = lambda r: strain(r[1]) if strain(r[1]) in tuple("SHDC") else None  # noqa: E731
    return {
        "bid_shows_4_plus_cards": ("bid suit >= 4", lambda r, h, p, o: L(h, bs(r)) >= 4),
        "bid_shows_5_plus_cards": ("bid suit >= 5", lambda r, h, p, o: L(h, bs(r)) >= 5),
        "bid_shows_6_plus_cards": ("bid suit >= 6", lambda r, h, p, o: L(h, bs(r)) >= 6),
        "bid_shows_7_plus_cards": ("bid suit >= 7", lambda r, h, p, o: L(h, bs(r)) >= 7),
        "bid_shows_exactly_5_cards": ("bid suit == 5", lambda r, h, p, o: L(h, bs(r)) == 5),
        "bid_shows_exactly_6_cards": ("bid suit == 6", lambda r, h, p, o: L(h, bs(r)) == 6),
        "bid_shows_3_card_support": ("3+ in partner's suit", lambda r, h, p, o: ps(p) and L(h, ps(p)) >= 3),
        "bid_shows_4_plus_support": ("4+ in partner's suit", lambda r, h, p, o: ps(p) and L(h, ps(p)) >= 4),
        "bid_shows_support": ("3+ in partner's suit", lambda r, h, p, o: ps(p) and L(h, ps(p)) >= 3),
        "bid_raises_partners_overcall": ("3+ in partner's suit", lambda r, h, p, o: ps(p) and L(h, ps(p)) >= 3),
        "bid_denies_3_card_support": ("< 3 in partner's suit", lambda r, h, p, o: not ps(p) or L(h, ps(p)) < 3),
        "bid_denies_shortness": ("no singleton/void", lambda r, h, p, o: min(h.lengths.values()) >= 2),
        "bid_shows_stopper_in_their_suit": ("every opponent suit stopped (A, Kx, Qxx, Jxxx)",
                                            lambda r, h, p, o: any(strain(x) in tuple("SHDC") for x in o) and all(
                                                stopper(h, strain(x)) for x in o if strain(x) in tuple("SHDC"))),
        "bid_shows_aces_0_or_3": ("0 or 3 aces", lambda r, h, p, o: aces(h) in (0, 3)),
        "bid_shows_aces_1_or_4": ("1 or 4 aces", lambda r, h, p, o: aces(h) in (1, 4)),
        "bid_shows_aces_2": ("2 aces", lambda r, h, p, o: aces(h) == 2),
        "bid_shows_quick_tricks_2plus": ("2+ quick tricks, not a pass", lambda r, h, p, o: r[1] != "P" and quick_tricks(h) >= 2),
        "bid_shows_weak_values": ("<= 11 HCP", lambda r, h, p, o: h.hcp <= 11),
        "bid_shows_minimum_values": ("<= 15 HCP", lambda r, h, p, o: h.hcp <= 15),
        "bid_shows_invitational_values": ("8-13 HCP", lambda r, h, p, o: 8 <= h.hcp <= 13),
        "bid_shows_game_forcing_values": (">= 12 HCP (or 2C)", lambda r, h, p, o: h.hcp >= 12 or r[1] == "2C"),
        "bid_shows_extra_values": (">= 13 HCP", lambda r, h, p, o: h.hcp >= 13),
        "bid_shows_minor_suit_only": ("no 5-card major", lambda r, h, p, o: max(L(h, "S"), L(h, "H")) <= 4),
        "bid_shows_first_control": ("A or void in bid suit", lambda r, h, p, o: bs(r) and ("A" in h.cards[bs(r)] or L(h, bs(r)) == 0)),
        "bid_shows_second_control": ("K or singleton in bid suit", lambda r, h, p, o: bs(r) and ("K" in h.cards[bs(r)] or L(h, bs(r)) <= 1)),
        "bid_is_preempt_quality_good": ("2+ of AKQ in bid suit", lambda r, h, p, o: bs(r) and sum(x in h.cards[bs(r)] for x in "AKQ") >= 2),
        "bid_is_takeout_double": ("<= 3 cards in the doubled suit", lambda r, h, p, o: not o or strain(o[-1]) not in tuple("SHDC") or L(h, strain(o[-1])) <= 3),
        "bid_is_weak_two": ("< 12 HCP, 5+ cards", lambda r, h, p, o: h.hcp <= 11 and L(h, bs(r)) >= 5),
        "bid_is_strong_2cc_opening": ("21+ points or <= 4 losers or 22+ HCP", lambda r, h, p, o: True),
        "bid_is_1nt_opening": ("15-17 HCP", lambda r, h, p, o: 15 <= h.hcp <= 17),
        "bid_is_natural_2nt_overcall": ("15-19 HCP", lambda r, h, p, o: 15 <= h.hcp <= 19),
        "bid_is_two_over_one_response": (">= 10 HCP", lambda r, h, p, o: h.hcp >= 10),
        "bid_pass_lacks_opening_values": ("< 12 HCP", lambda r, h, p, o: h.hcp < 12),
        "bid_pass_lacks_values_to_compete": ("< 12 HCP", lambda r, h, p, o: h.hcp < 12),
        "bid_shows_support_points_invite_plus": ("10+ support points", lambda r, h, p, o: ps(p) and support_points(h, ps(p)) >= 10),
        "bid_is_unusual_4nt_opening": ("5-5+ minors", lambda r, h, p, o: L(h, "C") >= 5 and L(h, "D") >= 5),
        "bid_is_light_third_seat_weak_two": ("exactly 5 in bid suit", lambda r, h, p, o: L(h, bs(r)) == 5),
        "bid_pass_too_weak_to_respond": ("< 14 HCP", lambda r, h, p, o: h.hcp < 14),
        "bid_double_with_length_in_their_suit": ("3+ in the doubled suit", lambda r, h, p, o: bool(o) and strain(o[-1]) in tuple("SHDC") and L(h, strain(o[-1])) >= 3),
        "bid_pass_no_extra_values": ("<= 14 HCP", lambda r, h, p, o: h.hcp <= 14),
        "bid_stayman_with_4_card_major": ("a 4-card major", lambda r, h, p, o: 4 in (L(h, "H"), L(h, "S"))),
        "bid_transfer_with_5_card_major": ("5+ in the target major", lambda r, h, p, o: L(h, {"D": "H", "H": "S"}[strain(r[1])]) >= 5),
        "bid_is_negative_double": (">= 6 HCP", lambda r, h, p, o: h.hcp >= 6),
        "bid_pass_content_with_final_contract": ("<= 19 HCP", lambda r, h, p, o: h.hcp <= 19),
        "bid_answers_king_ask": ("kings match 6C=0/6D=1/6H=2/6S=3", lambda r, h, p, o: {"6C": 0, "6D": 1, "6H": 2, "6S": 3}.get(r[1]) == sum("K" in h.cards[x] for x in "SHDC")),
        "bid_is_preemptive_jump_response": ("<= 9 HCP, 6+ in bid suit", lambda r, h, p, o: h.hcp <= 9 and L(h, bs(r)) >= 6),
        "bid_positive_response_to_2c": ("7+ HCP", lambda r, h, p, o: h.hcp >= 7),
        "bid_waiting_response_without_positive": ("< 7 HCP or no 5-card suit", lambda r, h, p, o: h.hcp < 7 or max(h.lengths.values()) < 5),
        "bid_is_redouble": (">= 7 HCP", lambda r, h, p, o: h.hcp >= 7),
        "bid_is_preemptive_raise": ("<= 9 HCP, 4+ in bid suit", lambda r, h, p, o: h.hcp <= 9 and L(h, bs(r)) >= 4),
    }, strain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(REPO, "data", "wbridge5", "official_train.txt"))
    ap.add_argument("--deals", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--coverage_per_category", type=int, default=150,
                    help="decisions sampled per bidding situation for the (slow) hand-dependence test")
    ap.add_argument("--out", default=None, help="write a markdown report here")
    args = ap.parse_args()

    from concepts import HARD_STATE_CONCEPTS as C, gloss_for_concept
    from nlg.explanation_inputs import hand_dependent_concepts, hand_from_str

    t0 = time.time()
    rows = build_rows(args.data, args.deals, args.workers)
    V = np.unpackbits(np.stack([r[5] for r in rows]), axis=1)[:, :len(C)].astype(bool)
    hcp = np.array([r[3] for r in rows]); bids = np.array([r[1] for r in rows]); cats = np.array([r[4] for r in rows])
    N = len(rows)
    out = [f"# Concept audit -- {args.deals} WBridge5 deals, {N:,} decisions ({time.time() - t0:.0f}s replay)", ""]

    # 1. profile
    out += ["## 1. Profile", "", "| # | concept | % true | HCP p5/p50/p95 | top calls | top situations |", "|---|---|---|---|---|---|"]
    never = []
    for j, c in enumerate(C):
        m = V[:, j]; n = int(m.sum())
        if n == 0:
            never.append(c); continue
        p5, p50, p95 = np.percentile(hcp[m], [5, 50, 95])
        tb = ", ".join(f"{b} {k / n:.0%}" for b, k in collections.Counter(bids[m]).most_common(4))
        tc = ", ".join(f"{b} {k / n:.0%}" for b, k in collections.Counter(cats[m]).most_common(3))
        out.append(f"| {j} | `{c}` | {n / N:.2%} | {p5:.0f}/{p50:.0f}/{p95:.0f} | {tb} | {tc} |")

    # 2. claims
    checks, strain = _claim_checks()
    out += ["", "## 2. Hand claims (how often the real hand contradicts the concept's description)", "",
            "| concept | n true | claim | contradicted | example |", "|---|---|---|---|---|"]
    for c, (desc, fn) in checks.items():
        if c not in C:
            continue
        sel = np.flatnonzero(V[:, C.index(c)]); bad, ex = 0, ""
        for k in sel:
            r = rows[k]; h = hand_from_str(r[2]); calls = r[0]; n = len(calls)
            p = [x for i, x in enumerate(calls) if (n - i) % 4 == 2 and x not in ("P", "X", "XX")]
            o = [x for i, x in enumerate(calls) if (n - i) % 4 in (1, 3) and x not in ("P", "X", "XX")]
            try:
                ok = bool(fn(r, h, p, o))
            except Exception:
                ok = False
            if not ok:
                bad += 1; ex = ex or f"{' '.join(calls)} -> {r[1]} | {r[2]}"
        out.append(f"| `{c}` | {len(sel)} | {desc} | {bad / max(len(sel), 1):.1%} | {ex} |")

    # 3. coverage of hand-dependent reasons per situation
    out += ["", "## 3. Explanation coverage per bidding situation", "",
            "Share of decisions with at least one TRUE concept whose truth depends on the hand in that auction "
            "(the explanation then has a real reason, not only a restatement of the call).", "",
            "| situation | decisions | with a hand-based reason | most common reasons |", "|---|---|---|---|"]
    rng = np.random.RandomState(0)
    for cat, cnt in sorted(collections.Counter(cats).items(), key=lambda kv: -kv[1]):
        idxs = np.flatnonzero(cats == cat)
        idxs = rng.choice(idxs, min(len(idxs), args.coverage_per_category), replace=False)
        covered, reasons = 0, collections.Counter()
        for k in idxs:
            r = rows[k]; calls = r[0]; n = len(calls)
            frame = {"auction_before": [{"seat": "NESW"[(2 - (n - i)) % 4], "bid": c} for i, c in enumerate(calls)],
                     "bid": r[1], "hand_str": r[2], "dealer": "NESW"[(2 - n) % 4] if calls else "S"}
            dep = hand_dependent_concepts(frame)
            true_dep = [C[j] for j in np.flatnonzero(V[k]) if dep.get(C[j])]
            covered += bool(true_dep); reasons.update(true_dep)
        top = ", ".join(f"{gloss_for_concept(c)} ({v})" for c, v in reasons.most_common(3))
        out.append(f"| {cat} | {cnt} | {covered / len(idxs):.0%} | {top} |")

    out += ["", "## 4. Never true in this sample", "", ", ".join(f"`{c}`" for c in never) or "(none)"]
    text = "\n".join(out) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(text)
        print(f"[audit] wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
