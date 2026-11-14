"""Predicates that need the hand (`HandFeatures`): shape, LTC, quick tricks, Rule of 20, support
points, preempt quality, and the suit-choice predicates (why this suit and not another that qualified)."""
from __future__ import annotations

from .types import AuctionContext, HandFeatures
from .predicates_auction import is_opening, is_response


# ============================================================
# Hand shape helpers
# ============================================================

def has_singleton(h: HandFeatures) -> bool:
    return 1 in h.lengths.values()


def has_void(h: HandFeatures) -> bool:
    return 0 in h.lengths.values()


def is_semi_balanced(h: HandFeatures) -> bool:
    lens = sorted(h.lengths.values(), reverse=True)
    return lens in [[5, 4, 2, 2], [6, 3, 2, 2], [5, 3, 3, 2]]


def suit_quality(h: HandFeatures, suit: str) -> int:
    if suit not in h.cards:
        return 0
    return sum(1 for r in ("A", "K", "Q") if r in h.cards[suit])


def denies_major_fit_after_1nt_response(a: AuctionContext, h: HandFeatures) -> bool:
    """Responder bids 1N over partner's 1H/1S opening without 3+ support.

    Shared by bid_denies_major_fit and bid_denies_3_card_support, which stay
    two separate concepts in the vocabulary although the signal is the same.
    """
    return (
        is_response(a)
        and a.bid in ("1N", "1NT")
        and a.partner_last_bid() in ("1H", "1S")
        and h.lengths.get(a.partner_last_bid()[-1], 0) < 3
    )


# ============================================================
# Opening suit-choice predicates
#
# The structural StateRules for bid_is_1_level_opening/bid_shows_minor_suit/
# bid_shows_major_suit only describe the CHOSEN bid's own properties -- none
# of them compare the hand's actual suit lengths to explain why THIS strain
# was chosen over an alternative (e.g. why 1D and not 1C). These three
# predicates fill that gap, each checked against real WBridge5 training
# data (`training/concept_prevalence.py`).
# ============================================================

def opens_longer_minor(a: AuctionContext, h: HandFeatures) -> bool:
    """1C/1D opening choice: open the longer minor; when tied, 3-3 opens
    1C, 4-4+ opens 1D. Confirmed against 3,001 real WBridge5 1C/1D
    openings: 100% consistent (1233/1233 clubs-longer -> 1C, 1295/1295
    diamonds-longer -> 1D, 193/193 tied-at-3 -> 1C, 280/280 tied-at-4-or-5
    -> 1D)."""
    if not (is_opening(a) and a.bid in ("1C", "1D")):
        return False
    c, d = h.lengths.get("C", 0), h.lengths.get("D", 0)
    if a.bid == "1C":
        return c > d or (c == d and c == 3)
    return d > c or (c == d and c >= 4)


def opens_longer_major(a: AuctionContext, h: HandFeatures) -> bool:
    """1H/1S opening choice when both majors qualify (5+ cards each):
    open the longer; when tied, open 1S. Confirmed against 93 real
    WBridge5 1H/1S openings with 5+/5+ both majors: 100% consistent
    (15/15 + 8/8 longer-wins, 70/70 tied -> 1S)."""
    if not (is_opening(a) and a.bid in ("1H", "1S")):
        return False
    h_len, s_len = h.lengths.get("H", 0), h.lengths.get("S", 0)
    if h_len < 5 or s_len < 5:
        return False
    if a.bid == "1S":
        return s_len >= h_len
    return h_len > s_len


def prefers_notrump_over_major(a: AuctionContext, h: HandFeatures) -> bool:
    """1NT opening choice even when a 5-card major also qualifies:
    balanced 15-17 HCP takes priority over opening the major. Confirmed
    against 181 real WBridge5 hands qualifying for both: opened 1NT in
    180/181 (99.4%)."""
    if not (is_opening(a) and a.bid in ("1N", "1NT")):
        return False
    if not ((h.balanced or is_semi_balanced(h)) and 15 <= h.hcp <= 17):
        return False
    return h.lengths.get("H", 0) >= 5 or h.lengths.get("S", 0) >= 5


# ============================================================
# Classic hand-evaluation heuristics
#
# These widen the concept vocabulary beyond textbook SAYC bid-meaning
# rules (what a bid promises) into general hand-evaluation theory (how a
# hand's raw quality is judged before choosing a bid). Thresholds below
# are the standard textbook ones, confirmed non-degenerate against real
# WBridge5 training data with a prevalence audit (`training/concept_prevalence.py`),
# conditioned on each concept's parent situation (not raw prevalence over
# all decisions, which is misleadingly small since e.g. 1-level openings
# are themselves only ~7.4% of all decisions):
#   - rule-of-20 (>=20 among 1-level openings):      79.3%
#   - LTC<=7 among 1-level openings:                 86.1%
#   - quick_tricks>=2.0 among 1-level/overcalls:      95.0% (a low floor
#     by design -- meant to flag the rare hand that opens/overcalls
#     without it, not to discriminate strong hands from weak ones)
#   - support_points>=10 among raise responses:      49.8% (near-even
#     split, a good discriminating threshold)
#   - preempt quality "good" (>=2 top honors) among
#     preemptive openings/continuations:              33.1%
# None is within noise of 0% or 100%, so all keep their textbook values.
# ============================================================

def losing_trick_count(h: HandFeatures) -> int:
    """Standard "top three cards per suit" Losing Trick Count: for each
    suit, a missing A/K/Q among the cards actually held (capped at the
    suit's length, since a 2-card suit has at most 2 losers) counts as one
    loser. Lower is stronger; a classic opening hand is often LTC<=7."""
    total = 0
    for suit, length in h.lengths.items():
        cards = h.cards.get(suit, [])
        if length >= 1 and "A" not in cards:
            total += 1
        if length >= 2 and "K" not in cards:
            total += 1
        if length >= 3 and "Q" not in cards:
            total += 1
    return total


def quick_tricks(h: HandFeatures) -> float:
    """Standard quick-trick table, summed per suit: AK=2, AQ=1.5, KQ=1,
    A=1, K=0.5. Measures playing-trick strength independent of HCP, used
    (alongside HCP) to judge opening/overcalling quality."""
    total = 0.0
    for cards in h.cards.values():
        has_a, has_k, has_q = "A" in cards, "K" in cards, "Q" in cards
        if has_a and has_k:
            total += 2.0
        elif has_a and has_q:
            total += 1.5
        elif has_k and has_q:
            total += 1.0
        elif has_a:
            total += 1.0
        elif has_k:
            total += 0.5
    return total


def rule_of_20_satisfied(h: HandFeatures) -> bool:
    """Rule of 20: HCP plus the length of the two longest suits is at least
    20 -- a common "should I open a light hand" guideline that credits
    shape alongside raw HCP."""
    lens = sorted(h.lengths.values(), reverse=True)
    return h.hcp + lens[0] + lens[1] >= 20


def support_points(h: HandFeatures, trump_suit: str) -> int:
    """HCP plus standard shortness bonus (void=5, singleton=3,
    doubleton=1) in side suits once a trump fit is known -- distinguishes
    raw HCP from a hand's true worth as a supporting hand."""
    bonus = 0
    for suit, length in h.lengths.items():
        if suit == trump_suit:
            continue
        if length == 0:
            bonus += 5
        elif length == 1:
            bonus += 3
        elif length == 2:
            bonus += 1
    return h.hcp + bonus


def preempt_suit_quality_score(h: HandFeatures, suit: str) -> int:
    """Number of top honors (A/K/Q) held in the preempt suit."""
    return suit_quality(h, suit)


def wb5_points(h: HandFeatures) -> int:
    """WBridge5's "points" (notes/wbridge5_system.md, jdh8's WBridge5 book):
    HCP + distribution points (void 3, singleton 2, doubleton 1) + 1 per
    card beyond the fifth in every suit. Its weak-two/preempt/2C minimums
    are stated in these points, and the SL data fits them (every weak two
    in 150k deals has >= 7; 99.6% of 2C openings have 21+ or <= 4 losers)."""
    dist = sum({0: 3, 1: 2, 2: 1}.get(n, 0) for n in h.lengths.values())
    length = sum(max(0, n - 5) for n in h.lengths.values())
    return h.hcp + dist + length
