"""The concepts package: vocabulary and rules match one to one, the suit-choice predicates' edge cases, and
the card decoder's round trip."""
from __future__ import annotations

import numpy as np
import pytest

from concepts.hand import card_bit_index, decode_hand, SUIT_ORDER, RANKS
from concepts.rules import build_sayc_state_rules
from concepts.types import AuctionContext, HandFeatures
from concepts.vocabulary import HARD_STATE_CONCEPTS
from concepts.predicates_hand import (
    opens_longer_minor, opens_longer_major, prefers_notrump_over_major,
)
from concepts.predicates_auction import is_reverse_bid


# ============================================================
# 1. Vocabulary <-> rule round-trip
# ============================================================

def test_every_concept_has_a_rule_and_vice_versa():
    rules = build_sayc_state_rules()
    rule_names = {r.name for r in rules}
    concept_names = set(HARD_STATE_CONCEPTS)

    assert rule_names == concept_names, (
        f"missing rules for: {concept_names - rule_names}; "
        f"extra rules with no vocabulary entry: {rule_names - concept_names}"
    )
    assert len(HARD_STATE_CONCEPTS) == 146
    assert len(rules) == 146


def test_vocabulary_is_append_only():
    """The first 115 names are the hard_state_head columns of the earliest
    checkpoints; they may never be reordered, renamed or
    removed (retired concepts stay, defined as never-true). Pinned by hash so
    an accidental insert/rename fails loudly."""
    import hashlib
    import json
    first = HARD_STATE_CONCEPTS[:115]
    assert hashlib.sha256(json.dumps(first).encode()).hexdigest() == (
        "ee3239e4b49a0cb2d4c7b043efe1e454c6c78cecc69be1ed18d57ffac1b9352f")


# ============================================================
# 2. Suit-choice predicates: documented tie-breaking edge cases
# ============================================================

def _hand(lengths: dict, hcp: int = 12, cards=None) -> HandFeatures:
    """Build a minimal HandFeatures for predicate testing. `lengths` must
    sum to 13. `cards` (optional) overrides specific suits' card lists
    when a predicate needs to look at actual honors, not just counts."""
    assert sum(lengths.values()) == 13, "a hand must have 13 cards"
    cards = cards or {}
    full_cards = {}
    for suit in SUIT_ORDER:
        if suit in cards:
            full_cards[suit] = cards[suit]
        else:
            n = lengths.get(suit, 0)
            full_cards[suit] = list(RANKS[-n:]) if n else []  # arbitrary high cards, count-only tests
    stoppers = {s: False for s in SUIT_ORDER}
    return HandFeatures(hcp=hcp, lengths=lengths, balanced=False,
                         stoppers=stoppers, passed_hand=False, cards=full_cards)


def _opening_1level(bid: str) -> AuctionContext:
    return AuctionContext(bids=[], bid=bid, dealer=0)


def test_opens_longer_minor_tied_3_3_opens_clubs():
    h = _hand({"S": 4, "H": 3, "D": 3, "C": 3})
    assert opens_longer_minor(_opening_1level("1C"), h) is True
    assert opens_longer_minor(_opening_1level("1D"), h) is False


def test_opens_longer_minor_tied_4_4_opens_diamonds():
    h = _hand({"S": 2, "H": 3, "D": 4, "C": 4})
    assert opens_longer_minor(_opening_1level("1D"), h) is True
    assert opens_longer_minor(_opening_1level("1C"), h) is False


def test_opens_longer_minor_strictly_longer_wins():
    h = _hand({"S": 3, "H": 3, "D": 2, "C": 5})
    assert opens_longer_minor(_opening_1level("1C"), h) is True
    assert opens_longer_minor(_opening_1level("1D"), h) is False


def test_opens_longer_major_tied_5_5_opens_spades():
    h = _hand({"S": 5, "H": 5, "D": 2, "C": 1})
    assert opens_longer_major(_opening_1level("1S"), h) is True
    assert opens_longer_major(_opening_1level("1H"), h) is False


def test_opens_longer_major_strictly_longer_wins():
    h = _hand({"S": 5, "H": 6, "D": 1, "C": 1})
    assert opens_longer_major(_opening_1level("1H"), h) is True
    assert opens_longer_major(_opening_1level("1S"), h) is False


def test_opens_longer_major_requires_5_card_minimum_each():
    # Only 4 spades -- doesn't qualify as "both majors 5+", must not fire.
    h = _hand({"S": 4, "H": 5, "D": 2, "C": 2})
    assert opens_longer_major(_opening_1level("1S"), h) is False
    assert opens_longer_major(_opening_1level("1H"), h) is False


def test_prefers_notrump_over_major_balanced_15_17_with_5_card_major():
    h = HandFeatures(
        hcp=16, lengths={"S": 5, "H": 3, "D": 3, "C": 2}, balanced=True,
        stoppers={s: False for s in SUIT_ORDER}, passed_hand=False,
        cards={"S": ["A", "K", "Q", "J", "9"], "H": ["A", "K", "2"],
               "D": ["A", "3", "2"], "C": ["4", "2"]},
    )
    assert prefers_notrump_over_major(_opening_1level("1N"), h) is True


def test_prefers_notrump_over_major_false_outside_range():
    h = HandFeatures(
        hcp=20, lengths={"S": 5, "H": 3, "D": 3, "C": 2}, balanced=True,
        stoppers={s: False for s in SUIT_ORDER}, passed_hand=False,
        cards={"S": ["A", "K", "Q", "J", "9"], "H": ["A", "K", "2"],
               "D": ["A", "3", "2"], "C": ["4", "2"]},
    )
    assert prefers_notrump_over_major(_opening_1level("1N"), h) is False


def test_is_reverse_bid_true_for_genuine_reverse():
    # 1C (opening) - P - 1H (response) - P - 2S (rebid): 2S ranks higher
    # than 1C's clubs, no jump, at the 2-level -- a genuine reverse.
    a = AuctionContext(bids=["1C", "P", "1H", "P"], bid="2S", dealer=0)
    assert is_reverse_bid(a) is True


def test_is_reverse_bid_false_for_same_suit_rebid():
    # 1C - P - 1H - P - 2C: rebidding the SAME suit is not a reverse.
    a = AuctionContext(bids=["1C", "P", "1H", "P"], bid="2C", dealer=0)
    assert is_reverse_bid(a) is False


def test_is_reverse_bid_false_for_lower_ranking_new_suit():
    # 1S - P - 2C - P - 2D: diamonds ranks BELOW spades, not a reverse.
    a = AuctionContext(bids=["1S", "P", "2C", "P"], bid="2D", dealer=0)
    assert is_reverse_bid(a) is False


# ============================================================
# 3. Canonical card decoder round-trip
# ============================================================

ALL_CARDS = [s + r for s in SUIT_ORDER for r in RANKS]


@pytest.mark.parametrize("card", ALL_CARDS)
def test_card_bit_index_round_trips_through_decode_hand(card):
    obs = np.zeros(480, dtype=np.float32)
    obs[-52 + card_bit_index(card)] = 1.0
    hand = decode_hand(obs)
    suit, rank = card[0], card[1]
    assert hand.cards[suit] == [rank], (
        f"setting only {card}'s bit should decode to exactly that card, "
        f"got {hand.lengths} / {hand.cards}"
    )
    assert sum(hand.lengths.values()) == 1


def test_card_bit_index_is_a_bijection():
    indices = {card_bit_index(c) for c in ALL_CARDS}
    assert len(indices) == 52
    assert indices == set(range(52))


def test_decode_hand_full_hand_hcp():
    obs = np.zeros(480, dtype=np.float32)
    hand_cards = ["SA", "SK", "SQ", "S2", "H2", "H3", "H4", "H5",
                  "D2", "D3", "D4", "C2", "C3"]
    for c in hand_cards:
        obs[-52 + card_bit_index(c)] = 1.0
    hand = decode_hand(obs)
    assert hand.hcp == 9  # A+K+Q = 4+3+2
    assert hand.lengths == {"S": 4, "H": 4, "D": 3, "C": 2}


# ============================================================
# 5. The rules in English (explain.rule_justification.rule_conditions, the interface's rule view)
# ============================================================

def test_every_rule_renders_in_english_and_agrees_with_the_engine():
    """The Weights tab shows each concept's rule condition by condition. Every condition must render in words
    (no lambda source, also for trailing comments, one-line StateRule entries and closure names such as ACES) and
    the rule view must reach the engine's own verdict, bid_is_splinter's two parts too."""
    import re
    from concepts.rules import create_sayc_state_authority
    from explain.rule_justification import rule_conditions
    from nlg.explanation_inputs import hand_from_str
    auth = create_sayc_state_authority()
    positions = [(["1D"], "1S", "S:AQJ32 H:K32 D:32 C:Q32"), (["P", "1H", "P"], "2H", "S:A32 H:K32 D:J432 C:Q32"),
                 (["1C", "1S"], "X", "S:32 H:KQ32 D:AJ32 C:Q32"), (["1H", "P", "4N", "P"], "5D", "S:A2 H:KQ32 D:AJ32 C:Q32"),
                 (["1S", "P"], "4C", "S:KQ32 H:A32 D:AKJ32 C:-"), ([], "1N", "S:AQ532 H:K3 D:Q32 C:A2")]
    for bids, bid, hand in positions:
        a, h = AuctionContext(bids=bids, bid=bid, dealer=0), hand_from_str(hand)
        truth = auth.classify(a, h)
        for name in HARD_STATE_CONCEPTS:
            r = rule_conditions(name, a, h)
            assert r["holds"] == truth[name], (name, bids, bid)
            for c in (c for g in r["groups"] for c in g):
                assert not re.search(r"`|\ba\.|\bh\.|[a-z_]\(|internal rule", c["text"]), (name, c["text"])
    assert rule_conditions("bid_is_roman_keycard", a, h)["retired"]
    assert len(rule_conditions("bid_is_splinter", a, h)["groups"]) == 2
