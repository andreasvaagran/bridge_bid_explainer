"""The card-swap engine searches all 507 swaps and, among equally decisive ones, reports the smallest
change."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from concepts.hand import card_bit_index, decode_hand
from explain.counterfactual import all_single_swaps, best_swap_per_card, card_counterfactuals
from concepts.hand import replace_card


def _obs(hand):
    obs = np.zeros(480, np.float32)
    for s, rs in hand.items():
        for r in rs:
            obs[428 + card_bit_index(s + r)] = 1
    return obs


HAND = {"S": "AJ98754", "H": "T", "D": "QJ73", "C": "J"}


def test_all_single_swaps_is_exhaustive_and_valid():
    obs = _obs(HAND)
    batch, pairs = all_single_swaps(obs, decode_hand(obs))
    assert batch.shape == (507, 480) and len(set(pairs)) == 507
    for k in (0, 200, 506):
        h = decode_hand(batch[k])
        assert sum(h.lengths.values()) == 13
        card, swap = pairs[k]
        assert swap[1] in h.cards[swap[0]] and card[1] not in h.cards[card[0]]


def test_ties_are_broken_towards_the_smallest_change():
    pairs = [("SA", "HA"), ("SA", "H2"), ("SA", "SK"), ("S4", "H2")]
    delta = np.array([-0.98, -0.97, -0.50, -0.30])
    best = best_swap_per_card(pairs, delta)
    # for SA: HA and H2 are within 0.02 of the best; HA keeps the HCP (0 change) -> chosen
    assert pairs[best[0]] == ("SA", "HA")


def test_engine_reports_the_most_decisive_cards():
    obs = _obs(HAND)
    hand = decode_hand(obs)
    legal = np.ones(38, bool)

    def logits(batch):  # a toy policy: call 0 iff the hand holds seven or more spades
        out = np.zeros((len(batch), 38))
        n_spades = np.array([decode_hand(b).lengths["S"] for b in batch])
        out[:, 0] = np.where(n_spades >= 7, 5.0, -5.0)
        return out
    probs = np.exp(logits(obs[None])[0]); probs /= probs.sum()
    cfs = card_counterfactuals(obs, hand, legal, 0, probs, logits, "4S", [str(i) for i in range(38)])
    assert cfs and all(c["card"][0] == "S" and c["swap"][0] != "S" for c in cfs)
    assert all(c["delta"] <= -0.2 for c in cfs)


def test_reported_counterfactuals_are_distinct_changes():
    """Two swaps that leave the same hand pattern, HCP and call ("D7 for HQ",
    "D3 for HQ") say the same thing; only one is reported."""
    obs = _obs({"S": "42", "H": "KJT2", "D": "T732", "C": "542"})
    hand = decode_hand(obs)
    legal = np.ones(38, bool)

    def logits(batch):  # call 0 unless the hand has five or more hearts
        out = np.zeros((len(batch), 38))
        out[:, 0] = np.array([-5.0 if decode_hand(b).lengths["H"] >= 5 else 5.0 for b in batch])
        return out
    probs = np.exp(logits(obs[None])[0]); probs /= probs.sum()
    cfs = card_counterfactuals(obs, hand, legal, 0, probs, logits, "P", [str(i) for i in range(38)])
    shapes = set()
    for c in cfs:
        h = decode_hand(replace_card(obs, c["card"], c["swap"]))
        key = (tuple(h.lengths[s] for s in "SHDC"), h.hcp)
        assert key not in shapes
        shapes.add(key)
