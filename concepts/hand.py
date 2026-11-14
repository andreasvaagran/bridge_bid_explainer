"""The single decoder of the observation's 52 hand bits (`decode_hand`) and its inverse for writing
counterfactual hands (`replace_card`). Both derive from `card_bit_index`, so they cannot disagree. The layout is
interleaved: bit i is rank i // 4, suit "SHDC"[3 - i % 4]."""
from __future__ import annotations

import numpy as np

from .types import HandFeatures

RANKS = "23456789TJQKA"
SUIT_ORDER = "SHDC"
HCP_VALUES = {"A": 4, "K": 3, "Q": 2, "J": 1}


def card_bit_index(card: str) -> int:
    """`card` is a 2-character string, suit then rank (e.g. "SA" = spade
    ace, "HT" = heart ten). Returns the index into the observation's last
    52 entries that represents holding this card."""
    suit, rank = card[0], card[1]
    rank_idx = RANKS.index(rank)
    suit_idx = SUIT_ORDER.index(suit)
    return rank_idx * 4 + (3 - suit_idx)


# The read side is derived from card_bit_index, so the two directions cannot drift apart.
_ALL_CARDS = [s + r for s in SUIT_ORDER for r in RANKS]
_INDEX_TO_CARD = {card_bit_index(c): c for c in _ALL_CARDS}
assert len(_INDEX_TO_CARD) == 52, "card_bit_index must be a bijection over all 52 cards"


def is_balanced_shape(lengths: dict) -> bool:
    """True bridge "balanced": no singleton/void, at most one doubleton
    (i.e. shape is 4-3-3-3, 4-4-3-2, or 5-3-3-2)."""
    vals = sorted(lengths.values())
    if vals[0] <= 1:
        return False
    return sum(1 for v in vals if v == 2) <= 1


def is_stopper(ranks) -> bool:
    """Standard notrump stopper: A, Kx, Qxx or Jxxx (a bare K or Qx does not count)."""
    n = len(ranks)
    return "A" in ranks or ("K" in ranks and n >= 2) or ("Q" in ranks and n >= 3) or ("J" in ranks and n >= 4)


def decode_hand(obs: np.ndarray) -> HandFeatures:
    """Decode the last 52 entries of a bridge-bidding observation vector
    into a HandFeatures. `obs` may be longer than 52; only the last 52
    entries are read."""
    bits = obs[-52:]
    cards_by_suit = {s: [] for s in SUIT_ORDER}
    for i in range(52):
        if bits[i] > 0:
            card = _INDEX_TO_CARD[i]
            cards_by_suit[card[0]].append(card[1])

    lengths = {s: len(cards_by_suit[s]) for s in SUIT_ORDER}
    hcp = sum(HCP_VALUES.get(r, 0) for suit in cards_by_suit.values() for r in suit)
    balanced = is_balanced_shape(lengths)
    stoppers = {s: is_stopper(cards_by_suit[s]) for s in SUIT_ORDER}
    return HandFeatures(hcp=hcp, lengths=lengths, balanced=balanced,
                         stoppers=stoppers, passed_hand=False, cards=cards_by_suit)


def replace_card(obs: np.ndarray, old: str, new: str) -> np.ndarray:
    """Return a copy of `obs` with `old` removed from the hand and `new`
    added: the way to build a counterfactual observation (card swaps,
    attribution baselines). Uses the same `card_bit_index` as
    `decode_hand`, so a swap lands on the bit that is read back."""
    obs = obs.copy()
    obs[-52 + card_bit_index(old)] = 0.0
    obs[-52 + card_bit_index(new)] = 1.0
    return obs
