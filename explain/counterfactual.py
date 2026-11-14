"""Counterfactual explanations:
  * `search_minimal_flip`: the smallest set of concepts whose flip changes the concepts-only head's call (exact
    for k <= 2, since the head is linear; a shortlist for k = 3);
  * `card_counterfactuals`: the exhaustive search over all 507 one-card swaps in one batched forward pass. Per
    hand card it keeps the largest |Delta p(call)|, breaking near-ties towards the smallest change."""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from concepts.hand import RANKS, SUIT_ORDER as SUITS, decode_hand, replace_card
from concepts.types import HandFeatures

SUIT_SYMBOL = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
SUIT_NAMES = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}
HCP_MAP = {"A": 4, "K": 3, "Q": 2, "J": 1}
FULL_DECK = [s + r for s in SUITS for r in RANKS]

MIN_CF_DELTA = 0.20
MAX_COUNTERFACTUALS = 2


# ============================================================
# 1. Minimal concept-flip counterfactual
# ============================================================

def _masked_argmax_rows(logits_2d: np.ndarray, legal_1d: np.ndarray) -> np.ndarray:
    z = np.where(legal_1d[None, :], logits_2d, -np.inf)
    return z.argmax(axis=-1)


def search_minimal_flip(concept_acts: np.ndarray, legal: np.ndarray, W: np.ndarray, b: np.ndarray,
                         flip_active: bool, k_max: int = 3, shortlist_n: int = 20,
                         exclude: Optional[List[int]] = None) -> Optional[List[int]]:
    """flip_active=True searches over currently-active concepts (necessary
    set, flip to 0); flip_active=False searches over currently-inactive
    concepts (sufficient set, flip to 1). Returns the smallest witnessing
    index list found (indices into the full 0..num_hard-1 concept space),
    or None if no set up to size k_max changes the argmax.

    `exclude`: concept indices never flipped -- pass the pure call
    descriptors (bid_is_in_<strain>, see concepts.vocabulary), otherwise the
    "necessary" answer is nearly always the trivial "it is a bid in spades"."""
    is_active = concept_acts > 0.5
    idx = np.where(is_active == flip_active)[0]
    if exclude:
        idx = np.setdiff1d(idx, np.asarray(exclude, dtype=int))
    if len(idx) == 0:
        return None

    target = 0.0 if flip_active else 1.0
    base_logits = concept_acts @ W + b
    base_bid = int(_masked_argmax_rows(base_logits[None, :], legal)[0])
    deltas = (target - concept_acts[idx])[:, None] * W[idx, :]

    k1_logits = base_logits[None, :] + deltas
    k1_argmax = _masked_argmax_rows(k1_logits, legal)
    hits = np.where(k1_argmax != base_bid)[0]
    if len(hits) > 0:
        return [int(idx[hits[0]])]
    if k_max < 2:
        return None

    n = len(idx)
    pair_logits = base_logits[None, None, :] + deltas[:, None, :] + deltas[None, :, :]
    pair_argmax = _masked_argmax_rows(pair_logits.reshape(n * n, -1), legal).reshape(n, n)
    iu, ju = np.triu_indices(n, k=1)
    hit_mask = pair_argmax[iu, ju] != base_bid
    hit_pos = np.where(hit_mask)[0]
    if len(hit_pos) > 0:
        a, c = iu[hit_pos[0]], ju[hit_pos[0]]
        return [int(idx[a]), int(idx[c])]
    if k_max < 3:
        return None

    mags = np.abs(deltas).sum(axis=1)
    top = np.argsort(-mags)[:shortlist_n]
    combos = np.array(list(itertools.combinations(range(len(top)), 3)))
    if len(combos) == 0:
        return None
    triple_logits = base_logits[None, :] + deltas[top[combos]].sum(axis=1)
    triple_argmax = _masked_argmax_rows(triple_logits, legal)
    hits3 = np.where(triple_argmax != base_bid)[0]
    if len(hits3) > 0:
        a, b2, c = combos[hits3[0]]
        return [int(idx[top[a]]), int(idx[top[b2]]), int(idx[top[c]])]
    return None


def flip_shortlist_caveat(flip_set: Optional[List[str]]) -> str:
    """The caveat to print after a k=3 flip set ("" otherwise). search_minimal_flip() is exact for k <= 2
    but uses a magnitude shortlist for k=3. On weak/passing hands most concepts are inactive near zero,
    so delta_i ~= W[i, :] whatever the hand, and the k=3 set is the same few high-weight concepts across
    many decisions: less hand-specific than it reads."""
    if flip_set and len(flip_set) == 3:
        return (" (k=3, found via a magnitude-based shortlist -- on hands where "
                 "most concepts are already inactive this set is often the same "
                 "regardless of the specific hand; treat it as 'generally "
                 "high-leverage concepts' more than a hand-specific insight)")
    return ""


# ============================================================
# 2. Card-swap counterfactuals
# ============================================================

def cf_sentence(card: str, target: str, delta: float, new_bid: Optional[str],
                 new_conf: Optional[float], cf_hcp: int, cf_shape: str, probs_action: float) -> str:
    card_sym, target_sym = SUIT_SYMBOL[card[0]] + card[1], SUIT_SYMBOL[target[0]] + target[1]
    card_suit, target_suit = card[0], target[0]
    card_hcp, target_hcp = HCP_MAP.get(card[1], 0), HCP_MAP.get(target[1], 0)

    if card_suit == target_suit:
        suit_name = SUIT_NAMES[card_suit]
        verb = "upgrading" if target_hcp > card_hcp else "downgrading"
        change = f"{verb} {card_sym} to {target_sym} in {suit_name}"
    else:
        change = f"swapping {card_sym} for {target_sym} ({SUIT_NAMES[card_suit]} → {SUIT_NAMES[target_suit]})"

    shape_str = " - ".join(f"{SUIT_SYMBOL[s]}{n}" for s, n in zip(SUITS, cf_shape.split("=")))
    article = "an" if cf_hcp in (8, 11, 18) else "a"  # "an 8 HCP hand", "an 11", "an 18"
    hand_desc = f"giving {article} {cf_hcp} HCP hand ({shape_str})"
    if new_bid:
        new_bid = {"P": "Pass", "X": "Double", "XX": "Redouble"}.get(new_bid, new_bid)

    pct = int(round(abs(delta) * 100))
    new_conf_pct = int(round((probs_action + delta) * 100))
    if new_bid and new_conf is not None:
        outcome = (f"confidence in the original bid drops by {pct}% to {new_conf_pct}%, "
                   f"and the model switches to {new_bid} ({int(round(new_conf*100))}% confident)")
    else:
        direction = "stronger" if delta > 0 else "weaker"
        outcome = f"confidence in this bid would be {pct}% {direction} (new confidence: {new_conf_pct}%)"
    return f"If {change} — {hand_desc} — {outcome}."


def softmax_masked(logits, legal_):
    """Softmax over the legal calls (last axis)."""
    z = np.where(legal_, logits, -1e9)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def all_single_swaps(obs: np.ndarray, hand: HandFeatures) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Every hand card replaced by every card not in the hand: (507, obs_dim) and the (card, swap) pairs."""
    held = [s + r for s, rs in hand.cards.items() for r in rs]
    held_set = set(held)
    pairs = [(c, u) for c in held for u in FULL_DECK if u not in held_set]
    return np.stack([replace_card(obs, c, u) for c, u in pairs]), pairs


def best_swap_per_card(pairs: List[Tuple[str, str]], delta: np.ndarray, tol: float = 0.02) -> List[int]:
    """Per hand card, the index of its most impactful replacement; among
    replacements within `tol` of that |delta|, the one changing the hand
    least (|HCP change|, then a replacement in the same suit, then rank distance)."""
    by_card: Dict[str, List[int]] = {}
    for k, (c, _) in enumerate(pairs):
        by_card.setdefault(c, []).append(k)
    out = []
    for c, ks in by_card.items():
        ks = np.asarray(ks)
        top = np.abs(delta[ks]).max()
        near = [k for k in ks if abs(delta[k]) >= top - tol]

        def cost(k):
            card, swap = pairs[k]
            return (abs(HCP_MAP.get(swap[1], 0) - HCP_MAP.get(card[1], 0)), swap[0] != card[0],
                    abs(RANKS.index(swap[1]) - RANKS.index(card[1])))
        out.append(min(near, key=cost))
    return out


def card_counterfactuals(obs: np.ndarray, hand: HandFeatures, legal: np.ndarray, action: int,
                          probs: np.ndarray, policy_logits_fn, bid: str, bids: List[str]) -> List[Dict]:
    """Up to MAX_COUNTERFACTUALS card swaps that move p(call) by at least MIN_CF_DELTA, as sentences.
    `policy_logits_fn(obs_batch) -> (B, num_actions)` keeps this module independent of the checkpoint."""
    batch, pairs = all_single_swaps(np.asarray(obs, np.float32), hand)
    probs_cf = softmax_masked(np.asarray(policy_logits_fn(batch), np.float64), legal[None])
    delta = probs_cf[:, action] - float(probs[action])
    picked = [k for k in best_swap_per_card(pairs, delta) if abs(delta[k]) >= MIN_CF_DELTA]
    picked.sort(key=lambda k: -abs(delta[k]))
    results, seen = [], set()
    for k in picked:
        if len(results) >= MAX_COUNTERFACTUALS:
            break
        card, target = pairs[k]
        hand_cf = decode_hand(batch[k])
        new_action = int(np.argmax(probs_cf[k]))
        # two swaps that leave the same hand pattern, HCP and call say the same thing
        # ("D7 for HQ" and "D3 for HQ"): report distinct changes
        what = (tuple(hand_cf.lengths[s] for s in SUITS), hand_cf.hcp, new_action)
        if what in seen:
            continue
        seen.add(what)
        new_bid_str = bids[new_action] if new_action != action else None
        new_conf_val = round(float(probs_cf[k, new_action]), 2) if new_action != action else None
        cf_shape = "=".join(str(hand_cf.lengths[s]) for s in SUITS)
        sentence = cf_sentence(card, target, float(delta[k]), new_bid_str, new_conf_val,
                               hand_cf.hcp, cf_shape, float(probs[action]))
        entry = {"sentence": sentence, "card": card, "swap": target, "delta": round(float(delta[k]), 2)}
        if new_bid_str:
            entry["new_bid"], entry["new_conf"] = new_bid_str, new_conf_val
        results.append(entry)
    return results
