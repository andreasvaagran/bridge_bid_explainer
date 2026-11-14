"""Statistics over the 13 x 39 = 507 one-card swaps: how many keep the call, how often replacing each card
changes it, what each other call would need, and (concept models) the concepts that flip on those swaps."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from concepts.hand import RANKS, decode_hand
from explain.counterfactual import HCP_MAP, all_single_swaps, softmax_masked

HAND_ORDER = "SHDC"


def _cost(out_c: str, in_c: str) -> tuple:
    """How much a swap changes the hand: |HCP change|, then a suit change, then rank distance."""
    return (abs(HCP_MAP.get(in_c[1], 0) - HCP_MAP.get(out_c[1], 0)), in_c[0] != out_c[0],
            abs(RANKS.index(in_c[1]) - RANKS.index(out_c[1])))


def _shape(h) -> str:
    return "-".join(str(h.lengths[s]) for s in HAND_ORDER)


def card_swap_stats(obs: np.ndarray, legal: np.ndarray, action: int,
                    outputs_batch: Callable[[np.ndarray], Dict[str, np.ndarray]], bids: Sequence[str],
                    concept_names: Optional[Sequence[str]] = None, glosses: Optional[Callable[[str], str]] = None,
                    top_calls: int = 5, top_concepts: int = 3,
                    focus_concepts: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Statistics of the call over every one-card change of the hand.

    outputs_batch(obs (B, 480)) -> {"policy_logits": (B, 38)[, "concept_acts": (B, C)]}.
    concept_names / glosses: for the concept models, to report the concepts that
    flip on the swaps that lead to each other call."""
    obs = np.asarray(obs, np.float32)[:480]
    legal = np.asarray(legal, bool)
    hand = decode_hand(obs)
    batch, pairs = all_single_swaps(obs, hand)
    base = outputs_batch(obs[None])
    out = outputs_batch(batch)
    p0 = softmax_masked(np.asarray(base["policy_logits"], np.float64), legal)[0]
    p = softmax_masked(np.asarray(out["policy_logits"], np.float64), legal)
    new = p.argmax(1)
    changed = new != action
    hcp_delta = np.array([HCP_MAP.get(i[1], 0) - HCP_MAP.get(o[1], 0) for o, i in pairs])

    cards = []
    for s in HAND_ORDER:
        for r in hand.cards[s]:
            c = s + r
            ks = np.array([k for k, (o, _) in enumerate(pairs) if o == c])
            ch = changed[ks]
            to = Counter(bids[int(a)] for a in new[ks][ch])
            cards.append({"card": c, "changed": int(ch.sum()), "of": int(len(ks)), "share": round(float(ch.mean()), 3),
                          "mean_dp": round(float((p[ks, action] - p0[action]).mean()), 3),
                          "to": [{"call": k, "n": int(v)} for k, v in to.most_common(3)]})

    acts0 = acts = None
    if concept_names is not None and "concept_acts" in out:
        acts0 = np.asarray(base["concept_acts"])[0] >= 0.5
        acts = np.asarray(out["concept_acts"]) >= 0.5
    calls = []
    for a, n in Counter(int(x) for x in new[changed]).most_common(top_calls):
        ks = np.flatnonzero(new == a)
        k = min(ks, key=lambda k: _cost(*pairs[k]))
        h = decode_hand(batch[k])
        entry = {"call": bids[a], "n": int(len(ks)),
                 "simplest": {"out": pairs[k][0], "in": pairs[k][1], "hcp": int(h.hcp), "shape": _shape(h),
                              "p": round(float(p[k, a]), 3)},
                 "cards": [{"card": c, "n": int(v)} for c, v in Counter(pairs[k2][0] for k2 in ks).most_common(4)],
                 "hcp_change": [int(hcp_delta[ks].min()), int(hcp_delta[ks].max())]}
        if acts is not None:
            flips = []
            for i in range(len(concept_names)):
                name = concept_names[i]
                if name.startswith("bid_is_in_"):
                    continue
                share = float((acts[ks, i] != acts0[i]).mean())
                if share >= 0.5:
                    flips.append((share, name, "off" if acts0[i] else "on"))
            flips.sort(key=lambda x: -x[0])
            entry["concepts"] = [{"name": n, "gloss": glosses(n) if glosses else n, "turns": d, "share": round(sh, 2)}
                                 for sh, n, d in flips[:top_concepts]]
        calls.append(entry)

    # the cards behind each focus concept (the explanation's reasons): per held card, how many of
    # its 39 replacements switch the concept in the model's reading of the hand
    behind = []
    if acts is not None and focus_concepts:
        for name in focus_concepts:
            if name not in concept_names:
                continue
            i = list(concept_names).index(name)
            flip = acts[:, i] != acts0[i]
            per = Counter(pairs[k][0] for k in np.flatnonzero(flip))
            behind.append({"name": name, "flips": int(flip.sum()),
                           "cards": [{"card": c, "n": int(v), "of": 39} for c, v in per.most_common(4)]})

    by_delta = []
    for dlt in range(-4, 5):
        m = hcp_delta == dlt
        if not m.any():
            continue
        to = Counter(bids[int(a)] for a in new[m & changed])
        by_delta.append({"delta": dlt, "n": int(m.sum()), "changed": int((m & changed).sum()),
                         "to": to.most_common(1)[0][0] if to else None})
    return {"call": bids[action], "p": round(float(p0[action]), 3), "n_swaps": int(len(pairs)),
            "kept": int((~changed).sum()), "kept_share": round(float((~changed).mean()), 3),
            "cards": cards, "calls": calls, "by_hcp_change": by_delta, "concept_cards": behind}


def summary_sentence(stats: Dict[str, Any]) -> str:
    """One plain sentence: how robust the call is and which cards carry it."""
    call = {"P": "Pass", "X": "Double", "XX": "Redouble"}.get(stats["call"], stats["call"])
    kept, n = stats["kept"], stats["n_swaps"]
    key = sorted((c for c in stats["cards"] if c["changed"]), key=lambda c: -c["share"])[:2]
    s = f"{call} survives {kept} of the {n} one-card changes of this hand"
    if key:
        s += "; it depends most on the " + " and the ".join(
            f"{_sym(c['card'])} (replacing it changes the call {c['changed']} of {c['of']} times)" for c in key)
    return s + "."


def _sym(card: str) -> str:
    return {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}[card[0]] + ("10" if card[1] == "T" else card[1])
