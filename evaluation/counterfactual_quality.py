#!/usr/bin/env python3
"""How good are the card-swap counterfactuals? Coverage, flip rate and what the swaps change; their alignment
with the rule engine's description and, for concept models, the concept-flip counterfactuals."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json
import pickle
import time
from pathlib import Path

import numpy as np

from concepts import AuctionContext, create_sayc_state_authority
from concepts.hand import HCP_VALUES, RANKS, SUIT_ORDER, card_bit_index, decode_hand
from evaluation.common import (BIDS, MODELS, cluster_bootstrap, load_test_split, log_softmax_masked,
                                  masked_argmax, policy_fn, sample_rows, selfplay)

CARDS = [s + r for s in SUIT_ORDER for r in RANKS]
BIT = np.array([card_bit_index(c) for c in CARDS])          # card -> index in the last 52 bits
HCP = np.array([HCP_VALUES.get(c[1], 0) for c in CARDS])
SUIT = np.array([SUIT_ORDER.index(c[0]) for c in CARDS])
RANK = np.array([RANKS.index(c[1]) for c in CARDS])
THRESH = 0.20
TOP = 3
RULE_SAMPLE = 16  # random swaps per decision for the rule-alignment test


def _held(obs) -> np.ndarray:
    """Indices (into CARDS) of the 13 cards in the hand."""
    return np.flatnonzero(np.asarray(obs)[-52:][BIT] > 0)


def _swap_batch(obs, held, unseen):
    """(507, 480) observations: every held card replaced by every unseen card, plus (507, 2) index pairs."""
    pairs = np.array([(c, u) for c in held for u in unseen])
    batch = np.repeat(np.asarray(obs, np.float32)[None], len(pairs), 0)
    rows = np.arange(len(pairs))
    batch[rows, 428 + BIT[pairs[:, 0]]] = 0.0
    batch[rows, 428 + BIT[pairs[:, 1]]] = 1.0
    return batch, pairs


def _lengths(cards_idx) -> np.ndarray:
    return np.bincount(SUIT[cards_idx], minlength=4)


def _balanced(l) -> bool:
    v = sorted(l)
    return v[0] >= 2 and sum(1 for x in v if x == 2) <= 1


def analyse(model: str, pos, rows, seed: int = 0, concept_flips: bool = True) -> dict:
    path = MODELS.get(model, model)
    logits_fn, kind = policy_fn(path)
    rule = create_sayc_state_authority()
    rng = np.random.default_rng(seed)
    is_concept = "concept" in kind
    ck = enr = None
    if is_concept:
        from training.checkpoint import load_checkpoint
        from nlg.explanation_frames import Enricher
        import jax
        ck = load_checkpoint(str(path))
        enr = Enricher(ck)
        fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
        Wp = np.asarray(ck.params["policy_head"]["w"], np.float64)
        Wc_policy = Wp[-len(ck.hard_concepts):]  # the policy's own concept rows (all of W in bottleneck modes)
    from explain.counterfactual import best_swap_per_card, search_minimal_flip

    recs = []
    rule_tab = collections.Counter()      # (description changed, model flipped) over random swaps
    reason_broken = []                    # per flipping per-card-best swap: did a hand-based reason of the call break?
    t0 = time.time()
    for j, i in enumerate(rows):
        obs = pos.obs[i].astype(np.float32)
        lg = pos.legal[i]
        held = _held(obs)
        if len(held) != 13:
            continue
        unseen = np.setdiff1d(np.arange(52), held)
        base_logits = logits_fn(obs[None])[0]
        logp = log_softmax_masked(base_logits, lg)
        bid = int(np.argmax(np.where(lg, base_logits, -np.inf)))
        p0 = float(np.exp(logp[bid]))
        batch, pairs = _swap_batch(obs, held, unseen)
        L = logits_fn(batch)
        lp = log_softmax_masked(L, lg[None])
        pb = np.exp(lp[:, bid])
        delta = pb - p0
        new = masked_argmax(L, lg[None])
        flips = new != bid
        # per held card: the most impactful replacement, as explain.counterfactual.best_swap_per_card picks it
        best = np.asarray(best_swap_per_card([(CARDS[c], CARDS[u]) for c, u in pairs], delta))
        # the most decisive distinct changes, as explain.counterfactual.card_counterfactuals shows them (a swap
        # that leaves the same hand pattern, HCP and call as an earlier one adds nothing)
        cf, seen = [], set()
        for k in best[np.argsort(-np.abs(delta[best]))]:
            if abs(delta[k]) < THRESH or len(cf) >= TOP:
                break
            c, u = pairs[k]
            nh = np.append(held[held != c], u)
            what = (tuple(_lengths(nh)), int(HCP[nh].sum()), int(new[k]))
            if what not in seen:
                seen.add(what); cf.append(k)

        hand_l = _lengths(held)
        props = []
        for k in cf:
            c, u = pairs[k]
            new_held = np.append(held[held != c], u)
            l2 = _lengths(new_held)
            props.append({
                "delta": float(delta[k]), "flip": bool(flips[k]), "new_bid": BIDS[new[k]],
                "d_hcp": int(HCP[u] - HCP[c]), "cross_suit": bool(SUIT[u] != SUIT[c]),
                "rank_up": bool(RANK[u] > RANK[c]), "shape_l1": int(np.abs(l2 - hand_l).sum()),
                "d_longest": int(l2.max() - hand_l.max()), "balance_flip": _balanced(hand_l) != _balanced(l2),
                "card": CARDS[c], "swap": CARDS[u],
            })

        # rule alignment on a uniform sample of swaps (and on flipping per-card-best swaps)
        before, seat, dl = pos.auction(i)
        calls = [b["bid"] for b in before]
        ctx = AuctionContext(bids=calls, bid=BIDS[bid], dealer=dl)
        hand0 = decode_hand(obs)
        gt0 = rule.classify(ctx, hand0)
        sample = rng.choice(len(pairs), size=min(RULE_SAMPLE, len(pairs)), replace=False)
        for k in sample:
            gt1 = rule.classify(ctx, decode_hand(batch[k]))
            rule_tab[(gt1 != gt0, bool(flips[k]))] += 1
        for k in best[flips[best]]:
            gt1 = rule.classify(ctx, decode_hand(batch[k]))
            # a hand-based reason of the original call: true before, false after, same auction and call
            reason_broken.append(any(gt0[n] and not gt1[n] for n in gt0))

        rec = {"row": int(i), "deal": int(pos.deal[i]), "bid": BIDS[bid], "conf": p0, "n_cf": len(cf), "cfs": props,
               "any_flip": bool(flips.any()), "n_flipping_swaps": int(flips.sum()),
               "min_abs_dhcp_flip": (int(np.abs(HCP[pairs[flips, 1]] - HCP[pairs[flips, 0]]).min())
                                     if flips.any() else None)}

        if is_concept and concept_flips:
            out = {k2: np.asarray(v[0]) for k2, v in fwd(obs[None]).items()}
            a = np.asarray(out["concept_acts"], np.float64)
            co_bid = int(np.argmax(np.where(lg, out["concept_only_logits"], -np.inf)))
            for kind_flip, active in (("necessary", True), ("sufficient", False)):
                fs = search_minimal_flip(a, lg, enr.W, enr.b, flip_active=active, k_max=3, exclude=enr.descriptor_idx)
                rec[f"{kind_flip}_size"] = len(fs) if fs else None
                if fs:  # does the same flip change the REAL policy's call?
                    a2 = a.copy()
                    a2[fs] = 0.0 if active else 1.0
                    Lr = np.asarray(out["policy_logits"], np.float64) + (a2 - a) @ Wc_policy
                    rec[f"{kind_flip}_changes_policy"] = bool(masked_argmax(Lr, lg) != bid)
                    rec[f"{kind_flip}_head_agrees_with_policy"] = co_bid == bid
        recs.append(rec)
        if j % 250 == 0:
            print(f"[cf] {model}: {j:,}/{len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
    return summarise(model, kind, pos, recs, rule_tab, reason_broken)


def summarise(model, kind, pos, recs, rule_tab, reason_broken) -> dict:
    deals = np.array([r["deal"] for r in recs])
    cov = np.array([r["n_cf"] > 0 for r in recs], float)
    cfs = [c for r in recs for c in r["cfs"]]
    cf_deal = np.array([r["deal"] for r in recs for _ in r["cfs"]])
    ad = np.array([abs(c["delta"]) for c in cfs])

    def share(key, vals=None):
        v = np.array([float(c[key]) for c in cfs]) if vals is None else vals
        e, lo, hi = cluster_bootstrap(v, cf_deal) if len(v) else (float("nan"),) * 3
        return {"value": e, "ci95": [lo, hi]}

    buckets = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        b = min(int(r["conf"] * 10), 9) / 10
        buckets[b][0] += 1
        buckets[b][1] += r["n_cf"] > 0
    cat = collections.Counter()
    for c in cfs:
        cat[("strength + shape" if c["d_hcp"] else "shape only") if c["cross_suit"]
            else ("strength only" if c["d_hcp"] else "neither")] += 1
    flip_cat = collections.Counter()
    for c in cfs:
        if c["flip"]:
            flip_cat[("strength + shape" if c["d_hcp"] else "shape only") if c["cross_suit"]
                     else ("strength only" if c["d_hcp"] else "neither")] += 1
    n_rel = rule_tab[(True, True)] + rule_tab[(True, False)]
    n_irr = rule_tab[(False, True)] + rule_tab[(False, False)]
    tot = n_rel + n_irr
    # Cohen's kappa between "description changed" and "call changed"
    po = (rule_tab[(True, True)] + rule_tab[(False, False)]) / max(tot, 1)
    pa, pb = n_rel / max(tot, 1), (rule_tab[(True, True)] + rule_tab[(False, True)]) / max(tot, 1)
    pe = pa * pb + (1 - pa) * (1 - pb)
    res = {
        "model": model, "kind": kind, "positions": pos.source, "n_decisions": len(recs),
        "coverage": dict(zip(("value", "lo", "hi"), cluster_bootstrap(cov, deals))),
        "n_counterfactuals": len(cfs), "cf_per_covered_decision": len(cfs) / max(cov.sum(), 1),
        "flip_rate": share("flip"),
        "abs_delta": {"mean": float(ad.mean()) if len(ad) else None, "median": float(np.median(ad)) if len(ad) else None,
                      "q25": float(np.percentile(ad, 25)) if len(ad) else None,
                      "q75": float(np.percentile(ad, 75)) if len(ad) else None},
        "coverage_by_confidence": {f"{b:.1f}": {"decisions": v[0], "with_cf": v[1], "coverage": v[1] / v[0]}
                                   for b, v in sorted(buckets.items())},
        "mean_abs_d_hcp": float(np.mean([abs(c["d_hcp"]) for c in cfs])) if cfs else None,
        "median_abs_d_hcp": float(np.median([abs(c["d_hcp"]) for c in cfs])) if cfs else None,
        "mean_shape_l1": float(np.mean([c["shape_l1"] for c in cfs])) if cfs else None,
        "longest_suit_changed": share("d_longest", np.array([float(c["d_longest"] != 0) for c in cfs])),
        "balance_flip": share("balance_flip"), "cross_suit": share("cross_suit"), "rank_up": share("rank_up"),
        "change_type": dict(cat), "change_type_of_flips": dict(flip_cat),
        "any_single_swap_flips": float(np.mean([r["any_flip"] for r in recs])),
        "mean_flipping_swaps_per_decision": float(np.mean([r["n_flipping_swaps"] for r in recs])),
        "rule_alignment": {
            "swaps_sampled": tot,
            "p_flip_given_description_changed": rule_tab[(True, True)] / max(n_rel, 1),
            "p_flip_given_description_unchanged": rule_tab[(False, True)] / max(n_irr, 1),
            "share_description_changed": n_rel / max(tot, 1),
            "share_of_flips_with_description_change": rule_tab[(True, True)] / max(rule_tab[(True, True)] + rule_tab[(False, True)], 1),
            "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"),
            "flips_breaking_a_reason_of_the_call": float(np.mean(reason_broken)) if reason_broken else None,
            "n_flipping_best_swaps": len(reason_broken),
        },
    }
    if "necessary_size" in recs[0]:
        cf = {}
        for kf in ("necessary", "sufficient"):
            sizes = [r[f"{kf}_size"] for r in recs]
            found = [s for s in sizes if s]
            cf[kf] = {"coverage": len(found) / len(sizes),
                      "size_distribution": {str(k): sum(1 for s in found if s == k) / len(sizes) for k in (1, 2, 3)},
                      "changes_real_policy": float(np.mean([r[f"{kf}_changes_policy"] for r in recs if r[f"{kf}_size"]]))
                      if found else None}
        res["concept_flips"] = cf
    return res, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2000, help="decisions sampled uniformly from the position set")
    ap.add_argument("--source", choices=("test", "selfplay"), default="test")
    ap.add_argument("--selfplay_deals", type=int, default=1000)
    ap.add_argument("--random_vulnerability", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    test = load_test_split() if args.source == "test" else None
    for model in args.models:
        pos = test if test is not None else selfplay(MODELS.get(model, model), args.selfplay_deals,
                                                     random_vulnerability=args.random_vulnerability)
        rows = sample_rows(pos, args.n, args.seed)
        res, recs = analyse(model, pos, rows, args.seed)
        tag = f"{model.replace(' ', '_').replace('+', 'p').replace('/', '_')}__{args.source}"  # a path model: no subdirectories
        (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
        with open(out / f"{tag}_records.pkl", "wb") as f:
            pickle.dump(recs, f)
        ra = res["rule_alignment"]
        print(f"[cf] {model}: coverage {res['coverage']['value']:.3f}, flip rate {res['flip_rate']['value']:.3f}, "
              f"mean |d| {res['abs_delta']['mean']:.3f}, any single swap flips {res['any_single_swap_flips']:.3f}; "
              f"P(flip|desc changed) {ra['p_flip_given_description_changed']:.3f} vs "
              f"P(flip|unchanged) {ra['p_flip_given_description_unchanged']:.3f}", flush=True)


if __name__ == "__main__":
    main()
