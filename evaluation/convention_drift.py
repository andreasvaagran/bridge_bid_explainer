#!/usr/bin/env python3
"""Which system does each agent play, and how far is it from WBridge5's? Measured on its own self-play:
calls that break their situation's requirements (raw and standardized to WBridge5's mix, small vs large breaks),
HCP and length per situation, how often partner leaves a double in, and the opening distribution's JS divergence."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from concepts.hand import decode_hand
from evaluation.common import (BIDS, MODELS, SEATS, checkpoint_stamp, cluster_bootstrap, load_test_split, same_checkpoint,
                                  selfplay, wilson)
from nlg.double_meaning import shape_key
from nlg.explanation_frames import hand_str_letter_coded
from nlg.explanation_inputs import bid_category, deviation_notes

AGENTS = ["brl SL", "brl RL", "brl RL+FSP", "brl RL+PFSP", "SL joint", "SL independent", "SL sequential",
          "PPO joint", "PPO independent", "PPO sequential"]
PROFILE = ["opening_1nt", "opening_1_major", "opening_1_minor", "opening_weak_two", "opening_2c_strong",
           "opening_preempt", "opening_pass", "overcall_suit", "takeout_double", "response_1nt_stayman",
           "response_1nt_transfer", "response_simple_raise", "response_new_suit_1_level"]


def decisions(pos):
    """Per decision: a light frame (enough for bid_category / deviation_notes) and hand features."""
    for i in range(len(pos)):
        before, seat, dl = pos.auction(i)
        h = decode_hand(pos.obs[i])
        frame = {"table": {"you": seat, "partner": SEATS[(SEATS.index(seat) + 2) % 4],
                           "opponents": [SEATS[(SEATS.index(seat) + 1) % 4], SEATS[(SEATS.index(seat) + 3) % 4]]},
                 "auction_before": before, "bid": BIDS[int(pos.action[i])], "hand_str": hand_str_letter_coded(h),
                 "dealer": SEATS[dl], "vulnerability": pos.vul[i], "hcp": h.hcp}
        yield i, frame, h


def full_auctions(pos):
    """deal -> list of calls in order (every decision of the deal is in `pos`)."""
    auc = collections.defaultdict(list)
    for i in np.lexsort((pos.step, pos.deal)):
        auc[int(pos.deal[i])].append(BIDS[int(pos.action[i])])
    return auc


def leave_in(auctions) -> dict:
    """shape -> [doubles followed by an opponent's pass, of which partner passed]."""
    stats = collections.defaultdict(lambda: [0, 0])
    for calls in auctions.values():
        for t, c in enumerate(calls):
            if c != "X" or t + 2 >= len(calls) or calls[t + 1] != "P":
                continue
            who = lambda j: "m" if (t - j) % 4 == 0 else "p" if (t - j) % 4 == 2 else "o"  # noqa: E731
            key = shape_key((who(j), calls[j]) for j in range(t))
            stats[key][0] += 1
            stats[key][1] += calls[t + 2] == "P"
    return stats


def jsd_bits(p: dict, q: dict) -> float:
    keys = sorted(set(p) | set(q))
    P = np.array([p.get(k, 0) for k in keys], float); P /= P.sum()
    Q = np.array([q.get(k, 0) for k in keys], float); Q /= Q.sum()
    M = (P + Q) / 2
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0)))  # noqa: E731
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


_SPOTS = "23456789T"
_QJ_SWAPS = [("Q", "J"), ("J", "Q"), ("Q", "spot"), ("spot", "Q"), ("J", "spot"), ("spot", "J")]  # |dHCP| <= 2


def near_miss(frame) -> bool:
    """A breaking call that keeps the system after one card of the hand is replaced by another of the same suit
    among queens, jacks and spot cards: at most 2 HCP more or fewer, the same suit lengths, aces and kings."""
    cards = {}
    for part in frame["hand_str"].split():
        suit, ranks = part.split(":")
        cards[suit] = [] if ranks == "-" else list(ranks)
    order = "23456789TJQKA"
    for s, held in cards.items():
        spots_held = [r for r in held if r in _SPOTS]
        spot_free = next((r for r in _SPOTS if r not in held), None)
        for old, new in _QJ_SWAPS:
            out = spots_held[-1:] if old == "spot" else [old] if old in held else []  # any spot: HCP is the same
            put = spot_free if new == "spot" else new if new not in held else None
            if not out or put is None:
                continue
            ranks = sorted([r for r in held if r != out[0]] + [put], key=order.index, reverse=True)
            hand = " ".join(f"{t}:{''.join(ranks) if t == s else (''.join(cards[t]) or '-')}" for t in "SHDC")
            if not deviation_notes({**frame, "hand_str": hand}, explanation_notes=False):
                return True
    return False


def standardized(cats, dev, deal_of, ref: dict, n_boot: int = 1000, seed: int = 0):
    """The deviation rate with WBridge5's mix of situations (ref: situation -> share of its non-pass calls), with a
    percentile CI resampling whole deals as cluster_bootstrap does: (estimate, lo, hi, coverage)."""
    sits = sorted(ref)
    col = {c: j for j, c in enumerate(sits)}
    keep = np.array([c in col for c in cats], bool)
    uniq, inv = np.unique(np.asarray(deal_of)[keep], return_inverse=True)
    j = np.array([col[c] for c in np.asarray(cats)[keep]], int)
    C = np.zeros((len(uniq), len(sits))); D = np.zeros((len(uniq), len(sits)))
    np.add.at(C, (inv, j), 1); np.add.at(D, (inv, j), np.asarray(dev, float)[keep])
    w = np.array([ref[c] for c in sits])

    def rate(c, d):
        m = c > 0
        return float((w[m] * d[m] / c[m]).sum() / w[m].sum()), float(w[m].sum())
    est, cov = rate(C.sum(0), D.sum(0))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        boots.append(rate(k @ C, k @ D)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), cov


def analyse(name, pos, ref: dict | None = None) -> dict:
    t0 = time.time()
    dev, large, deal_of, cats = [], [], [], []
    cat_dev = collections.defaultdict(lambda: [0, 0, 0])  # calls, breaking, breaking by more than a near miss
    profiles = collections.defaultdict(lambda: {"hcp": [], "len": [], "balanced": []})
    openings = collections.Counter()
    examples = collections.defaultdict(list)
    for i, frame, h in decisions(pos):
        cat = bid_category(frame)
        bid = frame["bid"]
        notes = deviation_notes(frame, explanation_notes=False)
        if bid != "P":
            big = bool(notes) and not near_miss(frame)
            dev.append(bool(notes)); large.append(big); deal_of.append(int(pos.deal[i])); cats.append(cat)
            cat_dev[cat][0] += 1
            cat_dev[cat][1] += bool(notes)
            cat_dev[cat][2] += big
            if notes and len(examples[cat]) < 3:
                examples[cat].append({"auction": " ".join(b["bid"] for b in frame["auction_before"]) or "-",
                                      "bid": bid, "hand": frame["hand_str"], "note": notes[0]})
        if cat in PROFILE:
            pr = profiles[cat]
            pr["hcp"].append(h.hcp)
            s = bid[-1] if bid[-1] in "SHDC" else None
            pr["len"].append(h.lengths[s] if s else -1)
            pr["balanced"].append(h.balanced)
        if cat.startswith("opening_"):
            openings[bid] += 1
    e, lo, hi = cluster_bootstrap(np.array(dev, float), np.array(deal_of)) if dev else (float("nan"),) * 3
    prof = {}
    for cat, pr in profiles.items():
        hc = np.array(pr["hcp"]); ln = np.array(pr["len"])
        prof[cat] = {"n": len(hc), "hcp_p10_p50_p90": [float(x) for x in np.percentile(hc, [10, 50, 90])],
                     "balanced": float(np.mean(pr["balanced"])),
                     "suit_len_ge5": float(np.mean(ln[ln >= 0] >= 5)) if (ln >= 0).any() else None,
                     "suit_len_ge6": float(np.mean(ln[ln >= 0] >= 6)) if (ln >= 0).any() else None,
                     "suit_len_median": float(np.median(ln[ln >= 0])) if (ln >= 0).any() else None}
    calls_by_sit = {c: v[0] for c, v in sorted(cat_dev.items())}
    if ref is None:  # WBridge5 itself: its own mix
        ref = {c: n / len(dev) for c, n in calls_by_sit.items()}
    se, slo, shi, cov = standardized(cats, dev, deal_of, ref)
    le, llo, lhi = cluster_bootstrap(np.array(large, float), np.array(deal_of)) if dev else (float("nan"),) * 3
    lse, lslo, lshi, _ = standardized(cats, large, deal_of, ref)
    li = leave_in(full_auctions(pos))
    res = {"agent": name, "positions": pos.source, "decisions": len(pos), "non_pass_calls": len(dev),
           "deviation_rate_non_pass": {"value": e, "ci95": [lo, hi]},
           "deviation_rate_standardized": {"value": se, "ci95": [slo, shi], "coverage": cov,
                                           "weights": "WBridge5's non-pass calls per situation (test split)"},
           "large_deviation_rate_non_pass": {"value": le, "ci95": [llo, lhi]},
           "large_deviation_rate_standardized": {"value": lse, "ci95": [lslo, lshi]},
           "calls_by_situation": calls_by_sit,
           "deviation_by_situation": {c: {"calls": v[0], "deviating": v[1], "large": v[2], "rate": wilson(v[1], v[0])}
                                      for c, v in sorted(cat_dev.items()) if v[1] > 0 or v[0] >= 50},
           "examples": dict(examples), "profiles": prof, "opening_calls": dict(openings),
           "leave_in": {k: v for k, v in li.items() if v[0] >= 20},
           "calls_per_deal": len(pos) / len(np.unique(pos.deal)),
           "passed_out": float(np.mean([all(c == "P" for c in a) for a in full_auctions(pos).values()]))}
    print(f"[drift] {name}: {len(pos):,} decisions, deviation {e:.2%} [{lo:.2%}, {hi:.2%}] of non-pass calls "
          f"; standardized to WBridge5's mix {se:.2%} [{slo:.2%}, {shi:.2%}], large breaks {lse:.2%} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--agents", nargs="+", default=AGENTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "drift.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if "large_deviation_rate_standardized" not in res.get("WBridge5", {}):
        test = load_test_split()
        res["WBridge5"] = analyse("WBridge5 (test deals)", test)
        path.write_text(json.dumps(res, indent=1))
    w5 = res["WBridge5"]["calls_by_situation"]
    ref = {c: n / sum(w5.values()) for c, n in w5.items()}
    for a in args.agents:
        # recomputed when the checkpoint changed or the entry lacks the large-break rate
        if a in res and same_checkpoint(res[a], MODELS[a]) and "large_deviation_rate_standardized" in res[a]:
            continue
        keep = {k: res[a][k] for k in ("wbridge5_agreement",) if a in res and k in res[a]
                and same_checkpoint(res[a], MODELS[a])}  # imitation does not change with the measure
        res[a] = analyse(a, selfplay(MODELS[a], args.deals, seed=args.seed), ref) | keep
        res[a].update(checkpoint_stamp(MODELS[a]))
        path.write_text(json.dumps(res, indent=1))
    # imitation accuracy on WBridge5's test split for every agent here, not only the ones strength_matches plays
    todo = [a for a in res if a != "WBridge5" and "wbridge5_agreement" not in res[a]]
    if todo:
        from evaluation.strength_matches import imitation
        acc, _ = imitation(todo, load_test_split())
        for a in todo:
            res[a]["wbridge5_agreement"] = acc[a]
    ref = res["WBridge5"]["opening_calls"]
    for a in res:
        res[a]["opening_jsd_bits_vs_wbridge5"] = jsd_bits(res[a]["opening_calls"], ref)
    path.write_text(json.dumps(res, indent=1))
    for a, r in res.items():
        print(f"{a:18s} deviation {r['deviation_rate_non_pass']['value']:.2%}  "
              f"standardized {r.get('deviation_rate_standardized', {}).get('value', float('nan')):.2%}  opening JSD {r['opening_jsd_bits_vs_wbridge5']:.4f} bits")


if __name__ == "__main__":
    main()
