#!/usr/bin/env python3
"""Policy-relative meanings. A card from play maps each (situation, call) to the agent's HCP range and
minimum suit length, fitted on even deals and tested on odd ones against WBridge5's card. --save_cards writes brl's
cards to nlg/convention_cards, and `own_convention` reads them at explanation time."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from evaluation.convention_drift import decisions
from evaluation.common import MODELS, load_test_split, selfplay
from nlg.explanation_inputs import bid_category, deviation_notes, hand_from_str

AGENTS = ["brl SL", "brl RL", "brl RL+FSP", "brl RL+PFSP", "SL independent", "PPO independent"]
MIN_N = 20          # entries fitted on fewer calls are not part of the card
SHOW = [("opening_1_major", "1S"), ("opening_1_major", "1H"), ("opening_1_minor", "1D"), ("opening_1nt", "1NT"),
        ("opening_weak_two", "2S"), ("overcall_suit", "1S"), ("response_new_suit_1_level", "1S")]
CARD_DIR = Path(__file__).resolve().parents[1] / "nlg/convention_cards"   # --save_cards, read by own_convention
SUIT_NAME = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}


def calls(pos):
    """[(deal, key, hcp, bid-suit length or None, deviates)] for every non-pass call."""
    out = []
    for i, frame, h in decisions(pos):
        bid = frame["bid"]
        if bid == "P":
            continue
        suit = bid[-1] if bid[-1] in "SHDC" and bid[0].isdigit() else None
        out.append((int(pos.deal[i]), (bid_category(frame), bid), h.hcp, h.lengths[suit] if suit else None,
                    bool(deviation_notes(frame, explanation_notes=False))))
    return out


def fit(rows) -> dict:
    by = collections.defaultdict(lambda: ([], []))
    for _, key, hcp, ln, _ in rows:
        by[key][0].append(hcp)
        if ln is not None:
            by[key][1].append(ln)
    card = {}
    for key, (hc, ln) in by.items():
        if len(hc) < MIN_N:
            continue
        lo, hi = np.percentile(hc, [2.5, 97.5])
        card[key] = {"n": len(hc), "hcp": [int(np.floor(lo)), int(np.ceil(hi))],
                     "min_len": int(np.floor(np.percentile(ln, 2.5))) if ln else None}
    return card


def describe(card, rows) -> dict:
    described, width, missing = [], [], 0
    for _, key, hcp, ln, _ in rows:
        e = card.get(key)
        if e is None:
            missing += 1
            described.append(False)
            continue
        ok = e["hcp"][0] <= hcp <= e["hcp"][1] and (e["min_len"] is None or ln is None or ln >= e["min_len"])
        described.append(ok)
        if ok:
            width.append(e["hcp"][1] - e["hcp"][0])
    return {"described": float(np.mean(described)), "no_entry": missing / max(len(rows), 1),
            "mean_hcp_width": float(np.mean(width)) if width else None, "calls": len(rows)}


# ------------------------------------------------- explanation time: a call in the agent's own convention
@lru_cache(maxsize=None)
def card_for(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    p = CARD_DIR / f"{Path(checkpoint_path).stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def own_convention(checkpoint_path: str, frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """{"text", "fits", "hcp", "min_len", "n"} for this agent's call in this situation, or None."""
    card = card_for(checkpoint_path)
    if not card:
        return None
    bid = frame.get("bid", "")
    situation = bid_category(frame)
    e = card["entries"].get(f"{situation}|{bid}")
    if not e:
        return None
    h = hand_from_str(frame["hand_str"])
    suit = bid[-1] if bid[:1].isdigit() and bid[-1] in SUIT_NAME else None
    length = f" and {e['min_len']}+ {SUIT_NAME[suit]}" if suit and e.get("min_len") else ""
    fits = e["hcp"][0] <= h.hcp <= e["hcp"][1] and (not suit or not e.get("min_len") or h.lengths[suit] >= e["min_len"])
    text = (f"In this agent's own auctions, {bid} in this situation ({situation.replace('_', ' ')}) is made with "
            f"{e['hcp'][0]}-{e['hcp'][1]} HCP{length} ({e['n']} calls); this hand "
            f"{'fits' if fits else 'is outside'} that range.")
    return {"text": text, "fits": fits, "hcp": e["hcp"], "min_len": e.get("min_len"), "n": e["n"],
            "agent": card.get("agent")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=6000)
    ap.add_argument("--agents", nargs="+", default=AGENTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_cards", default=None,
                    help="directory for each agent's full card (fitted on all its deals), read by own_convention")
    args = ap.parse_args()
    t0 = time.time()
    data = {"WBridge5": calls(load_test_split())}
    print(f"[cards] WBridge5: {len(data['WBridge5']):,} non-pass calls ({time.time() - t0:.0f}s)", flush=True)
    for a in args.agents:
        data[a] = calls(selfplay(MODELS[a], args.deals, seed=1, random_vulnerability=False))
        print(f"[cards] {a}: {len(data[a]):,} non-pass calls ({time.time() - t0:.0f}s)", flush=True)
    split = lambda rows, r: [x for x in rows if x[0] % 2 == r]  # noqa: E731
    cards = {a: fit(split(rows, 0)) for a, rows in data.items()}
    res = {"deals_per_agent": args.deals, "min_calls_per_entry": MIN_N, "agents": {}, "examples": {}}
    for a, rows in data.items():
        test = split(rows, 1)
        # common support: calls whose (situation, call) has an entry in both cards, so that the
        # comparison is about the ranges, not about which card was fitted on more deals
        both = [x for x in test if x[1] in cards[a] and x[1] in cards["WBridge5"]]
        res["agents"][a] = {"own_card": describe(cards[a], test), "wbridge5_card": describe(cards["WBridge5"], test),
                            "common_support": {"calls": len(both), "share_of_calls": len(both) / max(len(test), 1),
                                               "own_card": describe(cards[a], both)["described"],
                                               "wbridge5_card": describe(cards["WBridge5"], both)["described"],
                                               "rules_no_deviation": float(np.mean([not x[4] for x in both]))},
                            "rules_no_deviation": float(np.mean([not x[4] for x in test])),
                            "card_entries": len(cards[a])}
        r = res["agents"][a]
        c = r["common_support"]
        print(f"[cards] {a:16s} own card describes {r['own_card']['described']:.3f} (width "
              f"{r['own_card']['mean_hcp_width']:.1f}), WBridge5's card {r['wbridge5_card']['described']:.3f}; "
              f"rules {r['rules_no_deviation']:.3f} | common support ({c['share_of_calls']:.0%} of calls): own "
              f"{c['own_card']:.3f}, WBridge5's {c['wbridge5_card']:.3f}, rules {c['rules_no_deviation']:.3f}", flush=True)
    for key in SHOW:
        res["examples"][f"{key[0]} {key[1]}"] = {a: cards[a].get(key) for a in cards}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    if args.save_cards:
        d = Path(args.save_cards)
        d.mkdir(parents=True, exist_ok=True)
        for a, rows in data.items():
            if not a.startswith("brl"):  # concept checkpoints share file names (params-best.pkl) and
                continue                  # stay in WBridge5's system; the rules describe them already
            card = fit(rows)
            (d / f"{Path(str(MODELS[a])).stem}.json").write_text(json.dumps({
                "agent": a, "checkpoint": Path(str(MODELS[a])).name, "deals": args.deals,
                "built_by": "python -m evaluation.convention_ranges --save_cards " + str(d),
                "entries": {f"{k[0]}|{k[1]}": v for k, v in sorted(card.items())}}, indent=0))
        print(f"[cards] saved the brl networks' cards -> {d}")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
