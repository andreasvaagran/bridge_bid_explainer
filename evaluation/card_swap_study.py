#!/usr/bin/env python3
"""Are card-swap statistics useful for explaining a bid? Per held-out decision, the 507 one-card
changes: how decisive single cards are, which cards matter for which calls, agreement with IG, and whether the
explanation's reasons and text follow the swaps."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np

DEFAULT_MODELS = ["SL independent", "SL joint", "SL sequential", "PPO joint", "brl SL", "brl RL+FSP"]
RANK_CLASS = {"A": "A", "K": "K", "Q": "Q", "J": "J", "T": "10"}
KEY_SHARE = 0.5


def _family(cat: str, bid: str) -> str:
    """A coarse family of calls for the per-family tables."""
    if bid == "P":
        return "pass"
    if bid in ("X", "XX"):
        return "double / redouble"
    if cat == "opening_1nt" or cat == "opening_2nt":
        return "notrump opening"
    if cat.startswith("opening_"):
        return "suit opening"
    if cat.startswith("overcall"):
        return "overcall"
    if "raise" in cat or "support" in cat:
        return "raise of partner"
    if bid.endswith("NT"):
        return "notrump bid"
    if cat.startswith("slam"):
        return "slam bidding"
    return "other bid"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


_HOLD_RE = re.compile(r"\b(?:\d+|a singleton|a void in) (spades?|hearts?|diamonds?|clubs?) \(([AKQJT2-9 ]+)\)"
                      r"|\bthe ((?:[AKQJT2-9] )*[AKQJT2-9]) of (spades|hearts|diamonds|clubs)\b"
                      r"|\b(spades|hearts|diamonds|clubs) \(([AKQJT2-9 ]+)\)"
                      r"|\b(ace|king|queen|jack) of (spades|hearts|diamonds|clubs)\b")


def cited_cards(text: str) -> set:
    """The cards an explanation names (inside a cited holding such as "5 clubs (A Q 8 7 2)")."""
    out = set()
    for m in _HOLD_RE.finditer(text):
        g = m.groups()
        if g[0]:
            suit, cards = g[0], g[1]
        elif g[2]:
            suit, cards = g[3], g[2]
        elif g[4]:
            suit, cards = g[4], g[5]
        else:
            suit, cards = g[7], {"ace": "A", "king": "K", "queen": "Q", "jack": "J"}[g[6]]
        s = suit[0].upper()
        out |= {s + c for c in cards.split()}
    return out


def _role(suit: str, bid: str, ours: set, partners: set, theirs: set) -> str:
    if bid[:1].isdigit() and bid[-1] == suit:
        return "suit of the call"
    if suit in partners:
        return "partner's suit"
    if suit in theirs:
        return "opponents' suit"
    if suit in ours:
        return "own earlier suit"
    return "side suit"


def analyse_model(args) -> dict:
    model, rows, seed = args
    import jax  # noqa: F401  (JAX_PLATFORMS is set by the parent)
    from concepts import gloss_for_concept
    from evaluation.common import MODELS, SEATS, load_test_split
    from explain.card_swap_stats import card_swap_stats
    from nlg.explanation_frames import BIDS, decision_frame
    from nlg.explanation_inputs import _suit_of, bid_category, natural_partner_suits, select_reasoned_concepts
    from nlg.teacher import explain
    from nlg.decision_record import Bidder, _upstream_frame

    pos = load_test_split()
    bidder = Bidder(str(MODELS.get(model, model)))
    concept = bidder.kind == "concept"
    names = list(bidder.ck.hard_concepts) if concept else None
    per = []
    by_family = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))   # family -> (rank|role) -> [changed, total]
    t0 = time.time()
    for j, i in enumerate(rows):
        obs = pos.obs[i].astype(np.float32)
        legal = pos.legal[i]
        out = bidder.outputs(obs)
        z = np.where(legal, out["policy_logits"], -np.inf)
        action = int(np.argmax(z))
        bid = BIDS[action]
        before, seat, dl = pos.auction(i)
        try:
            if concept:
                frame = decision_frame(bidder.enricher, obs, legal, out, int(pos.deal[i]), int(pos.step[i]), seat,
                                       SEATS[dl], dl, "None", before, counterfactuals=False)
            else:
                frame = _upstream_frame(bidder, obs, legal, out, int(pos.deal[i]), int(pos.step[i]), seat, SEATS[dl],
                                        dl, "None", before, counterfactuals=False)
            st = card_swap_stats(obs, legal, action, bidder.outputs_batch, BIDS, concept_names=names,
                                 glosses=gloss_for_concept)
        except Exception as e:  # a rare auction the decoders cannot read
            print(f"[{model}] position {i}: {type(e).__name__}: {e}", flush=True)
            continue
        cat = bid_category(frame)
        fam = _family(cat, bid)
        text = explain(frame)
        reasons = select_reasoned_concepts(frame) if concept else []
        # card-level evidence: swap importance against Integrated Gradients
        attr = {a["card"]: abs(a["value"]) for a in bidder.card_attribution(obs, action)}
        cards = st["cards"]
        imp = np.array([abs(c["mean_dp"]) for c in cards])
        ig = np.array([attr.get(c["card"], 0.0) for c in cards])
        rho = _spearman(imp, ig)
        key = [c["card"] for c in cards if c["share"] >= KEY_SHARE]
        named = cited_cards(text)
        # the explanation's reasons on the swapped hands (concept models)
        tie = None
        if concept and reasons:
            from explain.counterfactual import all_single_swaps
            from concepts.hand import decode_hand
            batch, pairs = all_single_swaps(obs, decode_hand(obs))
            ob = bidder.outputs_batch(batch)
            pr = ob["policy_logits"]
            new = np.argmax(np.where(legal[None, :], pr, -np.inf), axis=1)
            changed = new != action
            idx = [names.index(r) for r in reasons]
            a0 = np.asarray(out["concept_acts"])[idx] >= 0.5
            a1 = np.asarray(ob["concept_acts"])[:, idx] >= 0.5
            flip = (a1 != a0[None, :]).any(axis=1)
            tie = {"changed": int(changed.sum()), "flip_when_changed": int((flip & changed).sum()),
                   "kept": int((~changed).sum()), "flip_when_kept": int((flip & ~changed).sum())}
        # per-card change rates by rank class and suit role
        prior = frame.get("auction_before") or []
        t = frame.get("table") or {}
        ours = {_suit_of(b["bid"]) for b in prior if b["seat"] == t.get("you") and _suit_of(b["bid"])}
        partners = set(natural_partner_suits(frame))
        theirs = {_suit_of(b["bid"]) for b in prior if b["seat"] not in (t.get("you"), t.get("partner")) and _suit_of(b["bid"])}
        for c in cards:
            rc = RANK_CLASS.get(c["card"][1], "small")
            role = _role(c["card"][0], bid, ours, partners, theirs)
            for k in (f"rank:{rc}", f"role:{role}"):
                by_family[fam][k][0] += c["changed"]
                by_family[fam][k][1] += c["of"]
        per.append({"i": int(i), "bid": bid, "family": fam, "kept_share": st["kept_share"], "n_key": len(key),
                    "max_share": max(c["share"] for c in cards), "rho_ig": rho,
                    "key_named": [k in named for k in key], "tie": tie, "n_calls": len(st["calls"])})
        if j % 200 == 0:
            print(f"[{model}] {j}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    return {"model": model, "kind": bidder.kind, "per": per,
            "by_family": {f: {k: v for k, v in d.items()} for f, d in by_family.items()}}


def summarise(res: dict) -> dict:
    per = res["per"]
    n = len(per)
    kept = np.array([p["kept_share"] for p in per])
    rho = np.array([p["rho_ig"] for p in per if not np.isnan(p["rho_ig"])])
    key_any = np.mean([p["n_key"] > 0 for p in per])
    named = [x for p in per for x in p["key_named"]]
    out = {"model": res["model"], "kind": res["kind"], "decisions": n,
           "kept_share_median": float(np.median(kept)), "kept_share_mean": float(kept.mean()),
           "all_507_keep_the_call": float(np.mean(kept == 1.0)),
           "decisions_with_a_key_card": float(key_any), "key_cards_mean": float(np.mean([p["n_key"] for p in per])),
           "rho_swap_vs_ig_median": float(np.median(rho)) if len(rho) else None, "rho_n": int(len(rho)),
           "key_cards_named_by_the_explanation": float(np.mean(named)) if named else None, "key_cards_n": len(named)}
    ties = [p["tie"] for p in per if p["tie"]]
    if ties:
        ch = sum(t["changed"] for t in ties)
        kp = sum(t["kept"] for t in ties)
        out["reason_flips_when_call_changes"] = sum(t["flip_when_changed"] for t in ties) / max(ch, 1)
        out["reason_flips_when_call_kept"] = sum(t["flip_when_kept"] for t in ties) / max(kp, 1)
        out["tie_decisions"] = len(ties)
        out["call_changing_swaps"] = ch
    fam = collections.Counter(p["family"] for p in per)
    out["families"] = {}
    for f, cnt in fam.most_common():
        sub = [p for p in per if p["family"] == f]
        d = res["by_family"].get(f, {})
        rate = {k: round(v[0] / v[1], 3) for k, v in sorted(d.items()) if v[1] >= 200}
        out["families"][f] = {"n": cnt, "kept_share_mean": round(float(np.mean([p["kept_share"] for p in sub])), 3),
                              "with_key_card": round(float(np.mean([p["n_key"] > 0 for p in sub])), 3),
                              "change_rate": rate}
    return out


def markdown(S: list, meta: dict) -> str:
    L = [f"# Card-swap statistics: are they useful for explaining a bid? ({meta['date']})", "",
         f"{meta['n']} decisions sampled from the WBridge5 held-out test split (seed {meta['seed']}); each model's own "
         "call re-scored on all 507 one-card changes of the hand (`evaluation/card_swap_study.py`, "
         "`explain/card_swap_stats.py`). A card is *key* when replacing it by any other card changes the call in at least "
         "half of its 39 replacements.", "",
         "| Model | changes that keep the call (median) | decisions where all 507 keep it | decisions with a key card | "
         "swap importance vs IG (median Spearman) | reason concept flips: call changes / call kept | key cards the text names |",
         "|---|---|---|---|---|---|---|"]
    for s in S:
        tie = (f"{s['reason_flips_when_call_changes']:.0%} / {s['reason_flips_when_call_kept']:.0%}"
               if "reason_flips_when_call_changes" in s else "-")
        L.append(f"| {s['model']} | {s['kept_share_median']:.0%} | {s['all_507_keep_the_call']:.0%} | "
                 f"{s['decisions_with_a_key_card']:.0%} | {s['rho_swap_vs_ig_median']:+.2f} | {tie} | "
                 f"{s['key_cards_named_by_the_explanation']:.0%} of {s['key_cards_n']} |"
                 if s['key_cards_named_by_the_explanation'] is not None else
                 f"| {s['model']} | {s['kept_share_median']:.0%} | {s['all_507_keep_the_call']:.0%} | "
                 f"{s['decisions_with_a_key_card']:.0%} | {s['rho_swap_vs_ig_median']:+.2f} | {tie} | - |")
    L += ["", "## Which cards matter for which calls", "",
          "Share of replacements of a card that change the call, by the card's rank and by its suit's role, per "
          f"family of calls ({S[0]['model']}; cells with at least 200 replacements).", ""]
    s0 = S[0]
    cols = ["rank:A", "rank:K", "rank:Q", "rank:J", "rank:10", "rank:small", "role:suit of the call", "role:partner's suit",
            "role:opponents' suit", "role:own earlier suit", "role:side suit"]
    L.append("| Family (n) | " + " | ".join(c.split(":")[1] for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for f, d in s0["families"].items():
        L.append(f"| {f} ({d['n']}) | " + " | ".join(f"{d['change_rate'][c]:.0%}" if c in d["change_rate"] else "" for c in cols) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    from evaluation.common import load_test_split, sample_rows
    pos = load_test_split()
    rows = [int(r) for r in sample_rows(pos, args.n, seed=args.seed)]
    with mp.get_context("spawn").Pool(args.workers or len(args.models)) as pool:
        results = pool.map(analyse_model, [(m, rows, args.seed) for m in args.models])
    S = [summarise(r) for r in results]
    meta = {"n": len(rows), "seed": args.seed, "date": time.strftime("%Y-%m-%d"), "split": pos.source}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "summary": S,
                                          "per_decision": {r["model"]: r["per"] for r in results}}, indent=1))
    md = markdown(S, meta)
    if args.md:
        Path(args.md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
