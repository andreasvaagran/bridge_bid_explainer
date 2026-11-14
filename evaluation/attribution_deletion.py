#!/usr/bin/env python3
"""Is the Integrated Gradients card ranking faithful? Incremental deletion (AOPC; Nauta et al. 2023):
cards are taken out in IG, |IG|, HCP or random order (replaced or zeroed), and the drop of the call's probability is
measured, with key-card agreement as a second check."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

DEFAULT_MODELS = ["SL independent", "SL joint", "SL sequential", "PPO joint", "brl SL", "brl RL+FSP"]
ORDERINGS = ["IG", "|IG|", "HCP", "random", "swaps"]
KEY_SHARE = 0.5          # as in evaluation/card_swap_study.py
HONOUR = {"A": 4, "K": 3, "Q": 2, "J": 1}


def _probs(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    z = np.where(legal[None, :], logits.astype(np.float64), -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def analyse_model(args) -> dict:
    model, rows, seed, K, R = args
    import jax  # noqa: F401  (JAX_PLATFORMS is set by the parent)
    from concepts.hand import RANKS, card_bit_index, decode_hand, replace_card
    from evaluation.common import MODELS, load_test_split
    from explain.card_swap_stats import card_swap_stats
    from explain.counterfactual import FULL_DECK
    from nlg.explanation_frames import BIDS
    from nlg.decision_record import Bidder

    pos = load_test_split()
    bidder = Bidder(str(MODELS.get(model, model)))
    rng = np.random.default_rng(seed)
    per = []
    t0 = time.time()
    for j, i in enumerate(rows):
        obs = pos.obs[i].astype(np.float32)
        legal = pos.legal[i]
        logits0 = bidder.outputs(obs)["policy_logits"]
        action = int(np.argmax(np.where(legal, logits0, -np.inf)))
        p0 = _probs(logits0[None], legal)[0, action]
        hand = decode_hand(obs)
        held = [s + r for s, rs in hand.cards.items() for r in rs]
        if len(held) != 13:
            continue
        st = card_swap_stats(obs, legal, action, bidder.outputs_batch, BIDS)
        by_card = {c["card"]: c for c in st["cards"]}
        ig = {a["card"]: a["value"] for a in bidder.card_attribution(obs, action)}
        ig_v = np.array([ig.get(c, 0.0) for c in held])
        swap_v = np.array([-by_card[c]["mean_dp"] for c in held])
        key = np.array([by_card[c]["share"] >= KEY_SHARE for c in held])
        tie = rng.random(13)
        hcp_v = np.array([HONOUR.get(c[1], 0) + RANKS.index(c[1]) / 100 for c in held]) + tie * 1e-6
        # orderings (index lists into `held`); stable sort with a random tie-break
        order = {"IG": np.lexsort((tie, -ig_v)), "|IG|": np.lexsort((tie, -np.abs(ig_v))),
                 "HCP": np.argsort(-hcp_v, kind="stable"), "swaps": np.lexsort((tie, -swap_v))}
        rand_orders = [rng.permutation(13) for _ in range(R)]
        free = [c for c in FULL_DECK if c not in set(held)]
        repl = [rng.choice(len(free), size=K, replace=False) for _ in range(R)]   # draw r: replacements for k = 1..K

        batch, tags = [], []          # tags: (perturbation, ordering, k, draw)
        for name in ORDERINGS:
            for r in range(R):
                o = rand_orders[r] if name == "random" else order[name]
                cur = obs
                for k in range(K):
                    cur = replace_card(cur, held[o[k]], free[repl[r][k]])
                    batch.append(cur); tags.append(("replace", name, k, r))
            draws = range(R) if name == "random" else range(1)
            for r in draws:
                o = rand_orders[r] if name == "random" else order[name]
                cur = obs.copy()
                for k in range(K):
                    cur = cur.copy(); cur[-52 + card_bit_index(held[o[k]])] = 0.0
                    batch.append(cur); tags.append(("remove", name, k, r))
        p = _probs(bidder.policy_logits_batch(np.stack(batch)), legal)
        drop = p0 - p[:, action]
        changed = p.argmax(1) != action
        rec = {"i": int(i), "deal": int(pos.deal[i]), "bid": BIDS[action], "p0": float(p0)}
        for pert in ("replace", "remove"):
            for name in ORDERINGS:
                sel = [t for t, tg in enumerate(tags) if tg[0] == pert and tg[1] == name]
                d = drop[sel].reshape(-1, K).mean(0)
                c = changed[sel].reshape(-1, K).mean(0)
                rec[f"{pert}:{name}:drop"] = [round(float(x), 5) for x in d]
                rec[f"{pert}:{name}:changed"] = [round(float(x), 4) for x in c]
        rec["n_key"] = int(key.sum())
        if key.any() and not key.all():
            rec["top_ig_is_key"] = bool(key[order["|IG|"][0]])
            rec["auroc_absig_key"] = _auroc(np.abs(ig_v), key)
            rec["auroc_ig_key"] = _auroc(ig_v, key)
            rec["auroc_hcp_key"] = _auroc(hcp_v, key)
        per.append(rec)
        if j % 250 == 0:
            print(f"[{model}] {j}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    return {"model": model, "kind": bidder.kind, "per": per}


def summarise(res: dict, K: int) -> dict:
    from evaluation.common import cluster_bootstrap
    per = res["per"]
    deal = np.array([p["deal"] for p in per])
    out = {"model": res["model"], "kind": res["kind"], "decisions": len(per)}

    def ci(v):
        e, lo, hi = cluster_bootstrap(np.asarray(v, float), deal)
        return [round(e, 4), round(lo, 4), round(hi, 4)]
    for pert in ("replace", "remove"):
        d = {}
        for name in ORDERINGS:
            drops = np.array([p[f"{pert}:{name}:drop"] for p in per])
            ch = np.array([p[f"{pert}:{name}:changed"] for p in per])
            d[name] = {"aopc": ci(drops.mean(1)), "drop_by_k": [round(float(x), 4) for x in drops.mean(0)],
                       "changed_by_k": [round(float(x), 4) for x in ch.mean(0)],
                       "changed_at_1": ci(ch[:, 0]), f"changed_at_{K}": ci(ch[:, -1])}
        for name in ("IG", "|IG|", "HCP"):
            diff = (np.array([p[f"{pert}:{name}:drop"] for p in per]).mean(1)
                    - np.array([p[f"{pert}:random:drop"] for p in per]).mean(1))
            d[name]["aopc_minus_random"] = ci(diff)
            diff_h = (np.array([p[f"{pert}:{name}:drop"] for p in per]).mean(1)
                      - np.array([p[f"{pert}:HCP:drop"] for p in per]).mean(1))
            d[name]["aopc_minus_hcp"] = ci(diff_h)
        out[pert] = d
    kp = [p for p in per if "top_ig_is_key" in p]
    if kp:
        kd = np.array([p["deal"] for p in kp])

        def kci(v):
            e, lo, hi = cluster_bootstrap(np.asarray(v, float), kd)
            return [round(e, 4), round(lo, 4), round(hi, 4)]
        out["key_cards"] = {"decisions": len(kp),
                            "top_absig_is_key": kci([p["top_ig_is_key"] for p in kp]),
                            "random_card_is_key": kci([p["n_key"] / 13 for p in kp]),
                            "auroc_absig": kci([p["auroc_absig_key"] for p in kp]),
                            "auroc_ig": kci([p["auroc_ig_key"] for p in kp]),
                            "auroc_hcp": kci([p["auroc_hcp_key"] for p in kp])}
    return out


def markdown(S: list, meta: dict) -> str:
    K = meta["K"]
    L = [f"# Is the Integrated Gradients card ranking faithful? ({meta['date']})", "",
         f"{meta['n']} decisions of the WBridge5 test split (seed {meta['seed']}, the card-swap study's), "
         f"K = {K} cards taken out in each ordering's order, R = {meta['R']} random replacement draws "
         "(`evaluation/attribution_deletion.py`). AOPC: mean drop of the call's probability over k = 1..K; "
         "[95% deal-cluster bootstrap].", ""]
    for pert, what in (("replace", "card replaced by a random card not in the hand"), ("remove", "card bit set to 0")):
        L += [f"## {pert}: {what}", "",
              "| Model | AOPC IG | AOPC abs IG | AOPC HCP | AOPC random | AOPC swaps (ref.) | IG - random | "
              f"call changed after {K}: IG / HCP / random |", "|---|---|---|---|---|---|---|---|"]
        for s in S:
            d = s[pert]
            f = lambda x: f"{x[0]:.3f}"
            L.append(f"| {s['model']} | {f(d['IG']['aopc'])} | {f(d['|IG|']['aopc'])} | {f(d['HCP']['aopc'])} | "
                     f"{f(d['random']['aopc'])} | {f(d['swaps']['aopc'])} | "
                     f"{d['IG']['aopc_minus_random'][0]:+.3f} [{d['IG']['aopc_minus_random'][1]:+.3f}, "
                     f"{d['IG']['aopc_minus_random'][2]:+.3f}] | "
                     f"{d['IG'][f'changed_at_{K}'][0]:.0%} / {d['HCP'][f'changed_at_{K}'][0]:.0%} / "
                     f"{d['random'][f'changed_at_{K}'][0]:.0%} |")
        L.append("")
    L += ["## Key cards", "", "| Model | decisions | top abs-IG card is key | a random card is key | AUROC abs IG | "
          "AUROC signed IG | AUROC HCP order |", "|---|---|---|---|---|---|---|"]
    for s in S:
        k = s.get("key_cards")
        if k:
            L.append(f"| {s['model']} | {k['decisions']} | {k['top_absig_is_key'][0]:.0%} | {k['random_card_is_key'][0]:.0%} | "
                     f"{k['auroc_absig'][0]:.2f} | {k['auroc_ig'][0]:.2f} | {k['auroc_hcp'][0]:.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=6, help="cards taken out, k = 1..K")
    ap.add_argument("--R", type=int, default=8, help="random draws (replacement cards, random orderings)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    from evaluation.common import load_test_split, sample_rows
    pos = load_test_split()
    rows = [int(r) for r in sample_rows(pos, args.n, seed=args.seed)]
    with mp.get_context("spawn").Pool(args.workers or len(args.models)) as pool:
        results = pool.map(analyse_model, [(m, rows, args.seed, args.K, args.R) for m in args.models])
    S = [summarise(r, args.K) for r in results]
    meta = {"n": len(rows), "seed": args.seed, "K": args.K, "R": args.R, "date": time.strftime("%Y-%m-%d"),
            "split": pos.source}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "summary": S,
                                          "per_decision": {r["model"]: r["per"] for r in results}}, indent=1))
    md = markdown(S, meta)
    if args.md:
        Path(args.md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
