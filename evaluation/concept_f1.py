#!/usr/bin/env python3
"""Concept F1 of each concept head against the rule engine, for WBridge5's call, for the model's own call,
and on its own self-play, plus the calibration (ECE) of the concept probabilities."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from concepts import AuctionContext, create_sayc_state_authority, decode_hand
from concepts.vocabulary import HARD_STATE_CONCEPTS
from evaluation.common import (BIDS, CONCEPT_MODELS, MODELS, checkpoint_stamp, load_test_split, masked_argmax,
                                  same_checkpoint, selfplay)
from training.checkpoint import load_checkpoint

RULE = create_sayc_state_authority()


def labels_for(pos, rows, calls) -> np.ndarray:
    out = np.zeros((len(rows), len(HARD_STATE_CONCEPTS)), bool)
    for j, (i, c) in enumerate(zip(rows, calls)):
        before, seat, dl = pos.auction(i)
        gt = RULE.classify(AuctionContext(bids=[b["bid"] for b in before], bid=BIDS[int(c)], dealer=dl),
                           decode_hand(pos.obs[i]))
        out[j] = [gt[n] for n in HARD_STATE_CONCEPTS]
    return out


def f1_table(prob: np.ndarray, gt: np.ndarray, names) -> dict:
    pred = prob >= 0.5
    tp = (pred & gt).sum(0); fp = (pred & ~gt).sum(0); fn = (~pred & gt).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)
    has = gt.sum(0) > 0
    # expected calibration error over every (decision, concept) prediction
    p, y = prob.ravel(), gt.ravel().astype(float)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == k].mean() - y[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
    return {"macro_f1": float(np.nanmean(f1[has])), "micro_f1": float(2 * tp.sum() / max(denom.sum(), 1)),
            "ece": float(ece), "concepts_with_positives": int(has.sum()),
            "per_concept": {n: {"f1": float(f1[k]), "support": int(gt[:, k].sum())}
                            for k, n in enumerate(names) if has[k]}}


def evaluate(model: str, test, selfplay_deals: int) -> dict:
    path = MODELS[model]
    ck = load_checkpoint(str(path))
    K = len(ck.hard_concepts)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    t0 = time.time()
    acts, preds = [], []
    for s in range(0, len(test), 8192):
        r = np.arange(s, min(s + 8192, len(test)))
        out = fwd(test.obs_f32(r))
        acts.append(np.asarray(out["concept_acts"]))
        preds.append(masked_argmax(np.asarray(out["policy_logits"]), test.legal[r]))
    acts, preds = np.concatenate(acts), np.concatenate(preds)
    gt_wb5 = test.targets[:, :K]
    diff = np.flatnonzero(preds != test.action)
    gt_self = gt_wb5.copy()
    gt_self[diff] = labels_for(test, diff, preds[diff])[:, :K]
    res = {"model": model, "path": str(path), "bottleneck_mode": ck.bottleneck_mode,
           "test_decisions": len(test), "model_differs_from_wbridge5": float(len(diff) / len(test)),
           "wb5": f1_table(acts, gt_wb5, ck.hard_concepts), "self": f1_table(acts, gt_self, ck.hard_concepts)}
    ppo = model.startswith("PPO")
    sp = selfplay(path, selfplay_deals, seed=0, random_vulnerability=ppo)
    sp_acts = np.concatenate([np.asarray(fwd(sp.obs_f32(np.arange(s, min(s + 8192, len(sp)))))["concept_acts"])
                              for s in range(0, len(sp), 8192)])
    gt_sp = labels_for(sp, np.arange(len(sp)), sp.action)[:, :K]
    res["selfplay"] = f1_table(sp_acts, gt_sp, ck.hard_concepts)
    res["selfplay"]["positions"] = sp.source
    print(f"[concepts] {model}: macro F1 wb5 {res['wb5']['macro_f1']:.3f}, self {res['self']['macro_f1']:.3f}, "
          f"selfplay {res['selfplay']['macro_f1']:.3f}; ECE {res['wb5']['ece']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=CONCEPT_MODELS)
    ap.add_argument("--selfplay_deals", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    test = load_test_split()
    for m in args.models:
        if m in res and same_checkpoint(res[m], MODELS[m]):  # a name whose checkpoint changed is recomputed
            continue
        res[m] = evaluate(m, test, args.selfplay_deals)
        res[m].update(checkpoint_stamp(MODELS[m]))
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
