#!/usr/bin/env python3
"""One checkpoint on every decision of a prepared split: accuracy, NLL, concept macro/micro F1 and the
concepts-only head, with the same definitions as training/sl.py's validation."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os

import jax
import numpy as np

from training.checkpoint import load_checkpoint
from training.concept_bottleneck import NUM_ACTIONS, OBS_SIZE


def evaluate(model_path: str, split_dir: str, chunk: int = 8192) -> dict:
    ck = load_checkpoint(model_path)
    num_hard = len(ck.hard_concepts)
    obs_bits = np.load(os.path.join(split_dir, "obs_bits.npy"))
    legal_bits = np.load(os.path.join(split_dir, "legal_bits.npy"))
    target_bits = np.load(os.path.join(split_dir, "target_bits.npy"))
    actions = np.load(os.path.join(split_dir, "action.npy")).astype(np.int32)
    fwd = jax.jit(ck.net.apply)

    n = len(actions)
    correct = co_correct = agree = 0
    err_strain = err_type = err_other = 0  # error breakdown, see compare_modes.py
    nll_sum = 0.0
    tp = np.zeros(num_hard); fp = np.zeros(num_hard); fn = np.zeros(num_hard)
    for s in range(0, n, chunk):
        rows = np.arange(s, min(s + chunk, n))
        obs = np.unpackbits(obs_bits[rows], axis=1)[:, :OBS_SIZE].astype(np.float32)
        legal = np.unpackbits(legal_bits[rows], axis=1)[:, :NUM_ACTIONS].astype(np.bool_)
        # a checkpoint with fewer concepts matches the first num_hard columns
        # (concepts/vocabulary.py only ever appends)
        tgt = np.unpackbits(target_bits[rows], axis=1)[:, :num_hard] > 0
        action = actions[rows]
        out = fwd(ck.params, obs)
        pl = np.asarray(out["policy_logits"]); col = np.asarray(out["concept_only_logits"])
        pred = np.where(legal, pl, -np.inf).argmax(1)
        co_pred = np.where(legal, col, -np.inf).argmax(1)
        correct += int((pred == action).sum()); co_correct += int((co_pred == action).sum())
        wrong = pred != action
        is_bid_p, is_bid_a = pred >= 3, action >= 3
        same_lvl = (pred - 3) // 5 == (action - 3) // 5
        e_strain = wrong & is_bid_p & is_bid_a & same_lvl
        e_type = wrong & ((is_bid_p != is_bid_a) | (~is_bid_p & ~is_bid_a))
        err_strain += int(e_strain.sum()); err_type += int(e_type.sum())
        err_other += int((wrong & ~e_strain & ~e_type).sum())
        agree += int((pred == co_pred).sum())
        z = pl - pl.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll_sum += float(-logp[np.arange(len(action)), action].sum())
        pb = np.asarray(out["concept_logits"]) > 0.0
        tp += (pb & tgt).sum(0); fp += (pb & ~tgt).sum(0); fn += (~pb & tgt).sum(0)
    has_pos = (tp + fn) > 0
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return {
        "model_path": model_path, "split": split_dir, "bottleneck_mode": ck.bottleneck_mode,
        "acc": correct / n, "nll": nll_sum / n,
        "concept_macro_f1": float(f1[has_pos].mean()),
        "concept_micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
        "concept_only_acc": co_correct / n, "concept_policy_agreement": agree / n,
        "n_decisions": n,
        "error_breakdown": {k: v / max(err_strain + err_type + err_other, 1) for k, v in (
            ("same_level_wrong_strain", err_strain), ("call_type", err_type), ("other", err_other))},
        "per_concept_f1": {c: float(f1[i]) for i, c in enumerate(ck.hard_concepts) if has_pos[i]},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--split", required=True, help="a prepared split directory, e.g. data/wbridge5/prepared/test")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    r = evaluate(args.model_path, args.split)
    print(f"[split_accuracy] {r['bottleneck_mode']}: acc={r['acc']*100:.2f}% nll={r['nll']:.3f} "
          f"concept_F1 macro={r['concept_macro_f1']:.3f} micro={r['concept_micro_f1']:.3f} "
          f"concept_only_acc={r['concept_only_acc']*100:.2f}% agree={r['concept_policy_agreement']*100:.2f}% "
          f"(n={r['n_decisions']:,})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
