#!/usr/bin/env python3
"""Does the policy use its concepts, and which ones? Exact interventions through the linear concepts->call
layer: zero and mean ablation per concept, linear attribution, and the bottleneck diagnostics (soft, hard, oracle,
strain masked, concepts or hidden zeroed, concepts-only head, test-time intervention curves)."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from evaluation.common import (MODELS, Positions, cluster_bootstrap, concept_category, kl_masked,
                                  load_test_split, log_softmax_masked, masked_argmax, selfplay)
from training.checkpoint import load_checkpoint

BUDGETS = [0, 1, 2, 4, 8, 16, 32, 64, 146]


def head_parts(ck):
    W = np.asarray(ck.params["policy_head"]["w"], np.float64)
    b = np.asarray(ck.params["policy_head"]["b"], np.float64)
    K = len(ck.hard_concepts)
    if ck.bottleneck_mode == "joint":
        return W[:-K], W[-K:], b
    return None, W, b


def analyse(model: str, pos: Positions, chunk: int = 512, seed: int = 0) -> dict:
    ck = load_checkpoint(str(MODELS.get(model, model)))
    names = list(ck.hard_concepts)
    K = len(names)
    Wh, Wc, b = head_parts(ck)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    strain = [i for i, n in enumerate(names) if n.startswith("bid_is_in_")]
    has_t = pos.targets is not None
    n = len(pos)

    # pass 1: activations (for mean ablation) and a check that the linear
    # decomposition reproduces the network's own logits
    act_sum = np.zeros(K)
    max_err = 0.0
    for s in range(0, n, 4096):
        r = np.arange(s, min(s + 4096, n))
        out = fwd(pos.obs_f32(r))
        a = np.asarray(out["concept_acts"], np.float64)
        act_sum += a.sum(0)
        L = a @ Wc + b + (np.asarray(out["hidden"], np.float64) @ Wh if Wh is not None else 0.0)
        max_err = max(max_err, float(np.abs(L - np.asarray(out["policy_logits"], np.float64)).max()))
    mu = act_sum / n
    assert max_err < 1e-3, f"linear head decomposition off by {max_err}"

    acc = {k: np.zeros(n, bool) for k in ("soft", "hard", "oracle", "strain_masked", "concepts_zeroed",
                                          "hidden_zeroed", "concept_only_head")}
    agree = {k: np.zeros(n, bool) for k in acc}
    interv = {o: {m: np.zeros(n, bool) for m in BUDGETS} for o in ("uncertain", "random")}
    kl0 = np.zeros(K); flip0 = np.zeros(K); klm = np.zeros(K); flipm = np.zeros(K)
    d0 = np.zeros((K, 38)); dm = np.zeros((K, 38))
    margin = np.zeros(K); margin_abs = np.zeros(K)
    call_m = np.zeros((38, K)); call_n = np.zeros(38)  # per chosen call: summed margin contributions
    concept_share = np.zeros(n)  # joint: |concept part| / (|concept| + |hidden|) of the margin
    conf = np.zeros(n); pred_all = np.zeros(n, np.int64)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for s in range(0, n, chunk):
        r = np.arange(s, min(s + chunk, n))
        out = fwd(pos.obs_f32(r))
        a = np.asarray(out["concept_acts"], np.float64)
        L = np.asarray(out["policy_logits"], np.float64)
        lg = pos.legal[r]
        logp = log_softmax_masked(L, lg)
        p = np.exp(logp)
        pred = masked_argmax(L, lg)
        pred_all[r] = pred
        conf[r] = p[np.arange(len(r)), pred]
        cpart = a @ Wc
        hpart = L - cpart - b

        def variant(Lv, key):
            v = masked_argmax(Lv, lg)
            agree[key][r] = v == pred
            if has_t:
                acc[key][r] = v == pos.action[r]

        variant(L, "soft")
        variant(L - cpart + (a > 0.5) @ Wc, "hard")
        if has_t:
            t = pos.targets[r][:, :K].astype(np.float64)
            variant(L - cpart + t @ Wc, "oracle")
        variant(L - a[:, strain] @ Wc[strain], "strain_masked")
        variant(L - cpart, "concepts_zeroed")
        variant(cpart + b, "hidden_zeroed")
        variant(np.asarray(out["concept_only_logits"], np.float64), "concept_only_head")

        # margin over the runner-up, split into concept and hidden parts
        z = np.where(lg, L, -np.inf)
        z[np.arange(len(r)), pred] = -np.inf
        runner = z.argmax(1)
        dW = Wc[:, pred].T - Wc[:, runner].T            # (c, K)
        m = a * dW
        margin += m.sum(0); margin_abs += np.abs(m).sum(0)
        np.add.at(call_m, pred, m)
        call_n += np.bincount(pred, minlength=38)
        cm = m.sum(1)
        hm = hpart[np.arange(len(r)), pred] - hpart[np.arange(len(r)), runner]
        concept_share[r] = np.abs(cm) / np.maximum(np.abs(cm) + np.abs(hm), 1e-12)

        # per-concept ablations
        for delta, kl_acc, flip_acc, d_acc in ((a, kl0, flip0, d0), (a - mu, klm, flipm, dm)):
            Lk = L[:, None, :] - delta[:, :, None] * Wc[None]
            logpk = log_softmax_masked(Lk, lg[:, None, :])
            kl_acc += kl_masked(logp[:, None, :], logpk, lg[:, None, :]).sum(0)
            flip_acc += (masked_argmax(Lk, lg[:, None, :]) != pred[:, None]).sum(0)
            d_acc += (np.exp(logpk) - p[:, None, :]).sum(0)

        if has_t:  # test-time interventions
            unc_rank = np.argsort(np.argsort(np.abs(a - 0.5), axis=1), axis=1)
            rnd_rank = np.argsort(rng.random(a.shape), axis=1)
            for order, rank in (("uncertain", unc_rank), ("random", rnd_rank)):
                for mb in BUDGETS:
                    a2 = np.where(rank < mb, t, a)
                    interv[order][mb][r] = masked_argmax(L - cpart + a2 @ Wc, lg) == pos.action[r]
        if (s // chunk) % 40 == 0:
            print(f"[influence] {model}: {s + len(r):,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)

    def ci(x):
        e, lo, hi = cluster_bootstrap(x.astype(float), pos.deal)
        return {"value": e, "ci95": [lo, hi]}

    diag = {}
    for key in acc:
        if key == "hidden_zeroed" and Wh is None:
            continue  # the bottleneck modes have no hidden path
        if key == "oracle" and not has_t:
            continue
        diag[key] = {"agreement_with_model": ci(agree[key])}
        if has_t:
            diag[key]["accuracy_vs_wbridge5"] = ci(acc[key])
    per_concept = []
    lin = mu * np.abs(Wc).sum(1)
    tot_kl = kl0.sum()
    for k, name in enumerate(names):
        per_concept.append({
            "name": name, "category": concept_category(name), "mean_act": mu[k],
            "prevalence": float(pos.targets[:, k].mean()) if has_t else None,
            "kl_zero": kl0[k] / n, "flip_zero": flip0[k] / n, "kl_mean": klm[k] / n, "flip_mean": flipm[k] / n,
            "kl_share": kl0[k] / tot_kl if tot_kl > 0 else 0.0,
            "linear_attr": lin[k], "margin_mean": margin[k] / n, "margin_abs_mean": margin_abs[k] / n,
        })
    res = {
        "model": model, "path": str(MODELS.get(model, model)), "bottleneck_mode": ck.bottleneck_mode,
        "positions": pos.source, "n_decisions": n, "n_deals": int(len(np.unique(pos.deal))),
        "linear_check_max_abs_err": max_err,
        "diagnostics": diag,
        "interventions": ({o: {str(m): float(v[m].mean()) for m in BUDGETS} for o, v in interv.items()}
                          if has_t else None),
        "concept_share_of_margin": {"mean": float(concept_share.mean()),
                                    "median": float(np.median(concept_share))},
        "mean_confidence": float(conf.mean()),
        "per_concept": per_concept,
    }
    mats = {"W_c": Wc, "delta_zero": d0 / n, "delta_mean": dm / n, "names": np.asarray(names),
            "call_contrib": call_m / np.maximum(call_n, 1)[:, None], "call_n": call_n}
    return res, mats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="names from evaluation.common.MODELS, or paths")
    ap.add_argument("--source", choices=("test", "selfplay"), default="test")
    ap.add_argument("--selfplay_deals", type=int, default=4000)
    ap.add_argument("--random_vulnerability", action="store_true")
    ap.add_argument("--max_deals", type=int, default=None, help="test split: only the first N deals")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    test = load_test_split(max_deals=args.max_deals) if args.source == "test" else None
    for model in args.models:
        pos = test if test is not None else selfplay(MODELS.get(model, model), args.selfplay_deals,
                                                     random_vulnerability=args.random_vulnerability)
        res, mats = analyse(model, pos)
        tag = f"{model.replace(' ', '_').replace('/', '_')}__{args.source}"  # a path model: no subdirectories
        (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
        np.savez_compressed(out / f"{tag}.npz", **mats)
        d = res["diagnostics"]
        acc = {k: v.get("accuracy_vs_wbridge5", v["agreement_with_model"])["value"] for k, v in d.items()}
        top = sorted(res["per_concept"], key=lambda c: -c["kl_zero"])[:5]
        print(f"[influence] {model} on {res['positions']}: " + ", ".join(f"{k} {v:.4f}" for k, v in acc.items()))
        print("   top KL: " + "; ".join(f"{c['name']} {c['kl_zero']:.3g}" for c in top), flush=True)


if __name__ == "__main__":
    main()
