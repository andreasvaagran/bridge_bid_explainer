#!/usr/bin/env python3
"""Leakage by refitting: the concepts->call layer fitted on the soft probabilities and on the same values
thresholded at 0.5. The accuracy and cross-entropy gap says whether the policy reads more than which concepts hold."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from evaluation.concept_completeness import fit_g, own_call_labels
from evaluation.common import (CONCEPT_MODELS, SEED_MODELS, MODELS, checkpoint_stamp, cluster_bootstrap,
                                  load_test_split, log_softmax_masked, masked_argmax, same_checkpoint)


def nll(logits, legal, y):
    lp = log_softmax_masked(logits.astype(np.float64), legal)
    return -lp[np.arange(len(y)), y]


def analyse(model, pos, fit, ev, epochs, with_reference=False):
    from training.checkpoint import load_checkpoint
    t0 = time.time()
    ck = load_checkpoint(str(MODELS.get(model, model)))
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    A, L = [], []
    for s in range(0, len(pos), 8192):
        out = fwd(pos.obs_f32(np.arange(s, min(s + 8192, len(pos)))))
        A.append(np.asarray(out["concept_acts"])); L.append(np.asarray(out["policy_logits"]))
    A = np.concatenate(A).astype(np.float32)
    calls = masked_argmax(np.concatenate(L), pos.legal)
    K = A.shape[1]
    inputs = {"soft": A, "hard": (A > 0.5).astype(np.float32)}
    if with_reference:  # the rule engine's labels (upper bounds: they partly name the call)
        inputs["rule labels, own call"] = own_call_labels(pos, calls)[:, :K].astype(np.float32)
        inputs["rule labels, WBridge5's call"] = pos.targets[:, :K].astype(np.float32)
    res = {"model": model, **checkpoint_stamp(MODELS.get(model, model)), "bottleneck_mode": ck.bottleneck_mode,
           "model_accuracy": float((calls[ev] == pos.action[ev]).mean()), "heads": {}}
    for head, hidden in (("linear", 0), ("mlp", 256)):
        hits, ces = {}, {}
        for name, V in inputs.items():
            for target, y in (("wbridge5", pos.action), ("own", calls)):
                lg = fit_g(V[fit], y[fit], pos.legal[fit], V[ev], pos.legal[ev], hidden=hidden, epochs=epochs,
                           return_logits=True)
                hits[name, target] = (masked_argmax(lg, pos.legal[ev]) == y[ev]).astype(float)
                ces[name, target] = nll(lg, pos.legal[ev], y[ev])
        h = {}
        for target in ("wbridge5", "own"):
            h[target] = {name: {"accuracy": float(hits[name, target].mean()), "nll": float(ces[name, target].mean())}
                         for name in inputs}
            d, lo, hi = cluster_bootstrap(hits["soft", target] - hits["hard", target], pos.deal[ev])
            dn, lon, hin = cluster_bootstrap(ces["hard", target] - ces["soft", target], pos.deal[ev])
            h[target]["leakage_gap_accuracy"] = {"value": d, "ci95": [lo, hi]}
            h[target]["leakage_gap_nll"] = {"value": dn, "ci95": [lon, hin]}
        res["heads"][head] = h
        print(f"[leakage] {model} ({head}): WB5 soft {h['wbridge5']['soft']['accuracy']:.4f} hard "
              f"{h['wbridge5']['hard']['accuracy']:.4f} gap {h['wbridge5']['leakage_gap_accuracy']['value']:+.4f} "
              f"[{h['wbridge5']['leakage_gap_accuracy']['ci95'][0]:+.4f}, {h['wbridge5']['leakage_gap_accuracy']['ci95'][1]:+.4f}]"
              f", NLL gap {h['wbridge5']['leakage_gap_nll']['value']:+.4f}; own soft {h['own']['soft']['accuracy']:.4f}"
              f" hard {h['own']['hard']['accuracy']:.4f}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--max_deals", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--with_reference", action="store_true", help="also fit on the rule engine's labels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = args.models or [m for m in CONCEPT_MODELS + SEED_MODELS if m in MODELS]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pos = load_test_split(max_deals=args.max_deals)
    fit = pos.deal % 10 < 7
    ev = ~fit
    res = {"positions": pos.source, "fit_decisions": int(fit.sum()), "eval_decisions": int(ev.sum()), "models": {}}
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("positions") == res["positions"]:
            res["models"] = {m: r for m, r in old.get("models", {}).items() if same_checkpoint(r, MODELS.get(m, m))}
    for m in models:
        if m in res["models"]:
            continue
        res["models"][m] = analyse(m, pos, fit, ev, args.epochs, args.with_reference)
        out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
