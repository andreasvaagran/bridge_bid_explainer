#!/usr/bin/env python3
"""Linear probes for the 146 concepts in each layer of the brl networks and our models, with the
observation and a random-init network as controls. Labels are the concepts of the network's own call."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from evaluation.common import MODELS, load_test_split, masked_argmax
from evaluation.concept_f1 import labels_for

BRL = ["brl SL", "brl RL", "brl RL+FSP", "brl RL+PFSP"]


def brl_layers(path):
    """Activations of the four hidden layers of a brl DeepMind/relu network, and its policy logits."""
    obj = pickle.load(open(path, "rb"))
    p = obj["params"] if isinstance(obj, dict) and "params" in obj else obj
    W = [(np.asarray(p[f"actor_critic/linear{s}"]["w"], np.float32), np.asarray(p[f"actor_critic/linear{s}"]["b"], np.float32))
         for s in ("", "_1", "_2", "_3", "_4")]

    def f(x):
        hs = []
        h = x
        for w, b in W[:4]:
            h = np.maximum(h @ w + b, 0.0)
            hs.append(h)
        return hs, h @ W[4][0] + W[4][1]
    return f


def random_layers(seed=0):
    rng = np.random.default_rng(seed)
    W = []
    for din, dout in ((480, 1024), (1024, 1024), (1024, 1024), (1024, 1024)):
        W.append((rng.normal(0, np.sqrt(2.0 / din), (din, dout)).astype(np.float32), np.zeros(dout, np.float32)))

    def f(x):
        hs, h = [], x
        for w, b in W:
            h = np.maximum(h @ w + b, 0.0)
            hs.append(h)
        return hs, None
    return f


def fit_probe(Xtr, Ytr, Xte, epochs=25, batch=4096, lr=2e-3, l2=1e-5, seed=0, logits=False):
    """Held-out predictions (bool), or the logits with `logits=True` (concept scores for concept_completeness)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    d, k = Xtr.shape[1], Ytr.shape[1]
    params = {"w": jnp.zeros((d, k)), "b": jnp.asarray(np.log((Ytr.mean(0) + 1e-4) / (1 - Ytr.mean(0) + 1e-4)), jnp.float32)}
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, x, y):
        def loss(p):
            z = x @ p["w"] + p["b"]
            return optax.sigmoid_binary_cross_entropy(z, y).mean() + l2 * (p["w"] ** 2).sum()
        g = jax.grad(loss)(params)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state

    rng = np.random.default_rng(seed)
    n = len(Xtr)
    Xj, Yj = jnp.asarray(Xtr, jnp.float32), jnp.asarray(Ytr, jnp.float32)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = jnp.asarray(perm[s:s + batch])
            params, state = step(params, state, Xj[idx], Yj[idx])
    z = np.asarray(jnp.asarray(Xte, jnp.float32) @ params["w"] + params["b"])
    return z if logits else z > 0


def macro_f1(pred, y, min_pos=20):
    tp = (pred & y).sum(0); fp = (pred & ~y).sum(0); fn = (~pred & y).sum(0)
    keep = y.sum(0) >= min_pos
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1[keep].mean()), int(keep.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_deals", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--trunks", nargs="*", default=None,
                    help="label=checkpoint pairs: probe only these concept models' last trunk layers (e.g. the PPO "
                         "anchor-ablation runs) and add them to an existing --out file")
    ap.add_argument("--brl", nargs="*", default=None,
                    help="probe only these published networks (e.g. \"brl RL+FSP\") and add them to an existing --out file")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pos = load_test_split(max_deals=args.max_deals)
    X = pos.obs.astype(np.float32)
    tr = pos.deal % 10 < 7
    te = ~tr
    Y_wb5 = pos.targets
    res = {"positions": pos.source, "train_decisions": int(tr.sum()), "test_decisions": int(te.sum()), "probes": {}}
    if (args.trunks or args.brl) and out.exists():
        res = json.loads(out.read_text())

    def run(name, layers_fn):
        hs, logits = layers_fn(X)
        if logits is not None:
            calls = masked_argmax(logits, pos.legal)
            diff = np.flatnonzero(calls != pos.action)
            Y_own = Y_wb5.copy()
            Y_own[diff] = labels_for(pos, diff, calls[diff])
        else:
            Y_own = None
        r = {}
        for li, H in enumerate([X] + hs):
            lab = "input" if li == 0 else f"layer {li}"
            t0 = time.time()
            entry = {}
            for tag, Y in (("own_call", Y_own), ("wbridge5_call", Y_wb5)):
                if Y is None:
                    continue
                pred = fit_probe(H[tr], Y[tr], H[te], epochs=args.epochs)
                f1, nc = macro_f1(pred, Y[te])
                entry[tag] = {"macro_f1": f1, "concepts": nc}
            r[lab] = entry
            print(f"[probe] {name:14s} {lab:8s} " + "  ".join(f"{k} F1 {v['macro_f1']:.3f}" for k, v in entry.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        if logits is not None:
            r["same_call_as_wbridge5"] = float((calls == pos.action).mean())
        res["probes"][name] = r
        out.write_text(json.dumps(res, indent=1))

    from training.checkpoint import load_checkpoint
    if args.brl:
        for m in args.brl:
            run(m, brl_layers(MODELS[m]))
        out.write_text(json.dumps(res, indent=1))
        return
    if args.trunks:
        targets = [tuple(t.split("=", 1)) for t in args.trunks]
    else:
        for m in BRL:
            run(m, brl_layers(MODELS[m]))
        run("random init", random_layers())
        targets = [(m, str(MODELS[m])) for m in ("SL joint", "SL independent")]
    # this repo's models: the trunk's last hidden layer, own-call labels (the concept head itself is the reference)
    for m, path in targets:
        ck = load_checkpoint(str(path))
        fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))

        def layers(x, fwd=fwd):
            outs = [fwd(x[s:s + 8192]) for s in range(0, len(x), 8192)]
            hid = np.concatenate([np.asarray(o["hidden"]) for o in outs])
            lg = np.concatenate([np.asarray(o["policy_logits"]) for o in outs])
            return [hid], lg
        run(m if args.trunks else m + " (trunk)", layers)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
