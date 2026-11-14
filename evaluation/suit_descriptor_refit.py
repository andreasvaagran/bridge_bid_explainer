#!/usr/bin/env python3
"""Does the true bottleneck need the five suit descriptors? The independent model's concepts->call layer
is refitted on all concepts, without the descriptors, and on the descriptors alone."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from evaluation.common import MODELS, cluster_bootstrap, load_test_split

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data/wbridge5/prepared/train"


def concept_acts(ck, obs: np.ndarray) -> np.ndarray:
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o)["concept_acts"])
    return np.concatenate([np.asarray(fwd(jnp.asarray(obs[s:s + 8192], jnp.float32)))
                           for s in range(0, len(obs), 8192)])


def fit(C, legal, y, epochs: int, lr: float = 3e-3, batch: int = 8192, seed: int = 0):
    k = C.shape[1]
    params = {"w": jnp.zeros((k, legal.shape[1])), "b": jnp.zeros(legal.shape[1])}
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s, c, l, t):
        def loss(p):
            z = jnp.where(l, c @ p["w"] + p["b"], -1e9)
            return -jnp.mean(jax.nn.log_softmax(z)[jnp.arange(len(t)), t])
        g = jax.grad(loss)(p)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s
    rng = np.random.default_rng(seed)
    Cj, Lj, Yj = jnp.asarray(C), jnp.asarray(legal), jnp.asarray(y)
    for _ in range(epochs):
        perm = rng.permutation(len(C))
        for s in range(0, len(C), batch):
            i = jnp.asarray(perm[s:s + batch])
            params, state = step(params, state, Cj[i], Lj[i], Yj[i])
    return params


def predict(p, C, legal):
    z = np.asarray(C @ np.asarray(p["w"]) + np.asarray(p["b"]))
    return np.where(legal, z, -np.inf).argmax(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="SL independent")
    ap.add_argument("--train_decisions", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from training.checkpoint import load_checkpoint
    ck = load_checkpoint(str(MODELS[args.model]))
    names = list(ck.hard_concepts)
    desc = [i for i, n in enumerate(names) if n.startswith("bid_is_in_")]
    rest = [i for i in range(len(names)) if i not in desc]
    t0 = time.time()
    n = args.train_decisions
    total = len(np.load(TRAIN / "action.npy", mmap_mode="r"))
    rows = np.sort(np.random.default_rng(0).choice(total, size=min(n, total), replace=False))
    obs = np.unpackbits(np.load(TRAIN / "obs_bits.npy", mmap_mode="r")[rows], axis=1)[:, :480]
    legal = np.unpackbits(np.load(TRAIN / "legal_bits.npy", mmap_mode="r")[rows], axis=1)[:, :38] > 0
    y = np.load(TRAIN / "action.npy", mmap_mode="r")[rows].astype(np.int32)
    C = concept_acts(ck, obs)
    print(f"[refit] {len(rows):,} training decisions, concept activations ({time.time() - t0:.0f}s)", flush=True)
    test = load_test_split()
    Ct = concept_acts(ck, test.obs)
    res = {"model": args.model, "train_decisions": len(rows), "epochs": args.epochs,
           "strain_descriptors": [names[i] for i in desc], "fits": {}}
    W = np.asarray(ck.params["policy_head"]["w"]); b = np.asarray(ck.params["policy_head"]["b"])
    trained = np.where(test.legal, Ct @ W + b, -np.inf).argmax(1)
    e, lo, hi = cluster_bootstrap((trained == test.action).astype(float), test.deal)
    res["fits"]["trained policy layer (146 concepts)"] = {"agreement": e, "ci95": [lo, hi]}
    for tag, cols in (("refit, all 146 concepts", list(range(len(names)))), ("refit, without the 5 strain descriptors", rest),
                      ("refit, the 5 strain descriptors only", desc)):
        p = fit(C[:, cols], legal, y, args.epochs)
        pred = predict(p, Ct[:, cols], test.legal)
        e, lo, hi = cluster_bootstrap((pred == test.action).astype(float), test.deal)
        right_level_wrong_strain = float(np.mean([(a[0].isdigit() and a[0] == b[0] and a != b) for a, b in
                                                  zip(np.array(BIDS_)[pred[pred != test.action]],
                                                      np.array(BIDS_)[test.action[pred != test.action]])]))
        res["fits"][tag] = {"agreement": e, "ci95": [lo, hi], "concepts": len(cols),
                            "errors_right_level_wrong_strain": right_level_wrong_strain}
        print(f"[refit] {tag}: agreement {e:.4f} [{lo:.4f}, {hi:.4f}]; errors at the right level in the wrong strain "
              f"{right_level_wrong_strain:.2f} ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[done] -> {args.out}")


from evaluation.common import BIDS as BIDS_  # noqa: E402

if __name__ == "__main__":
    main()
