#!/usr/bin/env python3
"""Does a network use the concepts a probe can read from it? LEACE (Belrose et al. 2023) erases them from
a layer and the network runs on; reported: the share of calls that change, the KL, a random-subspace control and a
refitted probe that must fall to chance."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from evaluation.common import (MODELS, checkpoint_stamp, cluster_bootstrap, concept_category, kl_masked,
                                  load_test_split, log_softmax_masked, masked_argmax, same_checkpoint)

CALL_IDENTITY = ("strain", "pass / double")
DEFAULT_MODELS = ["brl SL", "brl RL", "brl RL+FSP", "brl RL+PFSP", "brl RL scratch", "SL plain", "SL joint",
                  "PPO joint", "PPO joint unanchored", "SL independent"]


# ---------------------------------------------------------------- networks as plain numpy

class Net:
    """A four-layer 1024-unit ReLU trunk and a head, as numpy: hidden(x, L) gives layer L's activations
    (L = 1..4), forward_from(h, L) the policy logits from (possibly edited) layer-L activations."""

    def __init__(self, trunk, head, kind):
        self.trunk, self.head, self.kind = trunk, head, kind

    def hidden(self, x, L):
        h = x
        for w, b in self.trunk[:L]:
            h = np.maximum(h @ w + b, 0.0)
        return h

    def forward_from(self, h, L):
        for w, b in self.trunk[L:]:
            h = np.maximum(h @ w + b, 0.0)
        return self.head(h)


def load_net(path) -> Net:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    p = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
    f32 = lambda a: np.asarray(a, np.float32)
    if isinstance(raw, dict) and "bottleneck_mode" in raw:  # this repo's concept models (DeepMind trunk)
        trunk = [(f32(p[k]["w"]), f32(p[k]["b"])) for k in ("linear", "linear_1", "linear_2", "linear_3")]
        Wc, bc = f32(p["hard_state_head"]["w"]), f32(p["hard_state_head"]["b"])
        Wp, bp = f32(p["policy_head"]["w"]), f32(p["policy_head"]["b"])
        if raw["bottleneck_mode"] == "joint":
            head = lambda h: np.concatenate([h, 1 / (1 + np.exp(-np.clip(h @ Wc + bc, -60, 60)))], -1) @ Wp + bp
        else:
            head = lambda h: (1 / (1 + np.exp(-np.clip(h @ Wc + bc, -60, 60)))) @ Wp + bp
        return Net(trunk, head, raw["bottleneck_mode"])
    trunk = [(f32(p[f"actor_critic/linear{s}"]["w"]), f32(p[f"actor_critic/linear{s}"]["b"]))
             for s in ("", "_1", "_2", "_3")]
    Wo, bo = f32(p["actor_critic/linear_4"]["w"]), f32(p["actor_critic/linear_4"]["b"])
    return Net(trunk, lambda h: h @ Wo + bo, "plain (brl)")


# ---------------------------------------------------------------- LEACE

class Whitener:
    """Sigma^{-1/2} and Sigma^{1/2} of the fit activations (pseudo-inverse over the non-degenerate
    eigen-directions: ReLU layers have dead units)."""

    def __init__(self, X, rel_tol=1e-7):
        self.mu = X.mean(0)
        Xc = (X - self.mu).astype(np.float64)
        S = Xc.T @ Xc / len(X)
        ev, U = np.linalg.eigh(S)
        keep = ev > rel_tol * ev.max()
        self.U, self.s = U[:, keep], np.sqrt(ev[keep])
        self.W = (self.U / self.s) @ self.U.T          # Sigma^{-1/2}
        self.Winv = (self.U * self.s) @ self.U.T       # Sigma^{1/2}
        self.rank = int(keep.sum())

    def eraser(self, Q):
        """The map x -> x - (x - mu) M^T for an orthonormal basis Q (columns) of the whitened subspace to
        remove, M = Sigma^{1/2} Q Q^T Sigma^{-1/2}, kept in its low-rank form (A, B) with M = A B^T:
        A = Sigma^{1/2} Q, B = Sigma^{-1/2} Q (Sigma^{-1/2} is symmetric), so applying it costs n d r, not n d^2."""
        return (self.Winv @ Q).astype(np.float32), (self.W @ Q).astype(np.float32)


def cross_cov(wh: Whitener, X, Z):
    """Cov(X, Z), d x k, in float64 (computed once per layer for all concepts)."""
    return (X - wh.mu).astype(np.float64).T @ (Z - Z.mean(0)).astype(np.float64) / len(X)


def leace_basis(wh: Whitener, X, Z, tol=1e-6, Sxz=None):
    """Orthonormal basis of colspace(Sigma^{-1/2} Cov(X, Z)) -- the subspace LEACE removes."""
    if Sxz is None:
        Sxz = cross_cov(wh, X, Z)
    A = wh.W @ Sxz
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > tol * max(s.max(), 1e-30)]


def random_basis(wh: Whitener, r, rng):
    """A random r-dimensional orthonormal basis inside the whitened support (the control)."""
    G = wh.U @ rng.standard_normal((wh.rank, r))
    Q, _ = np.linalg.qr(G)
    return Q


def apply(M, mu, X):
    """x - (x - mu) M^T for M = A B^T given as the pair (A, B)."""
    A, B = M
    return (X - ((X - mu.astype(np.float32)) @ B) @ A.T).astype(np.float32)


# ---------------------------------------------------------------- measurement

def probe_f1(Xfit, Yfit, Xev, Yev, epochs):
    """Held-out macro-F1 of a linear probe refitted on (possibly erased) activations."""
    from evaluation.concept_probes import fit_probe, macro_f1
    keep = (Yfit.sum(0) >= 20) & (Yev.sum(0) >= 20)
    if not keep.any():
        return None
    pred = fit_probe(Xfit, Yfit[:, keep], Xev, epochs=epochs)
    return macro_f1(pred, Yev[:, keep], min_pos=1)[0]


def effects(net, L, H_ev, legal, calls, wb5, deal, logp0):
    """Flip rate, mean KL and WBridge5 agreement of the network run on from erased layer-L activations."""
    out = np.concatenate([net.forward_from(H_ev[s:s + 8192], L) for s in range(0, len(H_ev), 8192)])
    logp = log_softmax_masked(out.astype(np.float64), legal)
    new = masked_argmax(out, legal)
    flip, lo, hi = cluster_bootstrap((new != calls).astype(float), deal)
    kl = float(kl_masked(logp0, logp, legal).mean())
    return {"flip": flip, "flip_ci95": [lo, hi], "kl": kl, "acc_wbridge5": float((new == wb5).mean())}


def analyse(model, pos, fit, ev, layers, n_random=5, probe_epochs=15, per_concept=True, seed=0):
    from concepts.vocabulary import HARD_STATE_CONCEPTS
    from evaluation.concept_completeness import own_call_labels
    t0 = time.time()
    net = load_net(MODELS.get(model, model))
    X = pos.obs_f32()
    logits = np.concatenate([net.forward_from(net.hidden(X[s:s + 8192], 0), 0) for s in range(0, len(X), 8192)])
    calls = masked_argmax(logits, pos.legal)
    Y = own_call_labels(pos, calls)
    names = list(HARD_STATE_CONCEPTS)[:Y.shape[1]]
    cats = np.array([concept_category(n) for n in names])
    Yf = Y.astype(np.float32)
    present = Y[fit].sum(0) > 0
    sets = {"all concepts": present,
            "no call identity": present & ~np.isin(cats, CALL_IDENTITY),
            "call identity only": present & np.isin(cats, CALL_IDENTITY)}
    for c in sorted(set(cats)):
        sets[f"category: {c}"] = present & (cats == c)
    legal_ev, calls_ev, wb5_ev, deal_ev = pos.legal[ev], calls[ev], pos.action[ev], pos.deal[ev]
    logp0 = log_softmax_masked(logits[ev].astype(np.float64), legal_ev)
    from evaluation.common import PLAIN_MODELS
    kind = "plain (joint recipe, concept losses off)" if model in PLAIN_MODELS else net.kind
    res = {"model": model, **checkpoint_stamp(MODELS.get(model, model)), "kind": kind,
           "acc_wbridge5": float((calls_ev == wb5_ev).mean()), "layers": {}}
    rng = np.random.default_rng(seed)
    for L in layers:
        H = np.concatenate([net.hidden(X[s:s + 8192], L) for s in range(0, len(X), 8192)])
        wh = Whitener(H[fit])
        S_all = cross_cov(wh, H[fit], Yf[fit])  # Cov(h, z) for every concept, once per layer
        lr = {"whitened_rank": wh.rank, "sets": {}}
        for sname, cols in sets.items():
            if not cols.any() or (L != layers[-1] and sname.startswith("category")):
                continue  # the categories only at the last layer
            Q = leace_basis(wh, None, None, Sxz=S_all[:, cols])
            r = Q.shape[1]
            He = apply(wh.eraser(Q), wh.mu, H[ev])
            e = effects(net, L, He, legal_ev, calls_ev, wb5_ev, deal_ev, logp0)
            ctrl = []
            for _ in range(n_random):
                Hr = apply(wh.eraser(random_basis(wh, r, rng)), wh.mu, H[ev])
                ctrl.append(effects(net, L, Hr, legal_ev, calls_ev, wb5_ev, deal_ev, logp0))
            entry = {"concepts": int(cols.sum()), "rank": r, "erased": e,
                     "random": {k: float(np.mean([c[k] for c in ctrl])) for k in ("flip", "kl", "acc_wbridge5")},
                     "random_sd": {k: float(np.std([c[k] for c in ctrl])) for k in ("flip", "kl")}}
            if sname in ("all concepts", "no call identity") and L == layers[-1]:  # can a refit probe still read them?
                Hfe = apply(wh.eraser(Q), wh.mu, H[fit])
                entry["probe_f1_before"] = probe_f1(H[fit], Y[fit][:, cols], H[ev], Y[ev][:, cols], probe_epochs)
                entry["probe_f1_after"] = probe_f1(Hfe, Y[fit][:, cols], He, Y[ev][:, cols], probe_epochs)
            lr["sets"][sname] = entry
            print(f"[erasure] {model} L{L} {sname:30s} rank {r:3d}: flip {e['flip']:.4f} "
                  f"(random {entry['random']['flip']:.4f}), KL {e['kl']:.4f} (random {entry['random']['kl']:.4f}), "
                  f"WB5 {e['acc_wbridge5']:.4f}"
                  + (f", probe F1 {entry['probe_f1_before']:.3f} -> {entry['probe_f1_after']:.3f}"
                     if "probe_f1_after" in entry and entry["probe_f1_after"] is not None else ""), flush=True)
        if per_concept and L == layers[-1]:
            pc = []
            for k in np.flatnonzero(present):
                Q = leace_basis(wh, None, None, Sxz=S_all[:, [k]])
                if Q.shape[1] == 0:
                    continue
                e = effects(net, L, apply(wh.eraser(Q), wh.mu, H[ev]), legal_ev, calls_ev, wb5_ev, deal_ev, logp0)
                pc.append({"name": names[k], "category": cats[k], "prevalence": float(Y[ev][:, k].mean()),
                           "flip": e["flip"], "kl": e["kl"]})
            lr["per_concept"] = pc
        res["layers"][str(L)] = lr
        del H
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="names from evaluation.common.MODELS, or paths")
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--max_deals", type=int, default=None)
    ap.add_argument("--random", type=int, default=3, help="random-subspace control draws per erasure")
    ap.add_argument("--probe_epochs", type=int, default=15)
    ap.add_argument("--no_per_concept", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = args.models or [m for m in DEFAULT_MODELS if m in MODELS and Path(MODELS[m]).exists()]
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
            print(f"[erasure] {m}: kept from {out}", flush=True)
            continue
        res["models"][m] = analyse(m, pos, fit, ev, args.layers, args.random, args.probe_epochs,
                                   not args.no_per_concept)
        out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
