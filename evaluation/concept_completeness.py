#!/usr/bin/env python3
"""How much of a network's bidding its concepts recover (Yeh et al. 2020): a small network fit from
concept scores to the call, as fidelity to the model's own call and Yeh's score against WBridge5's. Concept
models use their own activations, brl's networks a linear probe. Controls: the call-identity concepts alone and a
random-init network."""
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

from evaluation.common import (MODELS, checkpoint_stamp, cluster_bootstrap, concept_category, load_test_split,
                                  masked_argmax, same_checkpoint, SEED_MODELS, PLAIN_MODELS)
from training.concept_bottleneck import NUM_ACTIONS

CALL_IDENTITY = ("strain", "pass / double")
MIN_HEADROOM = 0.10  # Yeh's score needs acc(f) at least this far above the baseline
DEFAULT_MODELS = ["SL joint", "SL independent", "SL sequential", "PPO joint", "PPO independent", "PPO sequential",
                  "brl SL", "brl RL", "brl RL+FSP", "brl RL+PFSP"] + SEED_MODELS + ["brl RL scratch"] \
    + PLAIN_MODELS


def fit_g(V, y, legal, V_ev, legal_ev, hidden=256, epochs=10, batch=512, lr=1e-3, seed=0, return_logits=False):
    """Yeh's g: a one-hidden-layer MLP from concept scores to the 38 calls, cross-entropy over the legal
    calls only. Returns its calls on (V_ev, legal_ev), or its logits with return_logits=True (hidden=0: a
    linear map, the form of the bottlenecks' own concepts-to-call layer)."""
    mu, sd = V.mean(0), V.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)  # a concept (almost) constant on the fit deals is centred, not blown up
    V, V_ev = (V - mu) / sd, (V_ev - mu) / sd
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    d = V.shape[1]
    if hidden:
        params = {"w1": jax.random.normal(k1, (d, hidden)) * np.sqrt(2.0 / max(d, 1)), "b1": jnp.zeros(hidden),
                  "w2": jax.random.normal(k2, (hidden, NUM_ACTIONS)) * np.sqrt(1.0 / hidden),
                  "b2": jnp.zeros(NUM_ACTIONS)}
    else:
        params = {"w2": jnp.zeros((d, NUM_ACTIONS)), "b2": jnp.zeros(NUM_ACTIONS)}
    opt = optax.adam(lr)
    state = opt.init(params)

    def logits(p, x):
        h = jax.nn.relu(x @ p["w1"] + p["b1"]) if "w1" in p else x
        return h @ p["w2"] + p["b2"]

    @jax.jit
    def step(p, s, x, t, m):
        def loss(p):
            z = jnp.where(m, logits(p, x), -1e9)
            return -jnp.take_along_axis(jax.nn.log_softmax(z), t[:, None], 1).mean()
        upd, s = opt.update(jax.grad(loss)(p), s)
        return optax.apply_updates(p, upd), s

    rng = np.random.default_rng(seed)
    Vj, yj, mj = jnp.asarray(V, jnp.float32), jnp.asarray(y), jnp.asarray(legal)
    for _ in range(epochs):
        perm = rng.permutation(len(V))
        for s in range(0, len(V), batch):
            idx = jnp.asarray(perm[s:s + batch])
            params, state = step(params, state, Vj[idx], yj[idx], mj[idx])
    out = np.concatenate([np.asarray(logits(params, jnp.asarray(V_ev[s:s + 16384], jnp.float32)))
                          for s in range(0, len(V_ev), 16384)])
    return out if return_logits else masked_argmax(out, legal_ev)


def baseline_calls(y_fit, legal_ev):
    """a_r's calls: the most frequent call on the fit deals that is legal here."""
    freq = np.bincount(y_fit, minlength=NUM_ACTIONS).astype(float)
    return masked_argmax(np.broadcast_to(np.log(freq + 1e-3), legal_ev.shape), legal_ev)


def score(V, y, legal, fit, ev, deal, ref_acc, epochs=10, seed=0):
    """g: V -> y fit on `fit`, scored on `ev`. ref_acc is acc(f) for Yeh's score, 1 for fidelity. The
    completeness score is None when ref_acc is within MIN_HEADROOM of the baseline."""
    pred = fit_g(V[fit], y[fit], legal[fit], V[ev], legal[ev], epochs=epochs, seed=seed)
    a_r = float((baseline_calls(y[fit], legal[ev]) == y[ev]).mean())
    acc, lo, hi = cluster_bootstrap((pred == y[ev]).astype(float), deal[ev])
    res = {"g_accuracy": {"value": acc, "ci95": [lo, hi]}, "baseline_accuracy": a_r, "reference_accuracy": ref_acc}
    den = ref_acc - a_r
    if den < MIN_HEADROOM:
        res["completeness"] = {"value": None, "ci95": None,
                               "undefined": f"reference accuracy {ref_acc:.3f} is within {MIN_HEADROOM} "
                                            f"of the baseline {a_r:.3f}"}
    else:
        res["completeness"] = {"value": (acc - a_r) / den, "ci95": [(lo - a_r) / den, (hi - a_r) / den]}
    return res


def own_call_labels(pos, calls):
    """The rule engine's concepts of each position's call in `calls` (pos.targets holds those of
    WBridge5's call; relabel where the model bids otherwise)."""
    from evaluation.concept_f1 import labels_for
    Y = pos.targets.copy()
    diff = np.flatnonzero(calls != pos.action)
    if len(diff):
        Y[diff] = labels_for(pos, diff, calls[diff])[:, :Y.shape[1]]
    return Y


def probe_scores(H, Y, fit, epochs):
    """Concept probabilities from a linear probe H -> Y fit on the `fit` rows (evaluation/concept_probes.py)."""
    from evaluation.concept_probes import fit_probe
    z = fit_probe(H[fit], Y[fit], H, epochs=epochs, logits=True)
    return 1.0 / (1.0 + np.exp(-z))


def concept_scores(model, pos, fit, probe_epochs):
    """(v_c(x) for every position, the model's own calls, the concepts of those calls, concept names,
    how v_c was obtained)."""
    import pickle
    from concepts.vocabulary import HARD_STATE_CONCEPTS
    path = MODELS.get(model, model)
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict) and "bottleneck_mode" in raw:
        from training.checkpoint import load_checkpoint
        ck = load_checkpoint(str(path))
        fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
        V, L, Hs = [], [], []
        for s in range(0, len(pos), 8192):
            out = fwd(pos.obs_f32(np.arange(s, min(s + 8192, len(pos)))))
            V.append(np.asarray(out["concept_acts"])); L.append(np.asarray(out["policy_logits"]))
            if model in PLAIN_MODELS:
                Hs.append(np.asarray(out["hidden"]))
        calls = masked_argmax(np.concatenate(L), pos.legal)
        names = list(ck.hard_concepts)
        Y = own_call_labels(pos, calls)[:, :len(names)]
        if model in PLAIN_MODELS:  # never shown the concepts (its head is untrained): read them like brl's
            return (probe_scores(np.concatenate(Hs), Y, fit, probe_epochs), calls, Y, names,
                    "linear probe, last hidden layer")
        return np.concatenate(V), calls, Y, names, f"concept head ({ck.bottleneck_mode})"
    from evaluation.concept_probes import brl_layers
    hs, logits = brl_layers(path)(pos.obs_f32())
    calls = masked_argmax(logits, pos.legal)
    Y = own_call_labels(pos, calls)
    return (probe_scores(hs[-1], Y, fit, probe_epochs), calls, Y, list(HARD_STATE_CONCEPTS)[:Y.shape[1]],
            "linear probe, last hidden layer")


def analyse(model, pos, fit, ev, random_hidden=None, g_epochs=10, probe_epochs=25):
    """Every variant for the model's own concept scores; fidelity only for the random-init control."""
    t0 = time.time()
    V, calls, Y, names, how = concept_scores(model, pos, fit, probe_epochs)
    f_acc = float((calls[ev] == pos.action[ev]).mean())
    cats = [concept_category(n) for n in names]
    variants = {"all concepts": np.ones(len(names), bool),
                "no call identity": np.array([c not in CALL_IDENTITY for c in cats]),
                "call identity only": np.array([c in CALL_IDENTITY for c in cats])}
    sources = {"model": (V, how)}
    if random_hidden is not None:
        sources["random-init control"] = (probe_scores(random_hidden, Y, fit, probe_epochs),
                                          "linear probe, last hidden layer of a randomly initialised network")
    res = {"model": model, **checkpoint_stamp(MODELS.get(model, model)), "model_accuracy": f_acc, "sources": {}}
    for src, (S, src_how) in sources.items():
        entry = {"concept_scores": src_how, "variants": {}}
        for variant, cols in variants.items():
            e = {"concepts": int(cols.sum()),
                 "fidelity": score(S[:, cols], calls, pos.legal, fit, ev, pos.deal, 1.0, g_epochs)}
            if src == "model":
                e["vs_wbridge5"] = score(S[:, cols], pos.action, pos.legal, fit, ev, pos.deal, f_acc, g_epochs)
            entry["variants"][variant] = e
        res["sources"][src] = entry
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="names from evaluation.common.MODELS, or paths")
    ap.add_argument("--max_deals", type=int, default=None, help="only the first N deals of the test split")
    ap.add_argument("--g_epochs", type=int, default=10)
    ap.add_argument("--probe_epochs", type=int, default=25)
    ap.add_argument("--no_control", action="store_true", help="skip the random-init control")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = args.models or [m for m in DEFAULT_MODELS if m in MODELS and Path(MODELS[m]).exists()]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pos = load_test_split(max_deals=args.max_deals)
    fit = pos.deal % 10 < 7
    ev = ~fit
    random_hidden = None
    if not args.no_control:
        from evaluation.concept_probes import random_layers
        random_hidden = random_layers()(pos.obs_f32())[0][-1]
    res = {"positions": pos.source, "fit_decisions": int(fit.sum()), "eval_decisions": int(ev.sum()),
           "min_headroom": MIN_HEADROOM, "models": {}}
    if out.exists():  # resume: keep the models whose checkpoint is unchanged (same name, path and mtime)
        old = json.loads(out.read_text())
        if old.get("positions") == res["positions"] and old.get("min_headroom") == MIN_HEADROOM:
            res["models"] = {m: r for m, r in old.get("models", {}).items()
                             if "sources" in r and same_checkpoint(r, MODELS.get(m, m))}

    def fmt(s):
        v = s["completeness"]["value"]
        return "undefined" if v is None else f"{v:.3f}"

    for m in models:
        if m in res["models"]:
            print(f"[completeness] {m}: kept from {out}", flush=True)
            continue
        r = analyse(m, pos, fit, ev, random_hidden, args.g_epochs, args.probe_epochs)
        res["models"][m] = r
        out.write_text(json.dumps(res, indent=1))
        for src, entry in r["sources"].items():
            for v, e in entry["variants"].items():
                fi = e["fidelity"]
                wb = f"  eta(WBridge5) {fmt(e['vs_wbridge5'])}" if "vs_wbridge5" in e else ""
                print(f"[completeness] {m:16s} {src:19s} {v:18s} fidelity {fi['g_accuracy']['value']:.4f}  "
                      f"eta_own {fmt(fi)}{wb}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
