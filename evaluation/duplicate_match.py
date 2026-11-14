#!/usr/bin/env python3
"""Bidding strength: model A vs model B in duplicate on held-out deals (dds_results_500K.npy), in IMPs per
board from the double-dummy results. It loads our checkpoints and brl's networks alike. The vendored
src/duplicate.py drops pgx 2.6's state._dds_val at the second table (a model vs itself came out at -0.76
IMPs/board); `_duplicate_step` here copies it. --vulnerability none matches the SL data."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import math
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding, _observe

from src.duplicate import Table_info, _duplicate_init, _imp_reward

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DEFAULT_DDS = _os.path.join(_REPO, "dds_results", "dds_results_500K.npy")


def load_policy(path: str, activation: str = "relu", model_type: str = "DeepMind"):
    """(params, logits_fn(params, obs) -> policy logits) for either kind of model."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "bottleneck_mode" in obj:
        from training.checkpoint import load_checkpoint
        ck = load_checkpoint(path)
        return ck.params, (lambda p, obs: ck.net.apply(p, obs.astype(jnp.float32))["policy_logits"]), \
            f"{ck.bottleneck_mode} concept-bottleneck"
    from src.models import make_forward_pass
    fwd = make_forward_pass(activation=activation, model_type=model_type)
    params = obj["params"] if isinstance(obj, dict) and "params" in obj else obj
    return params, (lambda p, obs: fwd.apply(p, obs.astype(jnp.float32))[0]), f"upstream {model_type}/{activation}"


def _table_info(state):
    return Table_info(terminated=state.terminated, rewards=state.rewards, last_bid=state._last_bid,
                      last_bidder=state._last_bidder, call_x=state._call_x, call_xx=state._call_xx)


def _duplicate_init_fixed(state):
    """src/duplicate.duplicate_init, keeping the deal's double-dummy table."""
    dup = _duplicate_init(state).replace(_dds_val=state._dds_val)
    return dup.replace(observation=jax.lax.stop_gradient(_observe(dup, dup.current_player)))


def _duplicate_step(step_fn):
    """src/duplicate.duplicate_step with _duplicate_init_fixed (see module docstring)."""
    def wrapped_step(state, action, table_a_info, table_b_info):
        state = step_fn(state, action)
        next_state = jax.lax.cond(~table_a_info.terminated & state.terminated,
                                  lambda: _duplicate_init_fixed(state), lambda: state)
        next_state = jax.lax.cond(
            table_a_info.terminated & state.terminated & ~table_b_info.terminated,
            lambda: state.replace(rewards=_imp_reward(table_a_info.rewards, state.rewards)),
            lambda: next_state.replace(rewards=jnp.zeros(4, dtype=jnp.float32)))
        table_b_info = jax.lax.cond(state.terminated & table_a_info.terminated & ~table_b_info.terminated,
                                    lambda: _table_info(state), lambda: table_b_info)
        table_a_info = jax.lax.cond(~table_a_info.terminated & state.terminated,
                                    lambda: _table_info(state), lambda: table_a_info)
        return next_state, table_a_info, table_b_info
    return wrapped_step


def make_duplicate_match(env: BridgeBidding, logits_a, logits_b, n: int, non_vulnerable: bool):
    """jit-able (params_a, params_b, key) -> per-board IMPs for team A (players 0/1)."""
    step_fn = _duplicate_step(env.step)
    table_info = _table_info

    def act(params_a, params_b, state):
        legal = state.legal_action_mask.astype(jnp.bool_)
        la = logits_a(params_a, state.observation[None])[0]
        lb = logits_b(params_b, state.observation[None])[0]
        logits = jnp.where((state.current_player == 0) | (state.current_player == 1), la, lb)
        return jnp.argmax(jnp.where(legal, logits, jnp.finfo(jnp.float32).min))

    def match(params_a, params_b, key):
        state = jax.vmap(env.init)(jax.random.split(key, n))
        if non_vulnerable:
            state = state.replace(_vul_NS=jnp.zeros(n, jnp.bool_), _vul_EW=jnp.zeros(n, jnp.bool_))
            state = state.replace(observation=jax.vmap(_observe)(state, state.current_player))
        ta, tb = table_info(state), table_info(state)

        def cond(c):
            return ~c[0].terminated.all()

        def body(c):
            state, ta, tb, ret = c
            action = jax.vmap(lambda s: act(params_a, params_b, s))(state)
            state, ta, tb = jax.vmap(step_fn)(state, action, ta, tb)
            return state, ta, tb, ret + state.rewards[:, 0]

        _, _, _, ret = jax.lax.while_loop(cond, body, (state, ta, tb, jnp.zeros(n)))
        return ret

    return match


def evaluate(a: str, b: str, num_deals: int = 10_000, batch: int = 1000, seed: int = 0,
             dds_path: str = DEFAULT_DDS, vulnerability: str = "random",
             a_activation: str = "relu", a_model_type: str = "DeepMind",
             b_activation: str = "relu", b_model_type: str = "DeepMind") -> dict:
    env = BridgeBidding(dds_results_table_path=dds_path)
    pa, fa, kind_a = load_policy(a, a_activation, a_model_type)
    pb, fb, kind_b = load_policy(b, b_activation, b_model_type)
    match = jax.jit(make_duplicate_match(env, fa, fb, batch, vulnerability == "none"))
    key = jax.random.PRNGKey(seed)
    imps, t0 = [], time.time()
    for _ in range(math.ceil(num_deals / batch)):
        key, sub = jax.random.split(key)
        imps.append(np.asarray(match(pa, pb, sub)))
    imps = np.concatenate(imps)[:num_deals]
    se = float(imps.std(ddof=1) / np.sqrt(len(imps)))
    return {"a": a, "b": b, "a_kind": kind_a, "b_kind": kind_b, "deals": int(len(imps)),
            "imps_per_board": float(imps.mean()), "se": se,
            "ci95": [float(imps.mean() - 1.96 * se), float(imps.mean() + 1.96 * se)],
            "a_won": float((imps > 0).mean()), "tied": float((imps == 0).mean()), "b_won": float((imps < 0).mean()),
            "vulnerability": vulnerability, "dds_path": dds_path, "seed": seed,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="model A (the one whose IMPs are reported)")
    ap.add_argument("--b", required=True, help="model B (the opponent)")
    ap.add_argument("--num_deals", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dds_path", default=DEFAULT_DDS, help="held-out DDS table (not the 10M PPO training table)")
    ap.add_argument("--vulnerability", choices=("random", "none"), default="random")
    for side in ("a", "b"):
        ap.add_argument(f"--{side}_activation", default="relu", help="only for upstream (non-concept) models")
        ap.add_argument(f"--{side}_model_type", default="DeepMind", help="only for upstream (non-concept) models")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args()
    r = evaluate(args.a, args.b, args.num_deals, args.batch, args.seed, args.dds_path, args.vulnerability,
                 args.a_activation, args.a_model_type, args.b_activation, args.b_model_type)
    print(f"{r['a']}\n  vs {r['b']}\n  {r['deals']:,} boards ({r['vulnerability']} vulnerability): "
          f"{r['imps_per_board']:+.3f} IMPs/board +- {r['se']:.3f} (95% CI {r['ci95'][0]:+.3f} .. {r['ci95'][1]:+.3f}); "
          f"A better on {r['a_won']:.1%}, tied {r['tied']:.1%}, B better {r['b_won']:.1%}  [{r['seconds']}s]")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
