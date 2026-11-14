"""training/ppo_anchor.py: with the anchor off it equals the vendored update step; with it on the policy moves
toward the reference."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.update import make_update_step
from training.concept_bottleneck import NUM_ACTIONS, OBS_SIZE, build_net
from training.ppo_concept_labels import NUM_HARD
from training.ppo import PPOForward, Transition
from training.ppo_anchor import drift_metrics, kl_to_reference, make_anchored_update_step

S, E = 4, 8  # num_steps, num_envs
CONFIG = dict(actor_illegal_action_mask=True, actor_illegal_action_penalty=False, reward_scaling=False,
              value_clipping=True, clip_eps=0.2, vf_coef=0.5, ent_coef=0.001, illegal_action_l2norm_coef=0.0,
              minibatch_size=16, num_minibatches=2, num_steps=S, num_envs=E, update_epochs=2)


def _setup(seed=0):
    net = build_net("joint", "DeepMind", "relu", NUM_HARD, has_value=True)
    fwd = PPOForward(net)
    params = fwd.init(jax.random.PRNGKey(seed), jnp.zeros((1, OBS_SIZE)))
    rng = np.random.default_rng(seed)
    legal = rng.random((S, E, NUM_ACTIONS)) < 0.5
    legal[..., 0] = True
    action = np.array([[rng.choice(np.flatnonzero(legal[s, e])) for e in range(E)] for s in range(S)])
    traj = Transition(done=jnp.zeros((S, E), bool), action=jnp.asarray(action),
                      value=jnp.asarray(rng.normal(size=(S, E)), jnp.float32),
                      reward=jnp.zeros((S, E)), log_prob=jnp.asarray(rng.normal(size=(S, E)) - 3, jnp.float32),
                      obs=jnp.asarray(rng.random((S, E, OBS_SIZE)) < 0.1), legal_action_mask=jnp.asarray(legal))
    adv = jnp.asarray(rng.normal(size=(S, E)), jnp.float32)
    tgt = jnp.asarray(rng.normal(size=(S, E)), jnp.float32)
    return fwd, params, traj, adv, tgt


def test_anchor_off_matches_the_vendored_update_step():
    fwd, params, traj, adv, tgt = _setup()
    opt = optax.adam(1e-3, eps=1e-5)
    runner = (params, opt.init(params), None, None, 0, jax.random.PRNGKey(1))
    (p_vendored, *_), _ = make_update_step(CONFIG, fwd, opt)(runner, traj, adv, tgt)
    ref = jnp.zeros((S, E, NUM_ACTIONS))
    (p_anchored, *_), _ = make_anchored_update_step(CONFIG, fwd, opt, 0.0)(runner, traj, adv, tgt, ref)
    for a, b in zip(jax.tree_util.tree_leaves(p_vendored), jax.tree_util.tree_leaves(p_anchored)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_anchor_pulls_the_policy_toward_the_reference():
    fwd, params, traj, adv, tgt = _setup()
    _, ref_params, *_ = _setup(seed=7)
    obs = traj.obs.astype(jnp.float32)
    ref = fwd.apply(ref_params, obs)[0]
    legal = traj.legal_action_mask

    def kl_after(coef):
        opt = optax.adam(1e-3, eps=1e-5)
        runner = (params, opt.init(params), None, None, 0, jax.random.PRNGKey(1))
        step = jax.jit(make_anchored_update_step(CONFIG, fwd, opt, coef))
        for _ in range(5):
            runner, _ = step(runner, traj, adv, tgt, ref)
        return float(kl_to_reference(fwd.apply(runner[0], obs)[0], ref, legal).mean())

    assert kl_after(5.0) < kl_after(0.0)


def test_drift_metrics_identical_policies():
    logits = jnp.asarray(np.random.default_rng(0).normal(size=(6, NUM_ACTIONS)), jnp.float32)
    legal = jnp.ones((6, NUM_ACTIONS), bool)
    d = drift_metrics(logits, logits, legal)
    assert float(d["kl_to_sl"]) < 1e-6 and float(d["sl_agreement"]) == 1.0
