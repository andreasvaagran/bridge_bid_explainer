#!/usr/bin/env python3
"""Self-play PPO fine-tuning of a training/sl.py checkpoint, for any bottleneck mode.

All four seats bid with the current parameters (game_mode="competitive"); only the learner's calls are training
transitions. game_mode="free-run" has the opponents always pass; the mode is recorded in every checkpoint and in
launches.jsonl. Each update can add:
  * a concept step: rule-engine labels for the agent's own calls (training/ppo_concept_labels.py) supervise the
    concept head (concept_coef; 0 keeps the SL concept model fixed, sequential forces it);
  * a KL anchor to the SL policy, sl_anchor_coef * KL(pi || pi_SL) (training/ppo_anchor.py).
Drift from SL (KL and same-call rate) is logged every 10 updates to metrics.jsonl.

Run by pipeline/train.sh (ppo_run in pipeline/lib.sh, which adds the full settings and pipeline/converge.py's
stopping rule), e.g.

    python -m training.ppo bottleneck_mode=joint load_initial_model=True \\
        initial_model_path=runs/sl/joint/params-best.pkl sl_anchor_coef=0.01 \\
        sl_anchor_path=runs/sl/joint/params-best.pkl log_path=runs/ppo/joint exp_name=ppo"""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import pickle
import time
from typing import Literal, NamedTuple

import numpy as np

from training.ppo_concept_labels import (  # numpy + concepts only, no JAX
    AUCTION_BITS, HAND_BITS, HARD_STATE, NUM_HARD, FastLabeller, LabelPool,
)

# Concept-target building is pure-Python rule evaluation (32k classifications
# per update at num_envs=1024), so it runs on worker processes
# (ppo_concept_labels.LabelPool: each keeps a cache of per-auction rule plans and
# gets the decisions whose auction hashes to it). They are forked HERE, before
# the imports below: importing the network/env code initializes JAX's GPU
# backend, and forking after that copies a live CUDA context + XLA thread
# pools (JAX warns this can deadlock). Only when run as the entry point, so
# importing this module elsewhere never spawns processes.
_CONCEPT_POOL = None
if __name__ == "__main__":
    _CONCEPT_POOL = LabelPool(os.cpu_count() or 1)

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf
from pydantic import BaseModel
from pgx.bridge_bidding import BridgeBidding, download_dds_results

from training.concept_bottleneck import NUM_ACTIONS, OBS_SIZE, build_net
from src.gae import make_calc_gae
from src.roll_out import make_roll_out
from src.update import make_update_step
from training.ppo_anchor import drift_metrics, make_anchored_update_step

MIN_ACTION = 52


class PPOForward:
    """Strips concept_bottleneck.build_net's dict output down to the
    (policy_logits, value) tuple src/roll_out.py, src/gae.py, and
    src/update.py expect."""

    def __init__(self, net):
        self._net = net

    def init(self, rng, obs):
        return self._net.init(rng, obs)

    def apply(self, params, obs):
        out = self._net.apply(params, obs)
        return out["policy_logits"], out["value"]


class PPOConfig(BaseModel):
    seed: int = 0
    lr: float = 1e-6
    num_envs: int = 1024
    num_steps: int = 32
    total_timesteps: int = 2_621_440_000
    update_epochs: int = 10
    minibatch_size: int = 1024
    num_eval_envs: int = 2000

    save_model: bool = True
    save_model_interval: int = 1
    log_path: str = "rl_log"
    exp_name: str = "ppo"

    bottleneck_mode: Literal["joint", "independent", "sequential"] = "joint"
    load_initial_model: bool = False
    initial_model_path: str | None = None
    actor_activation: str = "relu"
    actor_model_type: Literal["DeepMind", "FAIR"] = "DeepMind"

    # "competitive" = all four seats bid (learner, both opponents and partner, all with
    # the current parameters); "free-run" = the opponents always pass (uncontested
    # bidding).
    game_mode: Literal["competitive", "free-run"] = "competitive"

    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    value_clipping: bool = True
    global_gradient_clipping: bool = True
    anneal_lr: bool = False
    reward_scaling: bool = False
    max_grad_norm: float = 0.5
    reward_scale: float = 7600
    illegal_action_l2norm_coef: float = 0

    actor_illegal_action_mask: bool = True
    actor_illegal_action_penalty: bool = False

    # Concept supervision during PPO: a separate optimizer step, every
    # `concept_update_freq` updates, that labels the rollout's (obs, action)
    # pairs with the rule engine and takes one BCE step on hard_state_head
    # (+ concept_only_policy_head's NLL). The PPO loss never reaches the
    # concept head (concept_acts is stop-gradiented in every mode), so this
    # is the only way it changes. 0 keeps the SL concept model fixed.
    concept_coef: float = 0.5
    concept_update_freq: int = 1

    # KL anchor to the SL policy (training/ppo_anchor.py). 0 = plain PPO.
    # sl_anchor_path defaults to initial_model_path; set it explicitly when
    # resuming from a PPO checkpoint, or the "anchor" would be that checkpoint.
    sl_anchor_coef: float = 0.0
    sl_anchor_path: str | None = None


def export_params(params, bottleneck_mode: str):
    """Same export tie as training/sl.py's _export_params: in "independent"/
    "sequential" mode policy_head and concept_only_policy_head take the
    identical input, and explain/ + nlg/explanation_frames.py read concept_only_policy_head
    as the concepts->bid map, so it must BE the policy. Without this, PPO
    trains policy_head while concept_only_policy_head only follows a small
    auxiliary NLL, and the two drift apart."""
    if bottleneck_mode == "joint":
        return params
    return {**params, "concept_only_policy_head": params["policy_head"]}


def save_checkpoint(path: str, params, cfg: "PPOConfig") -> None:
    """The checkpoint dict training.checkpoint.load_checkpoint reads, plus the
    PPO settings that decide what the run learned (game_mode above all).
    Written to a temp file and renamed, because runs are stopped by killing
    them (pipeline/converge.py) and the newest checkpoint is then trusted."""
    blob = {"params": export_params(params, cfg.bottleneck_mode), "hard_concepts": HARD_STATE,
            "bottleneck_mode": cfg.bottleneck_mode, "model_type": cfg.actor_model_type,
            "activation": cfg.actor_activation, "game_mode": cfg.game_mode,
            "ppo": {"lr": cfg.lr, "sl_anchor_coef": cfg.sl_anchor_coef, "concept_coef": cfg.concept_coef,
                    "num_envs": cfg.num_envs, "num_steps": cfg.num_steps}}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blob, f)
    os.replace(tmp, path)


def make_concept_loss_fn(net, concept_only_coef: float):
    @jax.jit
    def concept_loss_and_grad(params, obs, hard_targets, action, legal, concept_coef):
        def loss_fn(p):
            out = net.apply(p, obs)
            concept_logits, concept_only_logits = out["concept_logits"], out["concept_only_logits"]

            bce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(concept_logits, hard_targets))

            co_masked = jnp.where(legal, concept_only_logits, jnp.finfo(jnp.float32).min)
            co_logp = jax.nn.log_softmax(co_masked)
            concept_only_nll = -jnp.mean(jnp.take_along_axis(co_logp, action[:, None], axis=1))
            concept_only_acc = jnp.mean(jnp.argmax(co_masked, axis=1) == action)

            total = bce_loss + concept_only_coef * concept_only_nll
            return total, {"bce_loss": bce_loss, "concept_only_acc": concept_only_acc}

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return concept_coef * loss, grads, aux

    return concept_loss_and_grad


def make_concept_step(net, optimizer, concept_only_coef: float):
    """The whole concept step as one compiled call: loss, gradient, the optimizer update and applying it. Takes
    the rollout as it sits on the device, (num_steps, num_envs, ...), so the observations are not copied to the
    host and back, and the labels as uint8."""
    loss_and_grad = make_concept_loss_fn(net, concept_only_coef)

    @jax.jit
    def concept_step(params, opt_state, obs, targets_u8, action, legal, concept_coef):
        _, grads, aux = loss_and_grad(params, obs.reshape(-1, obs.shape[-1]), targets_u8.astype(jnp.float32),
                                      action.reshape(-1), legal.reshape(-1, legal.shape[-1]), concept_coef)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, aux

    return concept_step


@jax.jit
def pack_obs_jax(obs):
    """ppo_concept_labels.pack_obs on the device: the labeller's input is 61 bytes a decision instead of 480 floats."""
    flat = obs.reshape(-1, obs.shape[-1])
    return (jnp.packbits(flat[:, AUCTION_BITS] > 0.5, axis=1), jnp.packbits(flat[:, HAND_BITS] > 0, axis=1))


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def train(cfg: PPOConfig, pool=None):
    """`pool`: optional ppo_concept_labels.LabelPool (see _CONCEPT_POOL) for concept
    targets; None labels in this process -- identical results, just slower."""
    config = cfg.model_dump()
    config["num_updates"] = config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    config["num_minibatches"] = config["num_envs"] * config["num_steps"] // config["minibatch_size"]

    if not os.path.isdir("dds_results"):
        download_dds_results()

    env = BridgeBidding()
    net = build_net(cfg.bottleneck_mode, cfg.actor_model_type, cfg.actor_activation, NUM_HARD, has_value=True)
    actor_forward = PPOForward(net)
    # concept_only_policy_head is a real diagnostic head only in "joint" mode;
    # elsewhere it is exported as policy_head (export_params), so training it
    # is wasted work -- same rule as training/sl.py.
    concept_only_coef = 0.1 if cfg.bottleneck_mode == "joint" else 0.0
    if cfg.bottleneck_mode == "sequential" and cfg.concept_coef > 0:
        # "sequential" means the concept model is frozen once trained (SL
        # phase A); PPO's own loss cannot reach it (concept_acts is
        # stop-gradiented), but the auxiliary concept step would keep
        # retraining the trunk and hard_state_head under the policy.
        print("[init] sequential mode: concept_coef forced to 0 (concept model stays frozen)")
        cfg = cfg.model_copy(update={"concept_coef": 0.0})

    _lr = cfg.lr
    optimizer = (
        optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(_lr, eps=1e-5))
        if cfg.global_gradient_clipping else optax.adam(_lr, eps=1e-5)
    )
    concept_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr, eps=1e-5))
    concept_step = make_concept_step(net, concept_optimizer, concept_only_coef)
    labeller = pool if pool is not None else FastLabeller()

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    params = actor_forward.init(init_rng, jnp.zeros((1, OBS_SIZE)))
    opt_state = optimizer.init(params)
    concept_opt_state = concept_optimizer.init(params)

    if cfg.load_initial_model and cfg.initial_model_path:
        print(f"[init] warm-starting from {cfg.initial_model_path}")
        with open(cfg.initial_model_path, "rb") as f:
            ckpt = pickle.load(f)
        sl_params = ckpt["params"] if (isinstance(ckpt, dict) and "params" in ckpt) else ckpt
        params = hk.data_structures.merge(params, sl_params)
        print("[init] trunk + policy_head + hard_state_head (+ concept_only_policy_head "
              "if present) loaded; value_head stays freshly initialized")

    # frozen SL reference: the anchor target and the drift baseline
    anchor_path = cfg.sl_anchor_path or (cfg.initial_model_path if cfg.sl_anchor_coef > 0 else None)
    ref_logits_fn = drift_fn = None
    if anchor_path:
        from training.checkpoint import load_checkpoint
        ref = load_checkpoint(anchor_path)
        print(f"[init] SL reference for anchor/drift: {anchor_path} (anchor coef {cfg.sl_anchor_coef})")
        ref_logits_fn = jax.jit(lambda o: ref.net.apply(ref.params, o.astype(jnp.float32))["policy_logits"])
        drift_fn = jax.jit(lambda p, o, legal, rl: drift_metrics(
            actor_forward.apply(p, o.astype(jnp.float32))[0], rl, legal))
    # All three MUST be jitted (as the vendored ppo.py does). Unjitted,
    # roll_out rebuilds its lax.scan body closure every call with the
    # current params captured as constants, so each update compiles and
    # caches a new executable and host memory grows ~130 MB per update.
    if cfg.sl_anchor_coef > 0:
        update_step = jax.jit(make_anchored_update_step(config, actor_forward, optimizer, cfg.sl_anchor_coef))
    else:
        update_step = jax.jit(make_update_step(config, actor_forward, optimizer))
    roll_out = jax.jit(make_roll_out(config, env, actor_forward, actor_forward))  # self-play: opp == actor
    calc_gae = jax.jit(make_calc_gae(config, actor_forward))

    rng, env_rng = jax.random.split(rng)
    env_state = jax.vmap(env.init)(jax.random.split(env_rng, cfg.num_envs))
    runner_state = (params, opt_state, env_state, env_state.observation, 0, rng)

    save_dir = os.path.join(cfg.log_path, cfg.exp_name, "rl_params")
    if cfg.save_model:
        os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.log_path, cfg.exp_name, "metrics.jsonl")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(os.path.join(cfg.log_path, cfg.exp_name, "launches.jsonl"), "a") as lf:
        lf.write(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), **cfg.model_dump()}) + "\n")
    print(f"=== START PPO TRAINING (bottleneck_mode={cfg.bottleneck_mode}, game_mode={cfg.game_mode}) ===")
    print(f"    {config['num_updates']} updates planned, concept supervision "
          f"coef={cfg.concept_coef} every {cfg.concept_update_freq} updates")

    for i in range(config["num_updates"]):
        t0 = time.time()

        runner_state, traj_batch = roll_out(runner_state, runner_state[0])
        do_concept = cfg.concept_coef > 0 and i % cfg.concept_update_freq == 0
        if do_concept:
            # The labels depend on the rollout only, not on the parameters, so the CPU workers label it while
            # the GPU runs the PPO update below; collect() waits for them only after that is queued.
            auction_bits, hand_bits = pack_obs_jax(traj_batch.obs)
            labeller.submit(np.asarray(auction_bits), np.asarray(hand_bits), np.asarray(traj_batch.action).reshape(-1))
        advantages, targets = calc_gae(runner_state, traj_batch)
        ref_logits = None
        if ref_logits_fn is not None and (cfg.sl_anchor_coef > 0 or i % 10 == 0):
            flat = traj_batch.obs.reshape(-1, traj_batch.obs.shape[-1])
            ref_logits = ref_logits_fn(flat).reshape(traj_batch.obs.shape[:2] + (NUM_ACTIONS,))
        if cfg.sl_anchor_coef > 0:
            runner_state, loss_info = update_step(runner_state, traj_batch, advantages, targets, ref_logits)
        else:
            runner_state, loss_info = update_step(runner_state, traj_batch, advantages, targets)

        if do_concept:
            t_wait = time.time()
            all_targets = labeller.collect()  # (num_steps * num_envs, NUM_HARD) uint8, in the rollout's flat order
            label_wait = time.time() - t_wait
            new_params, concept_opt_state, concept_aux = concept_step(
                runner_state[0], concept_opt_state, traj_batch.obs, all_targets, traj_batch.action,
                traj_batch.legal_action_mask, cfg.concept_coef,
            )
            runner_state = (new_params,) + runner_state[1:]
        else:
            concept_aux = None

        if i % 10 == 0:
            msg = f"[update {i:06d}] {time.time()-t0:.1f}s"
            row = {"update": i, "time": time.time()}
            if concept_aux is not None:
                msg += (f"  concept_bce={float(concept_aux['bce_loss']):.3f}"
                        f"  concept_only_acc={float(concept_aux['concept_only_acc'])*100:5.1f}%")
                st = labeller.stats
                hit = st["plan_hits"] / max(1, st["plan_hits"] + st["plan_misses"])
                msg += f"  label_wait={label_wait:.2f}s  plan_hits={hit*100:4.1f}%"
                row.update(concept_bce=float(concept_aux["bce_loss"]),
                           concept_only_acc=float(concept_aux["concept_only_acc"]),
                           label_wait_s=label_wait, plan_hit_rate=hit, plans=st["plans"])
            if drift_fn is not None:
                flat_obs = traj_batch.obs.reshape(-1, traj_batch.obs.shape[-1])
                flat_legal = traj_batch.legal_action_mask.reshape(-1, NUM_ACTIONS)
                d = drift_fn(runner_state[0], flat_obs, flat_legal, ref_logits.reshape(-1, NUM_ACTIONS))
                msg += f"  kl_to_sl={float(d['kl_to_sl']):.4f}  sl_agree={float(d['sl_agreement'])*100:5.1f}%"
                row.update(kl_to_sl=float(d["kl_to_sl"]), sl_agreement=float(d["sl_agreement"]))
            print(msg, flush=True)
            with open(metrics_path, "a") as mf:
                mf.write(json.dumps(row) + "\n")

        if cfg.save_model and i != 0 and i % cfg.save_model_interval == 0:
            save_checkpoint(os.path.join(save_dir, f"params-{i:08d}.pkl"), runner_state[0], cfg)

    if cfg.save_model:
        save_checkpoint(os.path.join(save_dir, "params-final.pkl"), runner_state[0], cfg)
        print(f"[ppo] saved final checkpoint to {save_dir}/params-final.pkl")


if __name__ == "__main__":
    cfg = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
    try:
        train(cfg, _CONCEPT_POOL)
    finally:
        _CONCEPT_POOL.close()
