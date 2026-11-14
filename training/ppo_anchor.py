"""The PPO update step with a KL anchor to the frozen SL policy:

    loss = PPO loss + anchor_coef * KL(pi_theta(.|s) || pi_SL(.|s))   (over the legal calls)

The anchor keeps PPO close to the WBridge5 system the rules and explanations describe. It is a line-by-line port
of the vendored src/update.make_update_step, which has no hook for an extra loss term; with anchor_coef = 0 it
gives the same parameters (tests/test_ppo_anchor.py). `drift_metrics` gives KL to SL and the same-call rate."""
from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _masked_log_softmax(logits: jnp.ndarray, legal: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)


def kl_to_reference(logits: jnp.ndarray, ref_logits: jnp.ndarray, legal: jnp.ndarray) -> jnp.ndarray:
    """Per-state KL(pi || pi_ref) over the legal calls."""
    logp = _masked_log_softmax(logits, legal)
    logq = _masked_log_softmax(ref_logits, legal)
    return jnp.sum(jnp.where(legal, jnp.exp(logp) * (logp - logq), 0.0), axis=-1)


def drift_metrics(logits: jnp.ndarray, ref_logits: jnp.ndarray, legal: jnp.ndarray) -> dict:
    """Mean KL(pi || pi_SL) and argmax agreement with the SL policy."""
    legal = legal.astype(bool)
    agree = jnp.argmax(jnp.where(legal, logits, -1e9), -1) == jnp.argmax(jnp.where(legal, ref_logits, -1e9), -1)
    return {"kl_to_sl": kl_to_reference(logits, ref_logits, legal).mean(), "sl_agreement": agree.mean()}


def make_anchored_update_step(config, actor_forward_pass, optimizer, anchor_coef: float):
    """src/update.make_update_step + the anchor term. The returned
    update_step(runner_state, traj_batch, advantages, targets, ref_logits)
    takes ref_logits shaped like the rollout, (num_steps, num_envs, 38)."""

    def policy(mask, logits):
        if config["actor_illegal_action_mask"]:
            return distrax.Categorical(logits=logits + jnp.finfo(np.float64).min * (~mask))
        return distrax.Categorical(logits=logits)

    def reward_scaling(gae):
        return (gae - gae.mean()) / (gae.std() + 1e-8) if config["reward_scaling"] else gae

    def value_loss_fn(value, traj_batch, targets):
        if config["value_clipping"]:
            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                -config["clip_eps"], config["clip_eps"])
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        return 0.5 * jnp.square(value - targets).mean()

    def update_step(runner_state, traj_batch, advantages, targets, ref_logits):
        params, opt_state, env_state, last_obs, terminated_count, rng = runner_state

        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets, ref_logits = batch_info

                def _loss_fn(params, traj_batch, gae, targets, ref_logits):
                    logits, value = actor_forward_pass.apply(params, traj_batch.obs.astype(jnp.float32))
                    mask = traj_batch.legal_action_mask
                    pi = policy(mask, logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    value_loss = value_loss_fn(value, traj_batch, targets)

                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = reward_scaling(gae)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    masked_pi = distrax.Categorical(logits=logits + jnp.finfo(np.float64).min * (~mask))
                    entropy = masked_pi.entropy().mean()

                    illegal_action_probabilities = distrax.Categorical(logits=logits).probs * ~mask
                    illegal_action_loss = jnp.linalg.norm(illegal_action_probabilities, ord=2) / 2

                    anchor_kl = kl_to_reference(logits, ref_logits, mask.astype(bool)).mean()

                    total_loss = (loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                                  + config["illegal_action_l2norm_coef"] * illegal_action_loss
                                  + anchor_coef * anchor_kl)
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = jnp.float32(jnp.abs(ratio - 1.0) > config["clip_eps"]).mean()
                    return total_loss, (value_loss, loss_actor, entropy, approx_kl, clipfracs,
                                        illegal_action_loss, anchor_kl)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(params, traj_batch, advantages, targets, ref_logits)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), total_loss

            params, opt_state, traj_batch, advantages, targets, ref_logits, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["minibatch_size"] * config["num_minibatches"]
            assert batch_size == config["num_steps"] * config["num_envs"], \
                "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets, ref_logits)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config["num_minibatches"], -1] + list(x.shape[1:])), shuffled)
            (params, opt_state), total_loss = jax.lax.scan(_update_minbatch, (params, opt_state), minibatches)
            return (params, opt_state, traj_batch, advantages, targets, ref_logits, rng), total_loss

        update_state = (params, opt_state, traj_batch, advantages, targets, ref_logits, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
        params, opt_state, _, _, _, _, rng = update_state
        return (params, opt_state, env_state, last_obs, terminated_count, rng), loss_info

    return update_step
