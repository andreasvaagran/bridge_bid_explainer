"""The network every trainer builds (`build_net`): brl's trunk, a linear + sigmoid concept head over
the 146 concepts, and a policy head whose input depends on `bottleneck_mode`:

  joint        policy sees [hidden, stop_gradient(concepts)]: concepts inform but do not gate the policy.
  independent  policy sees stop_gradient(concepts) only: a true bottleneck. Policy gradients cannot reach the
               trunk or the concept head.
  sequential   the independent architecture trained in two phases: the concept head first (policy loss weight
               0), then the policy on the frozen concept predictions.

Only "sequential" matches Koh et al. (2020) exactly; the names are kept because every checkpoint stores them."""
from __future__ import annotations

from typing import Dict, Literal

import haiku as hk
import jax
import jax.numpy as jnp

NUM_ACTIONS = 38
OBS_SIZE = 480

BottleneckMode = Literal["joint", "independent", "sequential"]


def _fair_trunk(obs, act):
    input_ = obs
    x = hk.Linear(200)(obs)
    shortcut_1 = x
    x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    x = x + shortcut_1
    shortcut_2 = x
    x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    x = x + shortcut_2
    x = hk.Linear(200)(x)
    x = jnp.concatenate([x, input_], axis=-1)
    x = hk.Linear(200)(x)
    shortcut_3 = x
    x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    x = x + shortcut_3
    shortcut_4 = x
    x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    x = hk.Linear(200)(x); x = act(x)
    return x + shortcut_4


def _deepmind_trunk(obs, act):
    x = hk.Linear(1024)(obs); x = act(x)
    x = hk.Linear(1024)(x);   x = act(x)
    x = hk.Linear(1024)(x);   x = act(x)
    x = hk.Linear(1024)(x);   x = act(x)
    return x


def build_net(
    bottleneck_mode: BottleneckMode,
    model_type: str,
    activation: str,
    num_hard: int,
    has_value: bool = False,
):
    """Returns a haiku-transformed net whose `.apply(params, obs)` returns a
    dict with `policy_logits`, `concept_logits`, `concept_acts`, `hidden`,
    `concept_only_logits`, and `value` (if `has_value`) in every
    bottleneck_mode; explain/, nlg/ and evaluation/ rely on this. The module
    names (`hard_state_head`, `policy_head`, `concept_only_policy_head`,
    `value_head`) are the same in all modes, so checkpoints stay
    structurally comparable even though what feeds `policy_head` differs.
    """

    def net_fn(obs):
        act = jax.nn.relu if activation == "relu" else jax.nn.tanh
        hidden = _deepmind_trunk(obs, act) if model_type == "DeepMind" else _fair_trunk(obs, act)

        concept_logits = hk.Linear(num_hard, name="hard_state_head")(hidden)
        concept_acts = jax.nn.sigmoid(concept_logits)
        stopped = jax.lax.stop_gradient(concept_acts)

        if bottleneck_mode == "joint":
            # Soft bottleneck: the policy sees the raw trunk representation
            # too, so policy-loss gradients keep shaping `hidden` even
            # though they can never reach the concept head itself.
            policy_input = jnp.concatenate([hidden, stopped], axis=-1)
        else:
            # "independent"/"sequential": true bottleneck -- every path
            # from observation to bid passes through the concept vector.
            policy_input = stopped

        policy_logits = hk.Linear(NUM_ACTIONS, name="policy_head")(policy_input)

        # Present in every mode. In "joint" mode it is a diagnostic: a second
        # head on the concepts alone, never gradient-connected to `hidden`,
        # measuring concept sufficiency. In "independent"/"sequential" mode
        # it duplicates policy_head's input (separately initialized), so
        # downstream tooling (trust flag, concept-flip counterfactual) needs
        # no special case.
        concept_only_logits = hk.Linear(NUM_ACTIONS, name="concept_only_policy_head")(stopped)

        out: Dict[str, jnp.ndarray] = {
            "policy_logits": policy_logits,
            "concept_logits": concept_logits,
            "concept_acts": concept_acts,
            "hidden": hidden,
            "concept_only_logits": concept_only_logits,
        }
        if has_value:
            value_input = jnp.concatenate([hidden, stopped], axis=-1) if bottleneck_mode == "joint" else stopped
            out["value"] = hk.Linear(1, name="value_head")(value_input).squeeze(-1)
        return out

    return hk.without_apply_rng(hk.transform(net_fn))


def trunk_and_concept_param_mask(params) -> "jax.tree_util.PyTreeDef":
    """For "sequential" mode's training-phase B: a pytree matching `params`
    exactly, with 0.0 at every leaf belonging to the trunk or
    `hard_state_head` (frozen once phase B starts) and 1.0 everywhere else
    (`policy_head`/`concept_only_policy_head`/`value_head`, still training).
    training/sl.py multiplies the optimizer's updates by it."""
    trainable_prefixes = ("policy_head", "concept_only_policy_head", "value_head")

    def leaf_mask(module_name: str, name: str, value: jnp.ndarray) -> jnp.ndarray:
        is_trainable = any(module_name.startswith(p) for p in trainable_prefixes)
        return jnp.ones_like(value) if is_trainable else jnp.zeros_like(value)

    return hk.data_structures.map(leaf_mask, params)
