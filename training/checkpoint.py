"""`load_checkpoint`: the only way downstream code loads a training/sl.py or training/ppo.py checkpoint.
It reads the mode, trunk type, activation and concept list from the checkpoint itself and builds the net with the
trainers' own `build_net`, so the architecture is defined in exactly one place."""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import jax.numpy as jnp

from training.concept_bottleneck import OBS_SIZE, build_net


@dataclass(frozen=True)
class LoadedCheckpoint:
    net: object          # haiku-transformed net; net.apply(params, obs_batch) -> dict, see build_net's docstring
    params: object        # haiku params pytree
    hard_concepts: list    # concept names, aligned to hard_logits/concept_acts's last axis
    bottleneck_mode: str
    model_type: str
    activation: str
    has_value: bool


def load_checkpoint(path: str) -> LoadedCheckpoint:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    if not (isinstance(raw, dict) and "params" in raw):
        raise ValueError(
            f"{path} doesn't look like a training/sl.py or training/ppo.py "
            f"checkpoint (expected a dict with a 'params' key, got "
            f"{type(raw)}). This loader only reads checkpoints this repo's "
            f"own trainers produced."
        )

    params = raw["params"]
    hard_concepts = raw.get("hard_concepts")
    if hard_concepts is None:
        from concepts.vocabulary import HARD_STATE_CONCEPTS
        num_hard = params["hard_state_head"]["b"].shape[0]
        hard_concepts = list(HARD_STATE_CONCEPTS)[:num_hard]

    bottleneck_mode = raw.get("bottleneck_mode", "joint")
    model_type = raw.get("model_type")
    activation = raw.get("activation", "relu")
    has_value = "value_head" in params

    if model_type is None:
        # Infer from the trunk's first layer width (DeepMind 1024, FAIR 200);
        # only for checkpoints saved without the field (the trainers always
        # write it).
        model_type = "DeepMind" if params["linear"]["w"].shape[1] == 1024 else "FAIR"

    net = build_net(bottleneck_mode, model_type, activation, len(hard_concepts), has_value)

    # No fallback: a shape mismatch means the checkpoint and build_net
    # disagree about the architecture, and must fail here.
    dummy = jnp.zeros((1, OBS_SIZE))
    net.apply(params, dummy)

    return LoadedCheckpoint(
        net=net, params=params, hard_concepts=hard_concepts,
        bottleneck_mode=bottleneck_mode, model_type=model_type,
        activation=activation, has_value=has_value,
    )
