"""Integrated Gradients over the 480-bit observation for the chosen call: a continuous attribution per
card, complementing the discrete card swaps of counterfactual.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from concepts.hand import RANKS, SUIT_ORDER


@dataclass(frozen=True)
class CardAttribution:
    card: str
    attribution: float  # signed: positive = pushed toward the chosen bid, negative = pushed away


def _integrated_gradients_full(net, params, obs: np.ndarray, action: int, m_steps: int) -> np.ndarray:
    """The full (OBS_SIZE,) IG vector over every observation feature, not just the 52 card bits: the
    non-card features (e.g. vulnerability) carry attribution too, and the completeness axiom needs them."""
    obs_j = jnp.asarray(obs, dtype=jnp.float32)
    baseline = jnp.zeros_like(obs_j)

    def target_logit(o):
        out = net.apply(params, o[None, :])
        return out["policy_logits"][0, action]

    grad_fn = jax.grad(target_logit)
    alphas = jnp.linspace(0.0, 1.0, m_steps + 1)[1:]  # m_steps terms, excluding alpha=0
    interpolated = baseline[None, :] + alphas[:, None] * (obs_j - baseline)[None, :]
    grads = jax.vmap(grad_fn)(interpolated)  # (m_steps, OBS_SIZE)
    avg_grad = jnp.mean(grads, axis=0)
    return np.asarray((obs_j - baseline) * avg_grad)


def integrated_gradients_cards(
    net, params, obs: np.ndarray, action: int, m_steps: int = 50,
) -> List[CardAttribution]:
    """IG attribution per held card, largest |attribution| first.

    `net`/`params` as returned by training.checkpoint.load_checkpoint (net.apply(params, obs_batch) ->
    dict with "policy_logits"); `obs` is one (OBS_SIZE,) observation, `action` the chosen bid's index.
    The baseline is the all-zero observation ("no information"), the usual choice for a multi-hot input."""
    ig_full = _integrated_gradients_full(net, params, obs, action, m_steps)
    card_bits = ig_full[-52:]
    hand_bits = np.asarray(obs)[-52:]

    from concepts.hand import card_bit_index
    all_cards = [s + r for s in SUIT_ORDER for r in RANKS]
    index_to_card = {card_bit_index(c): c for c in all_cards}

    results = []
    for i in range(52):
        if hand_bits[i] > 0:
            results.append(CardAttribution(card=index_to_card[i], attribution=float(card_bits[i])))
    results.sort(key=lambda c: -abs(c.attribution))
    return results


