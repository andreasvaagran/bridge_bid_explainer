"""The trust flag: does the concepts-only head pick the same call as the policy? In joint mode this compares
two heads; in the bottleneck modes the two are tied, so they always agree."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from explain.counterfactual import softmax_masked


@dataclass(frozen=True)
class TrustFlag:
    agree: bool
    full_policy_bid: str
    full_policy_conf: float
    concept_only_bid: str
    concept_only_conf: float
    bottleneck_mode: str

    def explanation(self) -> str:
        if self.agree:
            return (f"AGREE — concept-only head independently picks the same bid "
                    f"({self.concept_only_bid}, p={self.concept_only_conf:.2f}); the concept "
                    f"vocabulary alone is sufficient to explain this decision.")
        if self.bottleneck_mode == "joint":
            return (f"DISAGREE — concept-only head would pick {self.concept_only_bid} "
                    f"(p={self.concept_only_conf:.2f}) instead of {self.full_policy_bid}; the full "
                    f"policy is using information beyond the concept vocabulary here, so treat "
                    f"the concept-based explanation with more caution than usual.")
        return (f"DISAGREE — but in {self.bottleneck_mode} mode, policy_head and "
                f"concept_only_policy_head see the identical input (concepts only); this "
                f"disagreement means the two separately-trained heads didn't converge to the "
                f"same function, not that the real policy used non-concept information "
                f"(there isn't any to use in this mode).")


def compute_trust_flag(policy_logits: np.ndarray, concept_only_logits: np.ndarray,
                        legal: np.ndarray, bids: list, bottleneck_mode: str) -> TrustFlag:
    full_probs = softmax_masked(policy_logits, legal)
    co_probs = softmax_masked(concept_only_logits, legal)
    full_action = int(np.argmax(full_probs))
    co_action = int(np.argmax(co_probs))
    return TrustFlag(
        agree=(full_action == co_action),
        full_policy_bid=bids[full_action], full_policy_conf=float(full_probs[full_action]),
        concept_only_bid=bids[co_action], concept_only_conf=float(co_probs[co_action]),
        bottleneck_mode=bottleneck_mode,
    )
