"""Builds explanation frames (the dict the teacher, the prompt and the checks read) from a checkpoint's own
decisions. `hand_str` is always letter-coded ("S:AKT H:J9 ..."), never suit symbols."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding

from concepts import AuctionContext, HandFeatures, decode_hand, create_sayc_state_authority
from explain.counterfactual import card_counterfactuals, flip_shortlist_caveat, search_minimal_flip, softmax_masked
from explain.rule_justification import render_concept_justification
from explain.trust_flag import compute_trust_flag
from training.checkpoint import load_checkpoint

BIDS = ["P", "X", "XX", "1C", "1D", "1H", "1S", "1NT", "2C", "2D", "2H", "2S", "2NT",
        "3C", "3D", "3H", "3S", "3NT", "4C", "4D", "4H", "4S", "4NT", "5C", "5D", "5H",
        "5S", "5NT", "6C", "6D", "6H", "6S", "6NT", "7C", "7D", "7H", "7S", "7NT"]
SEAT2CHAR = {0: "N", 1: "E", 2: "S", 3: "W"}


def hand_str_letter_coded(hand: HandFeatures) -> str:
    order = "23456789TJQKA"
    parts = []
    for s in ("S", "H", "D", "C"):
        ranks = sorted(hand.cards[s], key=lambda r: order.index(r), reverse=True)
        parts.append(f"{s}:{''.join(ranks) if ranks else '-'}")
    return " ".join(parts)


def build_frame(deal_id: int, step: int, seat: str, partner: str, opponents: List[str],
                 dealer: str, vulnerability: str, auction_before: List[Dict],
                 bid: str, confidence: float, hand: HandFeatures,
                 top_concepts: List[tuple],
                 symbolic_justification: Optional[str] = None,
                 trust: Optional[Dict] = None,
                 counterfactual_cards: Optional[List[Dict]] = None,
                 concept_flip: Optional[Dict] = None) -> Dict:
    frame = {
        "deal_id": deal_id, "step": step,
        "bid": bid, "confidence": round(float(confidence), 2),
        "table": {"you": seat, "partner": partner, "opponents": opponents},
        "auction_before": list(auction_before),
        "dealer": dealer, "vulnerability": vulnerability,
        "hcp": hand.hcp, "hand_str": hand_str_letter_coded(hand), "balanced": hand.balanced,
        "top_concepts": [[n, round(float(p), 3)] for n, p in top_concepts],
    }
    # enrichment fields (symbolic justification, trust flag, card and concept-flip counterfactuals) are
    # omitted, not written as null, when not computed, so a non-enriched frame keeps the bare schema
    if symbolic_justification is not None:
        frame["symbolic_justification"] = symbolic_justification
    if trust is not None:
        frame["trust"] = trust
    if counterfactual_cards is not None:
        frame["counterfactual_cards"] = counterfactual_cards
    if concept_flip is not None:
        frame["concept_flip"] = concept_flip
    return frame


class Enricher:
    """Per-checkpoint pieces every enriched frame needs (rule engine, the
    linear concepts->bid head, the batched policy for card counterfactuals).
    Shared by generate_frames (pgx self-play) and
    nlg/build_teacher_dataset.py (WBridge5 positions)."""

    def __init__(self, ck):
        self.ck = ck
        self.state_auth = create_sayc_state_authority()
        self.has_concept_only = "concept_only_policy_head" in ck.params
        self.W = np.asarray(ck.params["concept_only_policy_head"]["w"]) if self.has_concept_only else None
        self.b = np.asarray(ck.params["concept_only_policy_head"]["b"]) if self.has_concept_only else None
        # pure call descriptors, never offered as a counterfactual "reason"
        self.descriptor_idx = [i for i, c in enumerate(ck.hard_concepts) if c.startswith("bid_is_in_")]
        self._policy = jax.jit(lambda o: ck.net.apply(ck.params, o)["policy_logits"])

    def policy_logits_fn(self, obs_batch):
        return np.asarray(self._policy(np.asarray(obs_batch, np.float32)))


def decision_frame(en: "Enricher", obs, legal, out: Dict[str, np.ndarray], deal_id: int, step: int, seat: str,
                   dealer: str, dealer_int: int, vulnerability: str, auction_before: List[Dict],
                   top_k: int = 8, concept_threshold: float = 0.05, enrich: bool = True,
                   counterfactuals: bool = True) -> Dict:
    """One decision's frame from the model outputs `out` (policy_logits,
    concept_acts, concept_only_logits for this single observation). The
    bid is the model's own argmax over legal calls."""
    ck = en.ck
    policy_logits, concept_acts = out["policy_logits"], out["concept_acts"]
    probs = softmax_masked(policy_logits, legal)
    action = int(np.argmax(probs))
    bid = BIDS[action]
    hand = decode_hand(obs)
    seat_idx = list(SEAT2CHAR.values()).index(seat)
    partner = SEAT2CHAR[(seat_idx + 2) % 4]
    opponents = [SEAT2CHAR[(seat_idx + 1) % 4], SEAT2CHAR[(seat_idx + 3) % 4]]
    order = np.argsort(-concept_acts)
    top_concepts = [(ck.hard_concepts[i], concept_acts[i]) for i in order[:top_k]
                    if concept_acts[i] >= concept_threshold]

    symbolic_justification = trust = counterfactual_cards = concept_flip = None
    concept_evidence = runner_up = None
    if enrich:
        auction_ctx = AuctionContext(bids=[a["bid"] for a in auction_before], bid=bid, dealer=dealer_int)
        gt = en.state_auth.classify(auction_ctx, hand)
        # Prefer justifying the model's own top predicted concept that's also
        # ground-truth-true; fall back to the first ground-truth-true concept.
        for name in [n for n, _p in top_concepts] + list(ck.hard_concepts):
            if gt.get(name):
                try:
                    symbolic_justification = render_concept_justification(name, auction_ctx, hand)
                    break
                except ValueError:
                    continue
        if en.has_concept_only:
            concept_only_logits = out["concept_only_logits"]
            tf = compute_trust_flag(policy_logits, concept_only_logits, legal, BIDS, ck.bottleneck_mode)
            trust = {"agree": tf.agree, "full_policy_bid": tf.full_policy_bid,
                     "full_policy_conf": round(tf.full_policy_conf, 3), "concept_only_bid": tf.concept_only_bid,
                     "concept_only_conf": round(tf.concept_only_conf, 3), "bottleneck_mode": tf.bottleneck_mode,
                     "explanation": tf.explanation()}
            if counterfactuals:
                counterfactual_cards = card_counterfactuals(obs, hand, legal, action, probs, en.policy_logits_fn,
                                                            bid, BIDS)
            W, b = en.W, en.b
            necessary = search_minimal_flip(concept_acts, legal, W, b, flip_active=True, k_max=3,
                                            exclude=en.descriptor_idx)
            sufficient = search_minimal_flip(concept_acts, legal, W, b, flip_active=False, k_max=3,
                                             exclude=en.descriptor_idx)
            nec_names = [ck.hard_concepts[i] for i in necessary] if necessary else None
            suf_names = [ck.hard_concepts[i] for i in sufficient] if sufficient else None
            concept_flip = {"necessary": nec_names, "necessary_caveat": flip_shortlist_caveat(nec_names) or None,
                            "sufficient": suf_names, "sufficient_caveat": flip_shortlist_caveat(suf_names) or None}
            # Exact decomposition of the linear concepts->bid head's margin for
            # the chosen bid over its runner-up (bias aside):
            # margin = sum_i a_i * (W[i,bid] - W[i,runner]). In independent/
            # sequential mode that head IS the policy; in joint mode it's the
            # concept-only head (the concept account of the bid). nlg/explanation_inputs.py
            # ranks concepts by this instead of by activation.
            co_masked = np.where(legal, concept_only_logits, -np.inf)
            co_masked[action] = -np.inf
            runner = int(np.argmax(co_masked))
            contrib = concept_acts * (W[:, action] - W[:, runner])
            # ... and over ALL the alternatives the head weighed, each by its probability
            # (q_j = p_j / (1 - p_bid)): the first-order decomposition of the bid's log-odds.
            # When the runner-up has ~0%, the runner-up margin is an artefact of one arbitrary
            # call and can turn a real reason negative
            finite = np.isfinite(co_masked)
            if finite.any():
                co_p = np.where(finite, np.exp(np.where(finite, co_masked - co_masked[finite].max(), 0.0)), 0.0)
                q = co_p / co_p.sum()
                contrib_alt = concept_acts * (W[:, action] - W @ q)
            else:  # the only legal call (Pass after 7NT redoubled): nothing to weigh it against
                contrib_alt = np.zeros_like(contrib)
            keep = [i for i in range(len(ck.hard_concepts)) if concept_acts[i] >= 0.5 or gt.get(ck.hard_concepts[i])]
            keep.sort(key=lambda i: -max(abs(contrib[i]), abs(contrib_alt[i])))
            concept_evidence = [{"name": ck.hard_concepts[i], "act": round(float(concept_acts[i]), 3),
                                 "true": bool(gt.get(ck.hard_concepts[i])),
                                 "contribution": round(float(contrib[i]), 3),
                                 "contribution_alt": round(float(contrib_alt[i]), 3)} for i in keep[:20]]
            runner_up = {"bid": BIDS[runner], "p": round(float(probs[runner]), 3)}

    frame = build_frame(deal_id, step, seat, partner, opponents, dealer, vulnerability, auction_before, bid,
                        probs[action], hand, top_concepts, symbolic_justification=symbolic_justification,
                        trust=trust, counterfactual_cards=counterfactual_cards, concept_flip=concept_flip)
    if concept_evidence is not None:
        frame["concept_evidence"] = concept_evidence
        frame["runner_up"] = runner_up
    return frame


def generate_frames(model_path: str, dds_path: str, num_deals: int, out_path: str,
                     concept_threshold: float = 0.05, top_k: int = 8, seed_offset: int = 0,
                     enrich: bool = False, random_vulnerability: bool = False) -> int:
    """Walks `num_deals` real deals against the checkpoint's own greedy
    policy, writing one frame per bidding decision to `out_path` (JSONL).
    Returns the number of frames written. `top_concepts` are the
    checkpoint's OWN concept predictions, not ground truth: the signal the
    explainer is given at inference.

    `enrich=True` also runs explain/'s symbolic justification, trust flag,
    card and concept-flip counterfactuals at every decision (as
    explain/walkthrough.py does for one deal); slower, since the card
    counterfactuals need extra batched forward passes."""
    ck = load_checkpoint(model_path)
    env = BridgeBidding(dds_results_table_path=dds_path)
    _forward = jax.jit(lambda obs_b: ck.net.apply(ck.params, obs_b))
    _ = _forward(np.zeros((1, 480), np.float32))  # warm up the JIT cache once
    # pgx's init/step/observe are JAX functions: run eagerly they dominate this loop's time
    env_init, env_step, env_observe = jax.jit(env.init), jax.jit(env.step), jax.jit(env.observe)

    enricher = Enricher(ck)

    n_frames = 0
    with open(out_path, "w") as f:
        for deal_id in range(num_deals):
            rng = jax.random.PRNGKey(deal_id + seed_offset)
            state = env_init(rng)
            if not random_vulnerability:
                # The WBridge5 SL data is all non-vulnerable (OpenSpiel's bridge
                # game has no vulnerability setting), so an SL checkpoint has never
                # seen a vulnerable observation. PPO checkpoints train on pgx's
                # random vulnerability: pass random_vulnerability=True for those.
                state = state.replace(_vul_NS=jnp.bool_(False), _vul_EW=jnp.bool_(False))
                state = state.replace(observation=env_observe(state, state.current_player))
            auction_before: List[Dict] = []
            # Dealer, vulnerability and seats come from the pgx state itself.
            # pgx draws dealer/vulnerability at random and SHUFFLES player ids
            # across seats (state._shuffled_players), so neither deal_id % 4
            # nor current_player % 4 is a seat.
            # The auction runs clockwise from the dealer, so the seat to act
            # is (dealer + step) % 4 (pgx seats: 0=N, 1=E, 2=S, 3=W).
            dealer_int = int(state._dealer)
            dealer = SEAT2CHAR[dealer_int]
            vulnerability = {(False, False): "None", (True, False): "NS", (False, True): "EW",
                             (True, True): "Both"}[(bool(state._vul_NS), bool(state._vul_EW))]
            step = 0

            while not bool(np.asarray(state.terminated)) and step < 30:
                player = int(state.current_player)
                obs = np.asarray(env_observe(state, player), np.float32)[:480]
                legal = np.asarray(state.legal_action_mask, bool)

                out = {k: np.asarray(v[0]) for k, v in _forward(obs[None]).items()}
                seat = SEAT2CHAR[(dealer_int + step) % 4]
                frame = decision_frame(enricher, obs, legal, out, deal_id, step, seat, dealer, dealer_int,
                                       vulnerability, auction_before, top_k=top_k,
                                       concept_threshold=concept_threshold, enrich=enrich)
                bid, action = frame["bid"], BIDS.index(frame["bid"])
                f.write(json.dumps(frame, ensure_ascii=False) + "\n")
                n_frames += 1

                auction_before.append({"seat": seat, "bid": bid})
                rng, k = jax.random.split(rng)
                state = env_step(state, action, key=k)
                step += 1

    return n_frames


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dds_path", required=True)
    p.add_argument("--num_deals", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--random_vulnerability", action="store_true",
                    help="Keep pgx's random vulnerability (PPO checkpoints). Default: non-vulnerable "
                         "deals, matching the SL training data.")
    p.add_argument("--enrich", action="store_true",
                    help="Also compute explain/'s symbolic justification, trust flag, "
                         "card counterfactuals, and concept-flip counterfactuals per frame "
                         "(slower -- extra forward passes per decision)")
    args = p.parse_args()
    n = generate_frames(args.model_path, args.dds_path, args.num_deals, args.out, enrich=args.enrich,
                        random_vulnerability=args.random_vulnerability)
    print(f"[frames] wrote {n} frames from {args.num_deals} deals to {args.out}")
