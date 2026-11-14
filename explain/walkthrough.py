#!/usr/bin/env python3
"""Plays one deal against a checkpoint and prints every explanation method at each decision: a quick
end-to-end check after a change to explain/ or the network."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding

from concepts import AuctionContext, create_sayc_state_authority, decode_hand
from explain.integrated_gradients import integrated_gradients_cards
from explain.counterfactual import card_counterfactuals, flip_shortlist_caveat, search_minimal_flip, softmax_masked
from explain.rule_justification import render_concept_justification
from explain.trust_flag import compute_trust_flag
from training.checkpoint import load_checkpoint

BIDS = ["P", "X", "XX", "1C", "1D", "1H", "1S", "1NT", "2C", "2D", "2H", "2S", "2NT",
        "3C", "3D", "3H", "3S", "3NT", "4C", "4D", "4H", "4S", "4NT", "5C", "5D", "5H",
        "5S", "5NT", "6C", "6D", "6H", "6S", "6NT", "7C", "7D", "7H", "7S", "7NT"]
SUIT_SYMBOL = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
SEAT2CHAR = {0: "N", 1: "E", 2: "S", 3: "W"}


def format_hand(hand) -> str:
    return " ".join(f"{SUIT_SYMBOL[s]}{''.join(sorted(hand.cards[s], key=lambda r: '23456789TJQKA'.index(r), reverse=True)) or '-'}"
                     for s in ("S", "H", "D", "C"))


def run(args):
    ck = load_checkpoint(args.model_path)
    print(f"[model] bottleneck_mode={ck.bottleneck_mode} model_type={ck.model_type} "
          f"activation={ck.activation} num_hard={len(ck.hard_concepts)} has_value={ck.has_value}")

    has_concept_only = "concept_only_policy_head" in ck.params
    W = np.asarray(ck.params["concept_only_policy_head"]["w"]) if has_concept_only else None
    b = np.asarray(ck.params["concept_only_policy_head"]["b"]) if has_concept_only else None

    state_auth = create_sayc_state_authority()
    env = BridgeBidding(dds_results_table_path=args.dds_path)

    rng = jax.random.PRNGKey(args.seed)
    state = env.init(rng)
    # non-vulnerable, like every deal in the SL training data (see nlg/explanation_frames.py)
    state = state.replace(_vul_NS=jnp.bool_(False), _vul_EW=jnp.bool_(False))
    auction_before = []
    step = 1

    print(f"\n{'='*70}\n  Walkthrough | seed={args.seed}\n{'='*70}\n")

    while not bool(np.asarray(state.terminated)):
        player = int(state.current_player)
        obs = np.asarray(env.observe(state, player), np.float32)[:480]
        legal = np.asarray(state.legal_action_mask, bool)

        out = ck.net.apply(ck.params, obs[None])
        policy_logits = np.asarray(out["policy_logits"][0])
        concept_acts = np.asarray(out["concept_acts"][0])
        probs = softmax_masked(policy_logits, legal)
        action = int(np.argmax(probs))
        bid = BIDS[action]
        confidence = float(probs[action])

        hand = decode_hand(obs)
        seat = SEAT2CHAR[(int(state._dealer) + step - 1) % 4]  # pgx shuffles player ids across seats; see nlg/explanation_frames.py

        print(f"Step {step} | {seat}: {bid}  (p={confidence:.2f})")
        print(f"  Hand: {format_hand(hand)}  HCP={hand.hcp}  Balanced={hand.balanced}")

        auction = AuctionContext(bids=[b["bid"] for b in auction_before], bid=bid, dealer=0)
        true_concepts = [n for n, v in state_auth.classify(auction, hand).items() if v]
        print(f"  Ground-truth concepts ({len(true_concepts)}):",
              (", ".join(true_concepts[:5]) + (", ..." if len(true_concepts) > 5 else "")) or "(none)")
        focus = true_concepts[0] if true_concepts else None

        if has_concept_only:
            concept_only_logits = np.asarray(out["concept_only_logits"][0])
            trust = compute_trust_flag(policy_logits, concept_only_logits, legal, BIDS, ck.bottleneck_mode)
            print(f"  Trust flag: {trust.explanation()}")

            descriptor_idx = [i for i, c in enumerate(ck.hard_concepts) if c.startswith("bid_is_in_")]
            necessary = search_minimal_flip(concept_acts, legal, W, b, flip_active=True, k_max=3,
                                            exclude=descriptor_idx)
            sufficient = search_minimal_flip(concept_acts, legal, W, b, flip_active=False, k_max=3,
                                             exclude=descriptor_idx)
            nec_names = [ck.hard_concepts[i] for i in necessary] if necessary else None
            suf_names = [ck.hard_concepts[i] for i in sufficient] if sufficient else None
            print(f"  Concept flip — necessary: {nec_names}{flip_shortlist_caveat(nec_names)}")
            print(f"  Concept flip — sufficient: {suf_names}{flip_shortlist_caveat(suf_names)}")
            # justify the true concept that contributes most to THIS bid through the concepts->bid head
            idx = {c: i for i, c in enumerate(ck.hard_concepts)}
            restates = ("bid_is_pass", "bid_is_double", "bid_shows_major_suit", "bid_shows_minor_suit")
            cands = [c for c in true_concepts if c in idx and not c.startswith("bid_is_in_")]
            cands = [c for c in cands if c not in restates] or cands  # a reason, not the call restated
            if cands:
                focus = max(cands, key=lambda c: float(concept_acts[idx[c]]) * float(W[idx[c], action]))

        if focus:
            just = render_concept_justification(focus, auction, hand)
            print(f"  Symbolic justification ({focus}): {just}")

        def policy_logits_fn(obs_batch):
            return np.asarray(ck.net.apply(ck.params, obs_batch)["policy_logits"])

        cfs = card_counterfactuals(obs, hand, legal, action, probs, policy_logits_fn, bid, BIDS)
        for cf in cfs:
            print(f"  Counterfactual: {cf['sentence']}")

        if args.attribution:
            attrs = integrated_gradients_cards(ck.net, ck.params, obs, action, m_steps=args.ig_steps)
            top3 = attrs[:3]
            print("  Top card attributions (Integrated Gradients):",
                  ", ".join(f"{a.card}{a.attribution:+.3f}" for a in top3))

        print("-" * 70)

        auction_before.append({"seat": seat, "bid": bid})
        rng, k = jax.random.split(rng)
        state = env.step(state, action, key=k)
        step += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dds_path", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attribution", action="store_true", help="Also compute Integrated Gradients per step (slower)")
    p.add_argument("--ig_steps", type=int, default=30)
    run(p.parse_args())
