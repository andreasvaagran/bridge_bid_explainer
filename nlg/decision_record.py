"""One explained record per decision, shown by the terminal (explain_deals) and the GUI: explanation,
reasons and contributions, trust flag, concept flips, card counterfactuals and statistics, Integrated Gradients,
the rule-engine justification and the call in the agent's own convention. `Bidder` wraps a concept checkpoint or
an original brl network; the brl networks are explained post hoc. For concept models IG is taken through the
concepts, since the bottleneck stop-gradients them."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from concepts import gloss_for_concept
from concepts.hand import RANKS, SUIT_ORDER, card_bit_index, decode_hand
from explain.counterfactual import card_counterfactuals, softmax_masked
from explain.rule_justification import render_concept_justification, rule_conditions
from evaluation.convention_ranges import own_convention
from nlg.consistency_checks import check_explanation
from nlg.explanation_frames import BIDS, Enricher, build_frame, decision_frame
from nlg.prompt import check_style
from nlg.explanation_inputs import (auction_from_frame, bid_category, contribution_key, deviation_notes,
                           hand_dependent_concepts, hand_from_str, _gt, select_reasoned_concepts)
from explain.card_swap_stats import card_swap_stats, summary_sentence as card_stats_sentence
from nlg.teacher import explain as teacher_explain, reason_clauses, situation

SEATS = "NESW"
OBS_SIZE = 480


class Bidder:
    """A bidding model with one interface, whichever kind it is."""

    def __init__(self, path: str, label: Optional[str] = None, activation: str = "relu",
                 model_type: str = "DeepMind", ig_steps: int = 32):
        self.path = str(path)
        with open(self.path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "bottleneck_mode" in obj:
            from training.checkpoint import load_checkpoint
            self.ck = load_checkpoint(self.path)
            self.kind, self.mode = "concept", self.ck.bottleneck_mode
            self.net, self.params = self.ck.net, self.ck.params
            self.enricher = Enricher(self.ck)
            W = jnp.asarray(self.params["policy_head"]["w"])
            b = jnp.asarray(self.params["policy_head"]["b"])
            joint = self.mode == "joint"

            def logit_through_concepts(o, action):
                out = self.net.apply(self.params, o[None])
                inp = jnp.concatenate([out["hidden"], out["concept_acts"]], -1) if joint else out["concept_acts"]
                return (inp @ W + b)[0, action]
            target = logit_through_concepts
        else:
            from src.models import make_forward_pass
            fwd = make_forward_pass(activation=activation, model_type=model_type)
            params = obj["params"] if isinstance(obj, dict) and "params" in obj else obj

            class _Net:  # build_net's dict-shaped output, for the shared code paths
                @staticmethod
                def apply(p, o):
                    return {"policy_logits": fwd.apply(p, o)[0]}
            self.ck, self.kind, self.mode, self.enricher = None, "upstream", None, None
            self.net, self.params = _Net(), params

            def target(o, action):
                return self.net.apply(self.params, o[None])["policy_logits"][0, action]
        self._fwd = jax.jit(lambda o: self.net.apply(self.params, o))
        grad = jax.grad(target)
        alphas = jnp.linspace(0.0, 1.0, ig_steps + 1)[1:]

        @jax.jit
        def ig(o, action):  # Integrated Gradients, all-zero baseline (explain/integrated_gradients.py)
            grads = jax.vmap(lambda a: grad(a * o, action))(alphas)
            return o * grads.mean(0)
        self._ig = ig
        self.label = label or self._default_label()

    def _default_label(self) -> str:
        p = Path(self.path)
        if self.kind == "upstream":
            return f"brl {p.stem}"
        stage = "PPO" if "ppo" in p.parts or "rl_params" in p.parts else "SL"
        return f"{self.mode} {stage}"

    def outputs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        out = self._fwd(np.asarray(obs, np.float32)[None, :OBS_SIZE])
        return {k: np.asarray(v[0]) for k, v in out.items()}

    def policy_logits_batch(self, obs_batch) -> np.ndarray:
        return np.asarray(self._fwd(np.asarray(obs_batch, np.float32))["policy_logits"])

    def outputs_batch(self, obs_batch) -> Dict[str, np.ndarray]:
        out = self._fwd(np.asarray(obs_batch, np.float32)[:, :OBS_SIZE])
        return {k: np.asarray(v) for k, v in out.items()}

    def act(self, obs: np.ndarray, legal: np.ndarray) -> int:
        return int(np.argmax(np.where(legal, self.outputs(obs)["policy_logits"], -np.inf)))

    def card_attribution(self, obs: np.ndarray, action: int) -> List[Dict[str, Any]]:
        o = np.asarray(obs, np.float32)[:OBS_SIZE]
        vals = np.asarray(self._ig(o, action))[-52:]
        index_to_card = {card_bit_index(s + r): s + r for s in SUIT_ORDER for r in RANKS}
        return sorted(({"card": index_to_card[i], "value": round(float(vals[i]), 4)}
                       for i in range(52) if o[-52:][i] > 0), key=lambda x: -abs(x["value"]))


def _upstream_frame(bidder: Bidder, obs, legal, out, deal_id, step, seat, dealer, dealer_int, vulnerability,
                    auction_before, counterfactuals: bool) -> Dict[str, Any]:
    """A frame for a network without concepts: the bid, the hand and the
    auction, card counterfactuals (they only need the policy) and the rule
    engine's justification of what the call means for this hand."""
    probs = softmax_masked(out["policy_logits"], legal)
    action = int(np.argmax(probs))
    bid = BIDS[action]
    hand = decode_hand(obs)
    i = SEATS.index(seat)
    cf = (card_counterfactuals(obs, hand, legal, action, probs, bidder.policy_logits_batch, bid, BIDS)
          if counterfactuals else None)
    frame = build_frame(deal_id, step, seat, SEATS[(i + 2) % 4], [SEATS[(i + 1) % 4], SEATS[(i + 3) % 4]],
                        dealer, vulnerability, auction_before, bid, probs[action], hand, [],
                        counterfactual_cards=cf)
    gt, dep = _gt(frame), hand_dependent_concepts(frame)
    auction = auction_from_frame(frame)
    held = [n for n, v in gt.items() if v and dep.get(n) and not n.startswith("bid_is_in_")]
    for name in held:
        try:
            frame["symbolic_justification"] = render_concept_justification(name, auction, hand)
            break
        except ValueError:
            continue
    # The network has no concepts, so the teacher's reasons come from the RULE
    # ENGINE: the hand-based concepts that hold for this call (equal weight, in
    # vocabulary order). Marked as such; the network's own evidence is the
    # counterfactuals and the card attribution.
    frame["concept_evidence"] = [{"name": n, "act": 1.0, "true": True, "contribution": 1.0} for n in held[:3]]
    frame["concept_evidence_source"] = "rule engine (the network has no concept layer)"
    return frame


def analyze(bidder: Bidder, obs, legal, auction_before: List[Dict[str, str]], seat: str, dealer_int: int,
            vulnerability: str = "None", deal_id: int = 0, step: int = 0, counterfactuals: bool = True,
            attribution: bool = True) -> Dict[str, Any]:
    """The full explained record of `bidder`'s call at this position."""
    obs = np.asarray(obs, np.float32)[:OBS_SIZE]
    legal = np.asarray(legal, bool)
    out = bidder.outputs(obs)
    dealer = SEATS[dealer_int]
    if bidder.kind == "concept":
        frame = decision_frame(bidder.enricher, obs, legal, out, deal_id, step, seat, dealer, dealer_int,
                               vulnerability, auction_before, counterfactuals=counterfactuals)
    else:
        frame = _upstream_frame(bidder, obs, legal, out, deal_id, step, seat, dealer, dealer_int, vulnerability,
                                auction_before, counterfactuals)
    action = BIDS.index(frame["bid"])
    probs = softmax_masked(out["policy_logits"], legal)
    text = teacher_explain(frame)
    problems = check_explanation(frame, text) + ([check_style(text)] if check_style(text) else [])
    sit = situation(frame)
    reasons = select_reasoned_concepts(frame)

    dep = hand_dependent_concepts(frame)
    notes = _concept_notes(frame)
    if bidder.kind == "concept":
        # the contribution the reasons were ranked by (nlg.explanation_inputs.contribution_key): over the
        # runner-up, or over all the alternatives when the runner-up had under 5%
        ckey = contribution_key(frame)
        concepts = [{"name": e["name"], "gloss": gloss_for_concept(e["name"]), "activation": e["act"],
                     "true": _gt(frame).get(e["name"], e["true"]), "contribution": e[ckey],
                     "contribution_runner_up": e["contribution"], "reason": e["name"] in reasons,
                     "hand_based": bool(dep.get(e["name"])), "note": notes.get(e["name"])}
                    for e in frame["concept_evidence"] if not e["name"].startswith("bid_is_in_")]
        concepts.sort(key=lambda x: -abs(x["contribution"]))
    else:  # no concept layer: what the rule engine says holds for this call with this hand
        gt = _gt(frame)
        concepts = [{"name": n, "gloss": gloss_for_concept(n), "activation": None, "true": True,
                     "contribution": None, "reason": False, "hand_based": True, "note": notes.get(n)}
                    for n, v in gt.items() if v and dep.get(n) and not n.startswith("bid_is_in_")]
    flips = frame.get("concept_flip") or {}
    order = np.argsort(-probs)
    try:
        stats = card_swap_stats(obs, legal, action, bidder.outputs_batch, BIDS,
                                concept_names=list(bidder.ck.hard_concepts) if bidder.kind == "concept" else None,
                                glosses=gloss_for_concept, focus_concepts=reasons)
        stats["sentence"] = card_stats_sentence(stats)
    except Exception as e:  # never lose the explanation over the statistics
        stats = {"error": f"{type(e).__name__}: {e}"}
    record = {
        "seat": seat, "call": frame["bid"], "model": bidder.label, "model_kind": bidder.kind,
        "bottleneck_mode": bidder.mode,
        "hand": frame["hand_str"], "hcp": frame["hcp"],
        "shape": "-".join(str(len(p.split(":")[1].replace("-", ""))) for p in frame["hand_str"].split()),
        "explanation": text, "explanation_source": "teacher (rule-grounded)", "checks_passed": not problems,
        "check_problems": problems,
        "situation": bid_category(frame).replace("_", " "), "meaning": sit["meaning"], "key_facts": sit["facts"],
        "unusual": deviation_notes(frame),
        "probabilities": [{"call": BIDS[i], "p": round(float(probs[i]), 4)} for i in order[:8] if legal[i]],
        "legal_calls": [BIDS[i] for i in np.flatnonzero(legal)],
        "runner_up": frame.get("runner_up"),
        "reason_clauses": reason_clauses(frame),
        "hand_summary": hand_summary(frame),
        "concepts": concepts, "reasons": reasons,
        "concept_source": frame.get("concept_evidence_source", "the model's own concept layer"),
        "trust": frame.get("trust"),
        "concept_flip": {k: ([{"name": n, "gloss": gloss_for_concept(n)} for n in v] if isinstance(v, list) else v)
                         for k, v in flips.items()},
        "counterfactuals": frame.get("counterfactual_cards") or [],
        "card_stats": stats,
        "contrast_with": ("the alternatives" if bidder.kind == "concept" and contribution_key(frame) == "contribution_alt"
                          else (frame.get("runner_up") or {}).get("bid")),
        "card_attribution": bidder.card_attribution(obs, action) if attribution else [],
        "rule_justification": frame.get("symbolic_justification"),
        "system_evidence": _system_evidence(frame),
        # the call in the agent's OWN convention, learned from its auctions (evaluation/convention_ranges.py);
        # cards exist for brl's networks, whose RL versions left WBridge5's system
        "own_convention": own_convention(bidder.path, frame),
        "frame": frame,
    }
    return record


def score_breakdown(bidder: Bidder, obs, legal, vs: Optional[str] = None, frame: Optional[Dict[str, Any]] = None,
                    top: int = 10) -> Dict[str, Any]:
    """The policy layer at work (the interface's Weights tab): every legal bid's score z_j and probability, and
    the bid made against `vs` -- a bid, "alt" for all the alternatives weighed by their probability, or None for
    the runner-up (the alternatives when it has under 5%, as the reasons are ranked). The score difference splits
    exactly: z_a - z_vs = sum_k c_k (W[k,a] - W[k,vs]) + (b_a - b_vs), plus in the joint mode the part that comes
    from the hidden layer. `frame` marks the concepts the rules confirm for this hand and gives each shown concept
    its rule, condition by condition (explain.rule_justification.rule_conditions)."""
    from nlg.explanation_inputs import RUNNER_UP_MIN_P
    obs = np.asarray(obs, np.float32)[:OBS_SIZE]
    legal = np.asarray(legal, bool)
    out = bidder.outputs(obs)
    idx = np.flatnonzero(legal)
    res: Dict[str, Any] = {"model": bidder.label, "kind": bidder.kind, "mode": bidder.mode}
    if bidder.kind != "concept":  # brl's networks: the scores, nothing to split them into
        z = np.asarray(out["policy_logits"], np.float64)
        p = softmax_masked(z, legal)
        order = idx[np.argsort(-p[idx])]
        res.update(call=BIDS[int(order[0])],
                   bids=[{"call": BIDS[j], "score": round(float(z[j]), 3), "p": round(float(p[j]), 4)} for j in order])
        return res
    W = np.asarray(bidder.params["policy_head"]["w"], np.float64)
    b = np.asarray(bidder.params["policy_head"]["b"], np.float64)
    c = np.asarray(out["concept_acts"], np.float64)
    if bidder.mode == "joint":  # the policy reads [hidden, concepts]
        h = np.asarray(out["hidden"], np.float64)
        Wc, hid = W[len(h):], h @ W[:len(h)]
    else:
        Wc, hid = W, np.zeros(len(b))
    conc = c @ Wc
    z = conc + hid + b  # = the network's policy logits, recomputed so that the parts add up exactly
    p = softmax_masked(z, legal)
    order = idx[np.argsort(-p[idx])]
    a = int(order[0])
    others = [int(j) for j in order[1:]]
    if vs is None:
        vs = None if not others else "alt" if p[others[0]] < RUNNER_UP_MIN_P else BIDS[others[0]]
    if vs == "alt" and others:
        q = np.zeros(len(b))
        q[others] = p[others] / p[others].sum()
        w_vs, b_vs, hid_vs, z_vs, p_vs = Wc @ q, b @ q, hid @ q, z[others] @ q[others], 1.0 - p[a]
    elif vs in BIDS and BIDS.index(vs) in others:
        j = BIDS.index(vs)
        w_vs, b_vs, hid_vs, z_vs, p_vs = Wc[:, j], b[j], hid[j], z[j], p[j]
    elif vs is None:  # the only legal bid
        w_vs, b_vs, hid_vs, z_vs, p_vs = Wc[:, a], b[a], hid[a], z[a], 0.0
    else:
        raise ValueError(f"{vs} is not another legal bid here")
    d = c * (Wc[:, a] - w_vs)
    names = list(bidder.ck.hard_concepts)
    gt = _gt(frame) if frame else {}
    ctx = (auction_from_frame(frame), hand_from_str(frame["hand_str"])) if frame else None
    shown = sorted(range(len(names)), key=lambda k: -abs(d[k]))[:top]
    rest = [k for k in range(len(names)) if k not in set(shown)]
    r3 = lambda x: round(float(x), 3)  # noqa: E731
    res.update(
        call=BIDS[a], vs=vs, p=round(float(p[a]), 4), p_vs=round(float(p_vs), 4), margin=r3(z[a] - z_vs),
        bids=[{"call": BIDS[j], "score": r3(z[j]), "p": round(float(p[j]), 4), "concepts": r3(conc[j]),
               "hidden": r3(hid[j]) if bidder.mode == "joint" else None, "constant": r3(b[j])} for j in order],
        rows=[{"name": names[k], "gloss": gloss_for_concept(names[k]), "value": r3(c[k]), "w_call": r3(Wc[k, a]),
               "w_vs": r3(w_vs[k]), "contribution": r3(d[k]), "true": bool(gt.get(names[k])) if gt else None,
               "descriptor": names[k].startswith("bid_is_in_"),
               "points": [r3(c[k] * Wc[k, j]) for j in order],  # what the concept adds to each bid's score
               "rule": rule_conditions(names[k], *ctx) if ctx else None}
              for k in shown],
        rest={"n": len(rest), "contribution": r3(d[rest].sum())},
        constant=r3(b[a] - b_vs),
        hidden=r3(hid[a] - hid_vs) if bidder.mode == "joint" else None)
    return res


def hand_summary(frame: Dict[str, Any]) -> Dict[str, Any]:
    """The hand's numbers the concepts are computed from, for the interface's strip:
    HCP, points, losers, quick tricks, stoppers in the opponents' suits, support for
    partner's suit, the Rule of 20 at an opening decision."""
    import random as _random
    from concepts.predicates_hand import losing_trick_count, quick_tricks, rule_of_20_satisfied, support_points, wb5_points
    from concepts.hand import is_stopper
    from nlg.teacher import Ctx, _own_suit
    c = Ctx(frame, _random.Random(0))
    h = c.h
    out = {"hcp": h.hcp, "points": wb5_points(h), "losers": losing_trick_count(h), "quick_tricks": quick_tricks(h),
           "balanced": bool(h.balanced), "stoppers": {s: is_stopper(h.cards[s]) for s in c.opp_suits}}
    ps = c.partner_suit
    if ps:
        out["support"] = {"suit": ps, "length": h.lengths[ps], "support_points": support_points(h, ps),
                          "agreed": bool(_own_suit(c, ps))}
    if all(b["bid"] in ("P", "Pass") for b in c.prior):
        out["rule_of_20"] = bool(rule_of_20_satisfied(h))
    return out


def _concept_notes(frame: Dict[str, Any]) -> Dict[str, str]:
    """Concepts whose textbook name WBridge5's games contradict in THIS
    situation. The model was trained on the rules' names, so they cannot be
    renamed without retraining; the view says what the call really is
    (notes/wbridge5_system.md), as the explanation does."""
    from nlg.teacher import _conventional_call, Ctx
    import random as _random
    c = Ctx(frame, _random.Random(0))
    notes = {}
    if frame["bid"] not in ("X", "Dbl") and c.gt.get("bid_is_check_back"):
        notes["bid_is_check_back"] = "WBridge5 plays this bid as natural (four or more cards), not new minor forcing"
    if frame["bid"] in ("X", "Dbl"):
        from nlg.double_meaning import leave_in_rate
        from nlg.teacher import _not_for_penalties, meaning_phrase
        r = leave_in_rate(frame)
        if _not_for_penalties(c):  # a cuebid / support / competitive / lead-directing double
            notes["bid_is_penalty_double"] = notes["bid_double_with_length_in_their_suit"] = (
                f"in WBridge5's system this is {meaning_phrase(c)}, not a double for penalties")
        elif r and r[0] <= 0.3:
            notes["bid_is_penalty_double"] = notes["bid_double_with_length_in_their_suit"] = (
                f"WBridge5's partner took this double out {1 - r[0]:.0%} of the time: not a penalty double here")
        if r and r[0] >= 0.6:
            notes["bid_is_takeout_double"] = notes["bid_is_negative_double"] = (
                f"WBridge5's partner left this double in {r[0]:.0%} of the time: a penalty double here")
    conv = _conventional_call(c)
    if conv in ("nt_overcall_stayman", "nt_overcall_transfer"):
        for n in ("bid_is_cuebid", "bid_shows_4_plus_cards", "bid_shows_5_plus_cards", "bid_competes_for_partscore"):
            notes[n] = ("WBridge5 plays Stayman and transfers after partner's 1NT overcall: this is "
                        + ("Stayman" if conv == "nt_overcall_stayman" else "a transfer") + ", not a natural bid or a cuebid")
    if conv in ("nt_forcing_new_suit", "preemptive_jump_raise", "nt_minor_puppet"):
        why = {"nt_forcing_new_suit": "forcing opposite partner's 1NT, not a preempt or a partscore bid",
               "preemptive_jump_raise": "a preemptive jump raise (5-10 HCP) in WBridge5's games",
               "nt_minor_puppet": "an artificial relay to 3C, not spades"}[conv]
        for n in ("bid_shows_preemptive_intent", "bid_competes_for_partscore", "bid_shows_4_plus_cards",
                  "bid_shows_5_plus_cards", "bid_shows_major_suit"):
            notes.setdefault(n, why)
    return notes


def _system_evidence(frame: Dict[str, Any]) -> List[str]:
    """What WBridge5's own games say about this kind of call (the data the
    teacher's reading of it rests on): for a double, how often partner left it
    in, in this exact auction shape (nlg.double_meaning)."""
    out = []
    if frame["bid"] in ("X", "Dbl"):
        from nlg.double_meaning import leave_in_rate
        r = leave_in_rate(frame)
        if r:
            rate, n = r
            kind = ("so it is read as a penalty double" if rate >= 0.6 else
                    "so it is read as a takeout / competitive double" if rate <= 0.3 else
                    "an optional double: partner may pass or bid")
            out.append(f"In WBridge5's games, partner left a double in this auction in {rate:.0%} of {n:,} "
                       f"cases, {kind}.")
    return out


def format_record(r: Dict[str, Any], top_concepts: int = 5) -> str:
    """A terminal rendering of an explained decision (nlg/explain_deals.py --xai)."""
    L = [f"{r['seat']}: {r['call']}  [{r['model']}]  -- {r['situation']}",
         f"  Explanation: {r['explanation']}"]
    if r.get("llm_explanation"):
        L.append(f"  Qwen: {r['llm_explanation']}")
    alts = ", ".join(f"{a['call']} {a['p']:.0%}" for a in r["probabilities"][:4])
    L.append(f"  Bid probabilities: {alts}")
    if r["trust"]:
        L.append(f"  Trust: {'concepts alone pick the same call' if r['trust']['agree'] else 'concepts alone would bid ' + str(r['trust']['concept_only_bid'])}")
    shown = [c for c in r["concepts"] if c["contribution"] is not None][:top_concepts]
    if shown:
        L.append("  Concepts behind the call (contribution over the runner-up):")
        L += [f"    {c['contribution']:+.2f}  {'*' if c['reason'] else ' '} {c['gloss']}" + ("" if c["true"] else "  (model believes, rules disagree)")
              for c in shown]
    elif r["concepts"]:
        L.append("  Rules that hold for this call and hand: " + "; ".join(c["gloss"] for c in r["concepts"][:4]))
    if r["counterfactuals"]:
        L.append("  What if: " + r["counterfactuals"][0]["sentence"])
    if r["card_attribution"]:
        top = ", ".join(f"{a['card']} {a['value']:+.2f}" for a in r["card_attribution"][:4])
        L.append(f"  Cards that mattered most (Integrated Gradients): {top}")
    L += [f"  System data: {x}" for x in r.get("system_evidence") or []]
    if r.get("own_convention"):
        L.append(f"  Own convention: {r['own_convention']['text']}")
    return "\n".join(L)  # unusual-call notes are already the explanation's last sentence
